"""AprilTag tag36h11 detection (IDs 0–4) — drone position and orientation (TOP + SIDE)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

from config import AprilTagConfig, CameraConfig
from fiducial_detector import FiducialDetection, FiducialDetector

# Lab frame (navigation, geofence): x=left (+), y=forward (+), z=up (+).
# Tag body layout in solvePnP uses X=forward, Y=lateral on the drone.
TOP_MARKER_IDS = frozenset({0, 1, 2, 3, 4})
SIDE_MARKER_IDS = frozenset({1, 2, 3, 4})


@dataclass(frozen=True)
class MarkerOverlay:
    """Marker outline in frame pixels (for preview)."""

    marker_id: int
    corners: np.ndarray  # (4, 2) float
    in_layout: bool = True


@dataclass
class TopPoseObservation:
    ok: bool
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0
    yaw_rad: float = 0.0
    pitch_rad: float = 0.0
    roll_rad: float = 0.0
    reference_marker_id: int | None = None
    pose_quality: float = 0.0
    markers_seen: tuple[int, ...] = ()
    marker_overlays: Tuple[MarkerOverlay, ...] = ()
    centroid_u_px: float = 0.0
    centroid_v_px: float = 0.0
    hint: Optional[str] = None


@dataclass
class SideCorrectionObservation:
    ok: bool
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0
    yaw_rad: float = 0.0
    # Tag facing angle toward side camera: 0 = marker faces camera directly.
    # Increases as the drone yaws and the marker leaves the field of view.
    yaw_facing_rad: float = 0.0
    marker_id: int | None = None
    image_area_px: float = 0.0
    # Marker center in SIDE frame [px] — for holding the same image point.
    centroid_u_px: float = 0.0
    centroid_v_px: float = 0.0
    pose_quality: float = 0.0
    markers_seen: tuple[int, ...] = ()
    marker_overlays: Tuple[MarkerOverlay, ...] = ()
    hint: Optional[str] = None

    @property
    def yaw_correction_rad(self) -> float:
        return self.yaw_rad

    @property
    def z_from_marker_m(self) -> float:
        return self.z_m


@dataclass
class _MarkerPose:
    id: int
    rvec: np.ndarray
    tvec: np.ndarray
    corners: np.ndarray
    family: str


def _default_intrinsics(w: int, h: int, fov_deg: float = 58.0) -> Tuple[np.ndarray, np.ndarray]:
    fx = (w / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    fy = fx
    cx, cy = w / 2.0, h / 2.0
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64)
    return camera_matrix, dist


def _load_intrinsics(path: str, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    p = Path(path)
    if not p.is_file():
        return _default_intrinsics(w, h)
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    return camera_matrix, dist


def _rotation_to_euler_zyx(rmat: np.ndarray) -> tuple[float, float, float]:
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(rmat[2, 1], rmat[2, 2])
        pitch = np.arctan2(-rmat[2, 0], sy)
        yaw = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        roll = np.arctan2(-rmat[1, 2], rmat[1, 1])
        pitch = np.arctan2(-rmat[2, 0], sy)
        yaw = 0.0
    return float(yaw), float(pitch), float(roll)


def _unwrap_yaw(yaw: float) -> float:
    return float(np.arctan2(np.sin(yaw), np.cos(yaw)))


def _marker_corners_object_points(center_xyz: np.ndarray, half_side_m: float) -> np.ndarray:
    cx, cy, cz = float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])
    h = half_side_m
    return np.array(
        [
            [cx - h, cy - h, cz],
            [cx + h, cy - h, cz],
            [cx + h, cy + h, cz],
            [cx - h, cy + h, cz],
        ],
        dtype=np.float64,
    )


def _single_marker_object_points(marker_length_m: float) -> np.ndarray:
    half = float(marker_length_m) * 0.5
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def _estimate_marker_pose(
    corners: np.ndarray,
    marker_length_m: float,
    camera_matrix: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    img_pts = corners.reshape(4, 2).astype(np.float64)
    obj_pts = _single_marker_object_points(marker_length_m)
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        camera_matrix,
        dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        return None
    return rvec.reshape(3), tvec.reshape(3)


class AprilTagDetector:
    def __init__(self, tag_cfg: AprilTagConfig, cam_cfg: CameraConfig) -> None:
        self._tag = tag_cfg
        self._cam = cam_cfg
        self._fiducial = FiducialDetector(tag_cfg)
        self._half_marker = float(tag_cfg.marker_length_m) * 0.5
        self._body_centers = self._build_body_centers()
        self._allowed_ids = set(int(x) for x in tag_cfg.marker_ids)
        if tag_cfg.calibration_path:
            self._K, self._dist = _load_intrinsics(tag_cfg.calibration_path, cam_cfg.width, cam_cfg.height)
        else:
            self._K, self._dist = _default_intrinsics(cam_cfg.width, cam_cfg.height)
        self._frame_K_wh: Tuple[int, int] | None = None
        self._active_family = tag_cfg.family
        self._detection_mode = "none"

    @property
    def camera_matrix(self) -> np.ndarray:
        """Intrinsic matrix K (for projecting world points to preview pixels)."""
        return self._K

    @property
    def dist_coeffs(self) -> np.ndarray:
        return self._dist

    def _build_body_centers(self) -> Dict[int, np.ndarray]:
        hf = float(self._tag.layout_half_forward_m)
        hl = float(self._tag.layout_half_lateral_m)
        sz = float(self._tag.marker_side_height_m)
        return {
            0: np.array([0.0, 0.0, 0.0]),
            1: np.array([hf, hl, -sz]),
            2: np.array([hf, -hl, -sz]),
            3: np.array([-hf, -hl, -sz]),
            4: np.array([-hf, hl, -sz]),
        }

    def _dict_hint(self) -> str:
        return (
            f"Family: {self._active_family}. "
            f"Expected IDs: {sorted(self._allowed_ids)}."
        )

    def _ensure_intrinsics(self, w: int, h: int) -> None:
        if self._tag.calibration_path:
            return
        if self._tag.intrinsics_from_frame_size and self._frame_K_wh != (w, h):
            self._K, self._dist = _default_intrinsics(w, h)
            self._frame_K_wh = (w, h)

    def _run_fiducial(self, frame_bgr: np.ndarray) -> FiducialDetection:
        return self._fiducial.detect(frame_bgr)

    def _detect_markers(
        self, frame_bgr: np.ndarray
    ) -> tuple[list[_MarkerPose], list[int], Tuple[MarkerOverlay, ...]]:
        fh, fw = frame_bgr.shape[:2]
        self._ensure_intrinsics(fw, fh)
        det = self._run_fiducial(frame_bgr)
        self._detection_mode = det.method
        all_seen: list[int] = []
        poses: list[_MarkerPose] = []
        overlays: list[MarkerOverlay] = []

        if len(det.ids) == 0:
            return poses, all_seen, tuple(overlays)

        corners, ids = det.corners, det.ids
        family = self._active_family

        for i, mid in enumerate(ids.flatten()):
            marker_id = int(mid)
            all_seen.append(marker_id)
            overlays.append(
                MarkerOverlay(
                    marker_id=marker_id,
                    corners=corners[i].reshape(4, 2).copy(),
                    in_layout=marker_id in self._allowed_ids,
                )
            )
            if marker_id not in self._allowed_ids:
                continue
            pose = _estimate_marker_pose(
                corners[i], self._tag.marker_length_m, self._K, self._dist
            )
            if pose is None:
                continue
            rvec, tvec = pose
            poses.append(
                _MarkerPose(
                    id=marker_id,
                    rvec=rvec,
                    tvec=tvec,
                    corners=corners[i],
                    family=family,
                )
            )

        return poses, sorted(set(all_seen)), tuple(overlays)

    def _origin_from_marker(self, pose: _MarkerPose) -> tuple[np.ndarray, np.ndarray]:
        R, _ = cv2.Rodrigues(pose.rvec.reshape(3, 1))
        p_body = self._body_centers[pose.id]
        t_origin = pose.tvec.reshape(3) - R @ p_body
        return t_origin, R

    def _solve_multi_pnp(self, poses: Iterable[_MarkerPose]) -> Optional[tuple[np.ndarray, np.ndarray]]:
        obj_pts: list[np.ndarray] = []
        img_pts: list[np.ndarray] = []
        for pose in poses:
            if pose.id not in self._body_centers:
                continue
            center = self._body_centers[pose.id]
            obj_pts.append(_marker_corners_object_points(center, self._half_marker))
            img_pts.append(pose.corners.reshape(4, 2).astype(np.float64))
        if not obj_pts:
            return None
        object_points = np.vstack(obj_pts)
        image_points = np.vstack(img_pts)
        if object_points.shape[0] < 4:
            return None
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self._K,
            self._dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        return tvec.reshape(3), R

    def _fuse_from_markers(self, poses: list[_MarkerPose]) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if not poses:
            return None
        origins: list[np.ndarray] = []
        rotations: list[np.ndarray] = []
        for pose in poses:
            t_o, R = self._origin_from_marker(pose)
            origins.append(t_o)
            rotations.append(R)
        t_mean = np.mean(origins, axis=0)
        R_mean = np.mean(rotations, axis=0)
        u, _, vt = np.linalg.svd(R_mean)
        R_ortho = u @ vt
        if np.linalg.det(R_ortho) < 0:
            R_ortho[:, 2] *= -1
        return t_mean, R_ortho

    @staticmethod
    def _yaw_from_marker_image(pose: _MarkerPose) -> float:
        """Heading from SQUARE SHAPE in image — angle of marker local +X axis.

        Detector corners have fixed order tied to tag orientation
        (obj pts: 0=(-h,+h), 1=(+h,+h), 2=(+h,-h), 3=(-h,-h)), so edges
        0->1 and 3->2 point to local +X. For top-down view this is rotation in
        the image plane: unambiguous and robust to PnP flip ambiguity that
        could jump yaw with a single tag.
        """
        c = pose.corners.reshape(4, 2).astype(np.float64)
        ex = ((c[1] - c[0]) + (c[2] - c[3])) * 0.5
        return float(np.arctan2(ex[1], ex[0]))

    @staticmethod
    def _yaw_from_image_layout(poses: list[_MarkerPose]) -> Optional[float]:
        by_id = {p.id: p for p in poses}

        def _center_img(mid: int) -> Optional[np.ndarray]:
            p = by_id.get(mid)
            if p is None:
                return None
            return p.corners.reshape(-1, 2).mean(axis=0)

        front_pts = [_center_img(i) for i in (1, 2)]
        back_pts = [_center_img(i) for i in (3, 4)]
        front_pts = [x for x in front_pts if x is not None]
        back_pts = [x for x in back_pts if x is not None]
        if front_pts and back_pts:
            d = np.mean(front_pts, axis=0) - np.mean(back_pts, axis=0)
            if float(np.hypot(d[0], d[1])) > 2.0:
                return float(np.arctan2(d[1], d[0]))
        return None

    def _tvec_to_lab(self, tvec: np.ndarray) -> tuple[float, float, float]:
        """Camera tvec -> lab x=left, y=forward, z=up (height)."""
        tv = np.asarray(tvec, dtype=np.float64).reshape(3)
        ix = int(np.clip(self._tag.lab_x_tvec_index, 0, 2))
        iy = int(np.clip(self._tag.lab_y_tvec_index, 0, 2))
        x_m = float(self._tag.lab_x_sign) * float(tv[ix])
        y_m = float(self._tag.lab_y_sign) * float(tv[iy])
        z_m = float(abs(tv[2]))
        return x_m, y_m, z_m

    def _hint_no_detection(self, all_seen: list[int]) -> str:
        base = self._dict_hint()
        if all_seen:
            return (
                f"Detected other IDs: {all_seen} (not 0–4) — likely wrong family or false positives. {base}"
            )
        mm = int(round(self._tag.marker_length_m * 1000))
        return (
            f"No tags 0–4. Check focus, lighting, and black square size ({mm} mm, no border). {base}"
        )

    def detect_top_pose(self, frame_bgr: Optional[np.ndarray]) -> TopPoseObservation:
        if frame_bgr is None or frame_bgr.size == 0:
            return TopPoseObservation(ok=False, hint="No TOP frame.")

        poses, all_seen, overlays = self._detect_markers(frame_bgr)
        seen = tuple(sorted({p.id for p in poses}))
        reference_marker_id = 0 if 0 in seen else (seen[0] if seen else None)

        if not poses:
            return TopPoseObservation(
                ok=False,
                reference_marker_id=reference_marker_id,
                marker_overlays=overlays,
                hint=self._hint_no_detection(all_seen),
            )

        pnp = self._solve_multi_pnp(poses)
        fused = self._fuse_from_markers(poses)
        if pnp is not None:
            tvec, R = pnp
        elif fused is not None:
            tvec, R = fused
        else:
            return TopPoseObservation(
                ok=False,
                markers_seen=seen,
                marker_overlays=overlays,
                hint=f"TOP: detected {seen}, but pose estimation failed.",
            )

        _, pitch, roll = _rotation_to_euler_zyx(R)
        # Heading from reference marker shape in image (stable for top-down view),
        # NOT from PnP yaw (flip ambiguity with one tag).
        ref_pose = next((p for p in poses if p.id == reference_marker_id), poses[0])
        yaw = self._yaw_from_marker_image(ref_pose)

        x_m, y_m, z_m = self._tvec_to_lab(tvec)
        cen = ref_pose.corners.reshape(-1, 2).mean(axis=0)

        ok_pose = len(seen) >= 1
        if not ok_pose:
            return TopPoseObservation(
                ok=False,
                x_m=x_m,
                y_m=y_m,
                z_m=z_m,
                yaw_rad=_unwrap_yaw(yaw),
                pitch_rad=pitch,
                roll_rad=roll,
                reference_marker_id=reference_marker_id,
                pose_quality=min(1.0, len(seen) / 2.0),
                markers_seen=seen,
                marker_overlays=overlays,
                centroid_u_px=float(cen[0]),
                centroid_v_px=float(cen[1]),
                hint=f"TOP: too few tags ({seen}). Need ID 0 or ≥2 IDs.",
            )

        return TopPoseObservation(
            ok=True,
            x_m=x_m,
            y_m=y_m,
            z_m=z_m,
            yaw_rad=_unwrap_yaw(yaw),
            pitch_rad=pitch,
            roll_rad=roll,
            reference_marker_id=reference_marker_id,
            pose_quality=min(1.0, len(seen) / 3.0 + (0.25 if 0 in seen else 0.0)),
            markers_seen=seen,
            marker_overlays=overlays,
            centroid_u_px=float(cen[0]),
            centroid_v_px=float(cen[1]),
            hint="TOP: tracking relative to TAG 0" if 0 in seen else f"TOP: layout tracking without TAG 0 ({seen})",
        )

    def detect_side_correction(self, frame_bgr: Optional[np.ndarray]) -> SideCorrectionObservation:
        if frame_bgr is None or frame_bgr.size == 0:
            return SideCorrectionObservation(ok=False, hint="No SIDE frame.")

        poses, all_seen, overlays = self._detect_markers(frame_bgr)
        seen = tuple(sorted({p.id for p in poses}))

        if not poses:
            return SideCorrectionObservation(
                ok=False,
                marker_overlays=overlays,
                hint=self._hint_no_detection(all_seen),
            )

        def _area(p: _MarkerPose) -> float:
            c = p.corners.reshape(-1, 2)
            return float(cv2.contourArea(c.astype(np.float32)))

        best = max(poses, key=_area)
        image_area_px = _area(best)
        centroid = best.corners.reshape(-1, 2).mean(axis=0)
        frame_area = max(1.0, float(frame_bgr.shape[0] * frame_bgr.shape[1]))
        pose_quality = float(np.clip(image_area_px / (0.2 * frame_area), 0.0, 1.0))
        tvec = best.tvec.reshape(3)
        R, _ = cv2.Rodrigues(best.rvec.reshape(3, 1))
        yaw_corr = float(np.arctan2(R[1, 0], R[0, 0]))
        # Marker normal (+Z of object) in camera frame; when marker faces the
        # camera directly, normal is ~(0,0,-1). Horizontal angle of normal = drone
        # yaw relative to side camera (0 = tag facing camera).
        normal = R[:, 2]
        yaw_facing = float(np.arctan2(float(normal[0]), -float(normal[2])))

        return SideCorrectionObservation(
            ok=True,
            x_m=float(tvec[0]),
            y_m=float(-tvec[1]),
            z_m=float(abs(tvec[2])),
            yaw_rad=_unwrap_yaw(yaw_corr),
            yaw_facing_rad=_unwrap_yaw(yaw_facing),
            marker_id=best.id,
            image_area_px=image_area_px,
            centroid_u_px=float(centroid[0]),
            centroid_v_px=float(centroid[1]),
            pose_quality=pose_quality,
            markers_seen=seen,
            marker_overlays=overlays,
            hint=f"SIDE tag={best.id} area={image_area_px:.0f}px facing={np.degrees(yaw_facing):.0f}deg",
        )
