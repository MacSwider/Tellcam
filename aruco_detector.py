"""Detekcja wielu markerów ArUco (ID 0–4) — pozycja i orientacja drona (TOP + SIDE)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

from config import ArucoConfig, CameraConfig
from fiducial_detector import FiducialDetection, FiducialDetector

# Układ od góry (oś X = przód, Y = lewo):
#   [1]     [2]
#        [0]
#   [4]      [3]
TOP_MARKER_IDS = frozenset({0, 1, 2, 3, 4})
SIDE_MARKER_IDS = frozenset({1, 2, 3, 4})


@dataclass(frozen=True)
class MarkerOverlay:
    """Obrys markera w pikselach klatki (do podglądu)."""

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
    markers_seen: tuple[int, ...] = ()
    marker_overlays: Tuple[MarkerOverlay, ...] = ()
    hint: Optional[str] = None


@dataclass
class SideCorrectionObservation:
    ok: bool
    yaw_correction_rad: float = 0.0
    z_from_marker_m: float = 0.0
    markers_seen: tuple[int, ...] = ()
    marker_overlays: Tuple[MarkerOverlay, ...] = ()
    hint: Optional[str] = None


@dataclass
class _MarkerPose:
    id: int
    rvec: np.ndarray
    tvec: np.ndarray
    corners: np.ndarray
    dictionary_name: str


def _get_dictionary(name: str):
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Nieznany słownik ArUco: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


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


class ArucoDetector:
    def __init__(self, ar_cfg: ArucoConfig, cam_cfg: CameraConfig) -> None:
        self._ar = ar_cfg
        self._cam = cam_cfg
        self._fiducial = FiducialDetector(ar_cfg)
        self._half_marker = float(ar_cfg.marker_length_m) * 0.5
        self._body_centers = self._build_body_centers()
        self._allowed_ids = set(int(x) for x in ar_cfg.marker_ids)
        if ar_cfg.calibration_path:
            self._K, self._dist = _load_intrinsics(ar_cfg.calibration_path, cam_cfg.width, cam_cfg.height)
        else:
            self._K, self._dist = _default_intrinsics(cam_cfg.width, cam_cfg.height)
        self._frame_K_wh: Tuple[int, int] | None = None
        self._active_dictionary_name = ar_cfg.dictionary_name
        self._detection_mode = "none"

    def _build_body_centers(self) -> Dict[int, np.ndarray]:
        hf = float(self._ar.layout_half_forward_m)
        hl = float(self._ar.layout_half_lateral_m)
        sz = float(self._ar.marker_side_height_m)
        return {
            0: np.array([0.0, 0.0, 0.0]),
            1: np.array([hf, hl, -sz]),
            2: np.array([hf, -hl, -sz]),
            3: np.array([-hf, -hl, -sz]),
            4: np.array([-hf, hl, -sz]),
        }

    def _dict_hint(self) -> str:
        return (
            f"Słownik: {self._active_dictionary_name}. "
            "Wydruk: python generate_aruco_markers.py (ten sam słownik co w config). "
            f"Oczekiwane ID: {sorted(self._allowed_ids)}."
        )

    def _ensure_intrinsics(self, w: int, h: int) -> None:
        if self._ar.calibration_path:
            return
        if self._ar.intrinsics_from_frame_size and self._frame_K_wh != (w, h):
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
            return poses, all_seen, ()

        corners, ids = det.corners, det.ids
        dict_name = self._active_dictionary_name

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
            c = [corners[i]]
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                c, self._ar.marker_length_m, self._K, self._dist
            )
            rvec = rvecs[0].reshape(3)
            tvec = tvecs[0].reshape(3)
            poses.append(
                _MarkerPose(
                    id=marker_id,
                    rvec=rvec,
                    tvec=tvec,
                    corners=corners[i],
                    dictionary_name=dict_name,
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

    def _hint_no_detection(self, all_seen: list[int]) -> str:
        base = self._dict_hint()
        if all_seen:
            return (
                f"Wykryto inne ID: {all_seen} (nie 0–4) — prawdopodobnie zły słownik lub fałszywe trafienia. {base}"
            )
        return (
            f"Brak markerów 0–4. Sprawdź ostrość, światło, rozmiar (20 mm) i słownik. {base}"
        )

    def detect_top_pose(self, frame_bgr: Optional[np.ndarray]) -> TopPoseObservation:
        if frame_bgr is None or frame_bgr.size == 0:
            return TopPoseObservation(ok=False, hint="Brak klatki TOP.")

        poses, all_seen, overlays = self._detect_markers(frame_bgr)
        seen = tuple(sorted({p.id for p in poses}))

        if not poses:
            return TopPoseObservation(
                ok=False,
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
                hint=f"TOP: wykryto {seen}, ale estymacja pozy nieudana.",
            )

        yaw, pitch, roll = _rotation_to_euler_zyx(R)
        yaw_img = self._yaw_from_image_layout(poses)
        if yaw_img is not None and len(poses) >= 2:
            dy = np.arctan2(np.sin(yaw_img - yaw), np.cos(yaw_img - yaw))
            yaw = _unwrap_yaw(yaw + 0.35 * dy)

        x_m, y_m, z_m = float(tvec[0]), float(tvec[1]), float(abs(tvec[2]))

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
                markers_seen=seen,
                marker_overlays=overlays,
                hint=f"TOP: za mało markerów ({seen}). Potrzebny ID 0 lub ≥2 ID.",
            )

        return TopPoseObservation(
            ok=True,
            x_m=x_m,
            y_m=y_m,
            z_m=z_m,
            yaw_rad=_unwrap_yaw(yaw),
            pitch_rad=pitch,
            roll_rad=roll,
            markers_seen=seen,
            marker_overlays=overlays,
        )

    def detect_side_correction(self, frame_bgr: Optional[np.ndarray]) -> SideCorrectionObservation:
        if frame_bgr is None or frame_bgr.size == 0:
            return SideCorrectionObservation(ok=False, hint="Brak klatki SIDE.")

        poses, all_seen, overlays = self._detect_markers(frame_bgr)
        side_poses = [p for p in poses if p.id in SIDE_MARKER_IDS]
        seen = tuple(sorted({p.id for p in side_poses}))

        if not side_poses:
            return SideCorrectionObservation(
                ok=False,
                marker_overlays=overlays,
                hint=self._hint_no_detection(all_seen),
            )

        def _area(p: _MarkerPose) -> float:
            c = p.corners.reshape(-1, 2)
            return float(cv2.contourArea(c.astype(np.float32)))

        best = max(side_poses, key=_area)
        t_origin, R = self._origin_from_marker(best)
        z_m = float(abs(t_origin[2]))
        if z_m < 0.05:
            z_m = float(abs(best.tvec[2]))

        yaw_corr = float(np.arctan2(R[1, 0], R[0, 0]))
        if len(side_poses) >= 2:
            by_id = {p.id: p for p in side_poses}
            if 1 in by_id and 2 in by_id:
                c1 = by_id[1].corners.reshape(-1, 2).mean(axis=0)
                c2 = by_id[2].corners.reshape(-1, 2).mean(axis=0)
                yaw_corr = float(np.arctan2(c2[1] - c1[1], c2[0] - c1[0]))
            elif 4 in by_id and 3 in by_id:
                c4 = by_id[4].corners.reshape(-1, 2).mean(axis=0)
                c3 = by_id[3].corners.reshape(-1, 2).mean(axis=0)
                yaw_corr = float(np.arctan2(c3[1] - c4[1], c3[0] - c4[0]))

        return SideCorrectionObservation(
            ok=True,
            yaw_correction_rad=_unwrap_yaw(yaw_corr),
            z_from_marker_m=z_m,
            markers_seen=seen,
            marker_overlays=overlays,
        )
