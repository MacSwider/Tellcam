"""
Multi-threaded Tellcam vision pipeline: cameras, AprilTag detection, control, preview.

- CameraReader: read() only @ max FPS, overwrites latest frame
- AprilTagDetectionWorker: detection @ fixed rate on freshest frame
- ControlLoop (main thread): PID / fusion @ control_hz, no tag detection
- PreviewGUI: preview @ preview_hz, no detection
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np

from apriltag_detector import (
    AprilTagDetector,
    MarkerOverlay,
    SideCorrectionObservation,
    TopPoseObservation,
)
from config import AppConfig, CameraConfig, PipelineConfig as ConfigPipelineConfig, apriltag_for_role
from flight_hud import FlightHudSnapshot, draw_flight_hud
from state_estimator import DroneState, OdometrySample, StateEstimator
from video_source import VideoSource

log = logging.getLogger(__name__)

TOP = "top"
SIDE = "side"


PipelineConfig = ConfigPipelineConfig


@dataclass
class FrameSlot:
    frame: Optional[np.ndarray] = None
    timestamp: float = 0.0
    seq: int = 0
    width: int = 0
    height: int = 0


@dataclass
class MarkerOverlaySlot:
    overlays: Tuple[MarkerOverlay, ...] = ()
    timestamp: float = 0.0


@dataclass
class PoseSlot:
    """Latest pose observation (data for control)."""
    top: TopPoseObservation = field(default_factory=lambda: TopPoseObservation(ok=False))
    side: SideCorrectionObservation = field(default_factory=lambda: SideCorrectionObservation(ok=False))
    top_timestamp: float = 0.0
    side_timestamp: float = 0.0
    top_seq: int = 0
    side_seq: int = 0


@dataclass
class TrajectoryOverlay:
    """Route elements projected to pixels (TOP camera) for preview drawing."""
    waypoints_px: List[Tuple[int, int]] = field(default_factory=list)
    start_px: Optional[Tuple[int, int]] = None
    drone_px: Optional[Tuple[int, int]] = None
    current_index: int = 0
    total: int = 0


@dataclass
class StatusSnapshot:
    hud: FlightHudSnapshot = field(default_factory=FlightHudSnapshot)
    lines: List[str] = field(default_factory=list)
    control_hz: float = 0.0
    top_detection_hz: float = 0.0
    side_detection_hz: float = 0.0
    top_frame_age_s: float = 0.0
    side_frame_age_s: float = 0.0
    top_pose_age_s: float = 0.0
    side_pose_age_s: float = 0.0
    state: Optional[DroneState] = None


class VisionSharedState:
    """Shared state between threads (lock per slot)."""

    def __init__(self) -> None:
        self._frame_locks = {TOP: threading.Lock(), SIDE: threading.Lock()}
        self._pose_lock = threading.Lock()
        self._overlay_locks = {TOP: threading.Lock(), SIDE: threading.Lock()}
        self._status_lock = threading.Lock()
        self.frames: dict[str, FrameSlot] = {TOP: FrameSlot(), SIDE: FrameSlot()}
        self.marker_overlays: dict[str, MarkerOverlaySlot] = {
            TOP: MarkerOverlaySlot(),
            SIDE: MarkerOverlaySlot(),
        }
        self.poses = PoseSlot()
        self.status = StatusSnapshot()
        self.stop = threading.Event()
        self.mouse_clicks: queue.Queue[Tuple[int, int]] = queue.Queue(maxsize=8)
        self.ui_events: queue.Queue[str] = queue.Queue(maxsize=16)
        self.button_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._button_lock = threading.Lock()
        self._traj_lock = threading.Lock()
        self.trajectory: Optional[TrajectoryOverlay] = None

    def publish_trajectory(self, overlay: Optional[TrajectoryOverlay]) -> None:
        with self._traj_lock:
            self.trajectory = overlay

    def peek_trajectory(self) -> Optional[TrajectoryOverlay]:
        with self._traj_lock:
            return self.trajectory

    def set_button_rect(self, rect: Tuple[int, int, int, int]) -> None:
        with self._button_lock:
            self.button_rect = rect

    def get_button_rect(self) -> Tuple[int, int, int, int]:
        with self._button_lock:
            return self.button_rect

    def publish_frame(self, camera_id: str, frame: np.ndarray) -> None:
        with self._frame_locks[camera_id]:
            slot = self.frames[camera_id]
            slot.frame = frame
            slot.timestamp = time.perf_counter()
            slot.seq += 1
            slot.height, slot.width = int(frame.shape[0]), int(frame.shape[1])

    def snapshot_frame(self, camera_id: str) -> Tuple[Optional[np.ndarray], float, int]:
        with self._frame_locks[camera_id]:
            slot = self.frames[camera_id]
            if slot.frame is None:
                return None, 0.0, 0
            return slot.frame.copy(), slot.timestamp, slot.seq

    def peek_frame(self, camera_id: str) -> Tuple[Optional[np.ndarray], float]:
        """Preview: frame copy without blocking detection for long."""
        with self._frame_locks[camera_id]:
            slot = self.frames[camera_id]
            if slot.frame is None:
                return None, 0.0
            return slot.frame.copy(), slot.timestamp

    def publish_marker_overlay(self, camera_id: str, overlays: Tuple[MarkerOverlay, ...]) -> None:
        with self._overlay_locks[camera_id]:
            slot = self.marker_overlays[camera_id]
            slot.overlays = overlays
            slot.timestamp = time.perf_counter()

    def peek_marker_overlay(self, camera_id: str) -> Tuple[Tuple[MarkerOverlay, ...], float]:
        with self._overlay_locks[camera_id]:
            slot = self.marker_overlays[camera_id]
            return slot.overlays, slot.timestamp

    def publish_top_pose(self, obs: TopPoseObservation) -> None:
        with self._pose_lock:
            self.poses.top = obs
            self.poses.top_timestamp = time.perf_counter()
            self.poses.top_seq += 1

    def publish_side_pose(self, obs: SideCorrectionObservation) -> None:
        with self._pose_lock:
            self.poses.side = obs
            self.poses.side_timestamp = time.perf_counter()
            self.poses.side_seq += 1

    def read_poses(self) -> Tuple[TopPoseObservation, SideCorrectionObservation, float, float]:
        with self._pose_lock:
            return self.poses.top, self.poses.side, self.poses.top_timestamp, self.poses.side_timestamp

    def publish_status(self, snap: StatusSnapshot) -> None:
        with self._status_lock:
            self.status = snap

    def read_status(self) -> StatusSnapshot:
        with self._status_lock:
            return StatusSnapshot(
                hud=self.status.hud,
                lines=list(self.status.lines),
                control_hz=self.status.control_hz,
                top_detection_hz=self.status.top_detection_hz,
                side_detection_hz=self.status.side_detection_hz,
                top_frame_age_s=self.status.top_frame_age_s,
                side_frame_age_s=self.status.side_frame_age_s,
                top_pose_age_s=self.status.top_pose_age_s,
                side_pose_age_s=self.status.side_pose_age_s,
                state=self.status.state,
            )


class CameraReader(threading.Thread):
    """Camera thread: read() only @ max FPS, overwrite latest frame."""

    def __init__(
        self,
        name: str,
        camera_id: str,
        source_factory: Callable[[], VideoSource],
        shared: VisionSharedState,
        daemon: bool = True,
    ) -> None:
        super().__init__(name=name, daemon=daemon)
        self._camera_id = camera_id
        self._source_factory = source_factory
        self._shared = shared
        self._source: Optional[VideoSource] = None
        self.read_count = 0
        self.read_hz = 0.0

    def run(self) -> None:
        self._source = self._source_factory()
        log.info("[%s] CameraReader start", self.name)
        fps_n = 0
        fps_t0 = time.perf_counter()
        try:
            while not self._shared.stop.is_set():
                ok, frame = self._source.read_bgr()
                if ok and frame is not None:
                    self._shared.publish_frame(self._camera_id, frame.copy())
                    self.read_count += 1
                    fps_n += 1
                now = time.perf_counter()
                if now - fps_t0 >= 1.0:
                    self.read_hz = fps_n / (now - fps_t0)
                    fps_n = 0
                    fps_t0 = now
        except Exception:
            log.exception("[%s] CameraReader error", self.name)
        finally:
            if self._source:
                self._source.close()
            log.info("[%s] CameraReader stop (reads=%s)", self.name, self.read_count)

    def close(self) -> None:
        if self._source:
            self._source.close()


class AprilTagDetectionWorker(threading.Thread):
    """Detection thread: fixed rate, always freshest frame from CameraReader."""

    def __init__(
        self,
        name: str,
        camera_id: str,
        detection_hz: float,
        shared: VisionSharedState,
        apriltag_cfg,
        camera_cfg: CameraConfig,
        *,
        mode: str,
        daemon: bool = True,
    ) -> None:
        super().__init__(name=name, daemon=daemon)
        if mode not in (TOP, SIDE):
            raise ValueError(f"mode must be '{TOP}' or '{SIDE}', got {mode!r}")
        self._camera_id = camera_id
        self._mode = mode
        self._period = 1.0 / max(float(detection_hz), 0.1)
        self._shared = shared
        self._detector = AprilTagDetector(apriltag_cfg, camera_cfg)
        self.detection_count = 0
        self.detection_hz = 0.0
        self.last_detect_ms = 0.0
        self._slow_warn_t = 0.0
        self.last_input_width = 0
        self.last_input_height = 0

    @property
    def camera_matrix(self) -> np.ndarray:
        return self._detector.camera_matrix

    @property
    def dist_coeffs(self) -> np.ndarray:
        return self._detector.dist_coeffs

    def _detect(self, frame: np.ndarray) -> Union[TopPoseObservation, SideCorrectionObservation]:
        if self._mode == TOP:
            return self._detector.detect_top_pose(frame)
        return self._detector.detect_side_correction(frame)

    def _publish(self, obs: Union[TopPoseObservation, SideCorrectionObservation]) -> None:
        self._shared.publish_marker_overlay(self._camera_id, obs.marker_overlays)
        if self._mode == TOP:
            self._shared.publish_top_pose(obs)  # type: ignore[arg-type]
        else:
            self._shared.publish_side_pose(obs)  # type: ignore[arg-type]

    def run(self) -> None:
        log.info("[%s] AprilTagDetectionWorker start @ %.1f Hz", self.name, 1.0 / self._period)
        fps_n = 0
        fps_t0 = time.perf_counter()
        try:
            while not self._shared.stop.is_set():
                t0 = time.perf_counter()
                frame, _, _ = self._shared.snapshot_frame(self._camera_id)
                if frame is not None:
                    self.last_input_height, self.last_input_width = int(frame.shape[0]), int(frame.shape[1])
                    obs = self._detect(frame)
                    self._publish(obs)
                    self.detection_count += 1
                    fps_n += 1
                elapsed = time.perf_counter() - t0
                self.last_detect_ms = elapsed * 1000.0
                if elapsed > self._period * 1.5 and (time.perf_counter() - self._slow_warn_t) > 5.0:
                    self._slow_warn_t = time.perf_counter()
                    log.warning(
                        "[%s] slow detection: %.0f ms (limit ~%.0f ms @ %.1f Hz, frame=%sx%s)",
                        self.name,
                        self.last_detect_ms,
                        self._period * 1000.0,
                        1.0 / self._period,
                        frame.shape[1] if frame is not None else 0,
                        frame.shape[0] if frame is not None else 0,
                    )
                sleep_s = self._period - elapsed
                if sleep_s > 0:
                    if self._shared.stop.wait(timeout=sleep_s):
                        break
                now = time.perf_counter()
                if now - fps_t0 >= 1.0:
                    self.detection_hz = fps_n / (now - fps_t0)
                    fps_n = 0
                    fps_t0 = now
        except Exception:
            log.exception("[%s] AprilTagDetectionWorker error", self.name)
        finally:
            log.info("[%s] AprilTagDetectionWorker stop (detections=%s)", self.name, self.detection_count)


class ControlLoop:
    """Control loop @ control_hz — reads pose from shared state only."""

    def __init__(
        self,
        shared: VisionSharedState,
        pipeline_cfg: PipelineConfig,
        estimator: StateEstimator,
        *,
        on_tick: Optional[Callable[[DroneState, TopPoseObservation, SideCorrectionObservation, float], FlightHudSnapshot]] = None,
        odometry_provider: Optional[Callable[[], OdometrySample | None]] = None,
        top_detection: Optional[AprilTagDetectionWorker] = None,
        side_detection: Optional[AprilTagDetectionWorker] = None,
        top_camera: Optional[CameraReader] = None,
        side_camera: Optional[CameraReader] = None,
    ) -> None:
        self._shared = shared
        self._period = 1.0 / max(pipeline_cfg.control_hz, 1.0)
        self._estimator = estimator
        self._on_tick = on_tick
        self._odometry_provider = odometry_provider
        self._top_detection = top_detection
        self._side_detection = side_detection
        self._top_camera = top_camera
        self._side_camera = side_camera
        self.tick_count = 0
        self.control_hz = 0.0

    def spin(self) -> None:
        log.info("ControlLoop start @ %.1f Hz", 1.0 / self._period)
        fps_n = 0
        fps_t0 = time.perf_counter()
        last_t = time.perf_counter()
        while not self._shared.stop.is_set():
            tick_start = time.perf_counter()
            dt = tick_start - last_t
            last_t = tick_start

            now = time.perf_counter()
            top_obs, side_obs, top_ts, side_ts = self._shared.read_poses()
            odom: OdometrySample | None = None
            if self._odometry_provider is not None:
                try:
                    odom = self._odometry_provider()
                except Exception:
                    log.debug("odometry_provider failed", exc_info=True)
            state = self._estimator.update(
                top_obs, side_obs, top_ts=top_ts, side_ts=side_ts, now=now, odometry=odom
            )
            hud = FlightHudSnapshot()
            if self._on_tick:
                hud = self._on_tick(state, top_obs, side_obs, dt)

            with self._shared._frame_locks[TOP]:
                top_frame_ts = self._shared.frames[TOP].timestamp
            with self._shared._frame_locks[SIDE]:
                side_frame_ts = self._shared.frames[SIDE].timestamp

            snap = StatusSnapshot(
                hud=hud,
                control_hz=self.control_hz,
                top_detection_hz=self._top_detection.detection_hz if self._top_detection else 0.0,
                side_detection_hz=self._side_detection.detection_hz if self._side_detection else 0.0,
                top_frame_age_s=now - top_frame_ts if top_frame_ts else 0.0,
                side_frame_age_s=now - side_frame_ts if side_frame_ts else 0.0,
                top_pose_age_s=now - top_ts if top_ts else 0.0,
                side_pose_age_s=now - side_ts if side_ts else 0.0,
                state=state,
            )
            self._shared.publish_status(snap)

            self.tick_count += 1
            fps_n += 1
            if now - fps_t0 >= 1.0:
                self.control_hz = fps_n / (now - fps_t0)
                fps_n = 0
                fps_t0 = now

            elapsed = time.perf_counter() - tick_start
            sleep_s = self._period - elapsed
            if sleep_s > 0:
                if self._shared.stop.wait(timeout=sleep_s):
                    break

        log.info("ControlLoop stop (ticks=%s)", self.tick_count)


def draw_marker_overlays(
    frame_bgr: np.ndarray,
    overlays: Tuple[MarkerOverlay, ...],
    *,
    max_age_s: float = 0.5,
    overlay_age_s: float = 0.0,
) -> np.ndarray:
    """Draw marker outlines on frame copy (preview thread, no detection)."""
    if not overlays or overlay_age_s > max_age_s:
        return frame_bgr
    out = frame_bgr.copy()
    for m in overlays:
        color = (0, 255, 0) if m.in_layout else (0, 80, 255)
        pts = m.corners.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(out, [pts], True, color, 2, cv2.LINE_AA)
        center = m.corners.mean(axis=0).astype(int)
        cv2.putText(
            out,
            str(m.marker_id),
            (int(center[0]) + 4, int(center[1]) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def draw_trajectory_overlay(frame_bgr: np.ndarray, ov: Optional[TrajectoryOverlay]) -> np.ndarray:
    """Draw planned route on TOP frame: line, points, numbers, current target."""
    if ov is None or not ov.waypoints_px:
        return frame_bgr
    out = frame_bgr
    # Polyline: start -> successive waypoints.
    chain: List[Tuple[int, int]] = []
    if ov.start_px is not None:
        chain.append(ov.start_px)
    chain.extend(ov.waypoints_px)
    for i in range(len(chain) - 1):
        cv2.line(out, chain[i], chain[i + 1], (210, 200, 60), 2, cv2.LINE_AA)
    # Start point.
    if ov.start_px is not None:
        cv2.drawMarker(out, ov.start_px, (255, 170, 40), cv2.MARKER_TRIANGLE_UP, 16, 2)
        cv2.putText(out, "S", (ov.start_px[0] + 8, ov.start_px[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 170, 40), 2, cv2.LINE_AA)
    # Waypoints: reached (green), current (yellow, larger), future (gray).
    for idx, p in enumerate(ov.waypoints_px):
        if idx < ov.current_index:
            color = (90, 200, 90)
        elif idx == ov.current_index:
            color = (0, 215, 255)
        else:
            color = (170, 170, 170)
        radius = 10 if idx == ov.current_index else 6
        cv2.circle(out, p, radius, color, -1, cv2.LINE_AA)
        cv2.circle(out, p, radius, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(out, str(idx + 1), (p[0] + 9, p[1] - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    # Drone + line to current target.
    if ov.drone_px is not None:
        cv2.drawMarker(out, ov.drone_px, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        if 0 <= ov.current_index < len(ov.waypoints_px):
            cv2.line(out, ov.drone_px, ov.waypoints_px[ov.current_index],
                     (0, 140, 255), 1, cv2.LINE_AA)
    return out


class PreviewGUI(threading.Thread):
    """Preview @ preview_hz — resize/imshow + outlines from last detection."""

    def __init__(
        self,
        shared: VisionSharedState,
        pipeline_cfg: PipelineConfig,
        *,
        window_name: str = "Tellcam — TOP | SIDE",
        top_label: str = "TOP",
        side_label: str = "SIDE",
        preview_max_width: int = 1400,
        control_enabled: bool = False,
        top_enabled: bool = True,
        side_enabled: bool = True,
        on_mouse_setup: bool = True,
        daemon: bool = True,
    ) -> None:
        super().__init__(name="PreviewGUI", daemon=daemon)
        self._shared = shared
        self._period = 1.0 / max(pipeline_cfg.preview_hz, 1.0)
        self._show_overlay = pipeline_cfg.show_detection_overlay
        self._window_name = window_name
        self._top_label = top_label
        self._side_label = side_label
        self._preview_max_width = preview_max_width
        self._control_enabled = control_enabled
        self._top_enabled = top_enabled
        self._side_enabled = side_enabled
        self.button_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.button_rects: dict[str, Tuple[int, int, int, int]] = {}
        self._disp_scale = 1.0
        self.preview_hz = 0.0

    def _on_mouse(self, event, x, y, flags, param) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            # Click coordinates are in displayed (scaled) image space —
            # convert to panel coordinates where buttons were stored.
            scale = self._disp_scale if self._disp_scale > 0 else 1.0
            x = int(x / scale)
            y = int(y / scale)
            for action, (x0, y0, x1, y1) in self.button_rects.items():
                if x0 <= x <= x1 and y0 <= y <= y1:
                    try:
                        self._shared.ui_events.put_nowait(action)
                    except queue.Full:
                        pass
                    break

    @staticmethod
    def _placeholder_frame(label: str, width: int = 640, height: int = 480) -> np.ndarray:
        frame = np.full((height, width, 3), (24, 24, 24), dtype=np.uint8)
        cv2.putText(frame, label, (30, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2, cv2.LINE_AA)
        return frame

    @staticmethod
    def _pad_to_height(frame: np.ndarray, target_h: int) -> np.ndarray:
        h, w = frame.shape[:2]
        if h == target_h:
            return frame
        if h < target_h:
            pad = target_h - h
            top = pad // 2
            bottom = pad - top
            return cv2.copyMakeBorder(
                frame, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24)
            )
        scale = target_h / float(h)
        return cv2.resize(
            frame,
            (max(1, int(w * scale)), target_h),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _add_source_bar(frame: np.ndarray, label: str, *, size_tag: str = "") -> np.ndarray:
        bar_h = 30
        bar = np.full((bar_h, frame.shape[1], 3), (45, 45, 45), dtype=np.uint8)
        cv2.putText(bar, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 2, cv2.LINE_AA)
        if size_tag:
            tx = int(max(10, int(frame.shape[1]) - 220))
            cv2.putText(
                bar,
                size_tag,
                (tx, 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (160, 220, 160),
                1,
                cv2.LINE_AA,
            )
        return cv2.vconcat([bar, frame])

    def _frame_for_preview(self, camera_id: str, live: np.ndarray) -> np.ndarray:
        if not self._show_overlay:
            return live
        overlays, overlay_ts = self._shared.peek_marker_overlay(camera_id)
        age = time.perf_counter() - overlay_ts if overlay_ts else 999.0
        return draw_marker_overlays(live, overlays, max_age_s=1.5, overlay_age_s=age)

    @staticmethod
    def _draw_overlay(img, text_lines: List[str], color: Tuple[int, int, int] = (0, 220, 220)) -> None:
        y = 20
        for line in text_lines:
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            y += 22

    def run(self) -> None:
        log.info("PreviewGUI start @ %.1f Hz", 1.0 / self._period)
        fps_n = 0
        fps_t0 = time.perf_counter()
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self._window_name, self._on_mouse)
        try:
            while not self._shared.stop.is_set():
                t0 = time.perf_counter()
                top_frame, _ = self._shared.peek_frame(TOP)
                side_frame, _ = self._shared.peek_frame(SIDE)
                status = self._shared.read_status()

                if top_frame is None:
                    top_frame = self._placeholder_frame("TOP disabled" if not self._top_enabled else "TOP waiting...")
                if side_frame is None:
                    side_frame = self._placeholder_frame("SIDE disabled" if not self._side_enabled else "SIDE waiting...")

                top_disp = self._frame_for_preview(TOP, top_frame)
                if self._show_overlay and self._top_enabled:
                    top_disp = draw_trajectory_overlay(top_disp, self._shared.peek_trajectory())
                side_disp = self._frame_for_preview(SIDE, side_frame)
                top_size = f"{top_frame.shape[1]}×{top_frame.shape[0]}"
                side_size = f"{side_frame.shape[1]}×{side_frame.shape[0]}"
                left = self._add_source_bar(top_disp, self._top_label, size_tag=top_size)
                right = self._add_source_bar(side_disp, self._side_label, size_tag=side_size)
                panel_h = max(left.shape[0], right.shape[0])
                cameras_panel = cv2.hconcat(
                    [self._pad_to_height(left, panel_h), self._pad_to_height(right, panel_h)]
                )

                hud = status.hud
                hud.preview_hz = self.preview_hz
                hud.control_hz = status.control_hz
                terminal = draw_flight_hud(cameras_panel.shape[1], hud)
                term_h = terminal.shape[0]

                if self._control_enabled:
                    button_specs = [
                        ("takeoff", "Start", (0, 140, 0)),
                        ("travel", "Travel", (150, 90, 0)),
                        ("land", "Land", (0, 120, 200)),
                        ("emergency", "STOP", (0, 0, 220)),
                    ]
                    btn_w, btn_h, gap = 130, 36, 10
                    by0 = max(8, term_h - btn_h - 8)
                    total_w = len(button_specs) * btn_w + (len(button_specs) - 1) * gap
                    bx = max(10, terminal.shape[1] - total_w - 10)
                    self.button_rects = {}
                    for action, label, color in button_specs:
                        x0, y0 = bx, by0
                        x1, y1 = x0 + btn_w, y0 + btn_h
                        cv2.rectangle(terminal, (x0, y0), (x1, y1), color, -1)
                        cv2.rectangle(terminal, (x0, y0), (x1, y1), (255, 255, 255), 1)
                        cv2.putText(
                            terminal,
                            label,
                            (x0 + 12, y0 + 26),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        self.button_rects[action] = (
                            x0,
                            cameras_panel.shape[0] + y0,
                            x1,
                            cameras_panel.shape[0] + y1,
                        )
                        bx += btn_w + gap

                panel = cv2.vconcat([cameras_panel, terminal])
                disp = panel
                if self._preview_max_width and disp.shape[1] > self._preview_max_width:
                    scale = self._preview_max_width / float(disp.shape[1])
                    self._disp_scale = scale
                    disp = cv2.resize(
                        disp,
                        (int(disp.shape[1] * scale), int(disp.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    self._disp_scale = 1.0
                cv2.imshow(self._window_name, disp)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    self._shared.stop.set()
                    break

                fps_n += 1
                now = time.perf_counter()
                if now - fps_t0 >= 1.0:
                    self.preview_hz = fps_n / (now - fps_t0)
                    fps_n = 0
                    fps_t0 = now
                elapsed = now - t0
                sleep_s = self._period - elapsed
                if sleep_s > 0 and not self._shared.stop.wait(timeout=sleep_s):
                    pass
        except Exception:
            log.exception("PreviewGUI error")
        finally:
            cv2.destroyWindow(self._window_name)
            log.info("PreviewGUI stop")


@dataclass
class VisionPipeline:
    """Starts worker threads and blocks on ControlLoop in the main thread."""

    shared: VisionSharedState
    top_camera: Optional[CameraReader]
    side_camera: Optional[CameraReader]
    top_detection: Optional[AprilTagDetectionWorker]
    side_detection: Optional[AprilTagDetectionWorker]
    control: ControlLoop
    preview: Optional[PreviewGUI] = None

    def start_workers(self) -> None:
        for worker in (self.top_camera, self.side_camera, self.top_detection, self.side_detection):
            if worker:
                worker.start()
        if self.preview:
            self.preview.start()

    def stop(self) -> None:
        self.shared.stop.set()
        for t in (self.top_camera, self.side_camera, self.top_detection, self.side_detection):
            if t:
                t.join(timeout=3.0)
        if self.preview and self.preview.is_alive():
            self.preview.join(timeout=3.0)
        cv2.destroyAllWindows()

    def run(self) -> None:
        self.start_workers()
        try:
            self.control.spin()
        finally:
            self.stop()


def build_vision_pipeline(
    cfg: AppConfig,
    pipeline_cfg: PipelineConfig,
    top_source_factory: Optional[Callable[[], VideoSource]],
    side_source_factory: Callable[[], VideoSource],
    on_control_tick: Callable[[DroneState, TopPoseObservation, SideCorrectionObservation, float], FlightHudSnapshot],
    *,
    preview_enabled: bool = True,
    top_label: str = "TOP",
    side_label: str = "SIDE",
    preview_max_width: int = 1400,
    odometry_provider: Optional[Callable[[], OdometrySample | None]] = None,
) -> VisionPipeline:
    shared = VisionSharedState()
    top_camera: Optional[CameraReader] = None
    top_detection: Optional[AprilTagDetectionWorker] = None
    if top_source_factory is not None:
        top_camera = CameraReader("TopCamera", TOP, top_source_factory, shared)
        top_detection = AprilTagDetectionWorker(
            "TopAprilTag",
            TOP,
            pipeline_cfg.detection_hz_top,
            shared,
            apriltag_for_role(cfg, TOP),
            cfg.cameras,
            mode=TOP,
        )
    side_camera = CameraReader("SideCamera", SIDE, side_source_factory, shared)
    side_detection = AprilTagDetectionWorker(
        "SideAprilTag",
        SIDE,
        pipeline_cfg.detection_hz_side,
        shared,
        apriltag_for_role(cfg, SIDE),
        cfg.cameras,
        mode=SIDE,
    )
    control = ControlLoop(
        shared,
        pipeline_cfg,
        StateEstimator(cfg.tracking, cfg.stabilization),
        on_tick=on_control_tick,
        odometry_provider=odometry_provider,
        top_detection=top_detection,
        side_detection=side_detection,
        top_camera=top_camera,
        side_camera=side_camera,
    )
    preview: Optional[PreviewGUI] = None
    if preview_enabled:
        preview = PreviewGUI(
            shared,
            pipeline_cfg,
            top_label=top_label,
            side_label=side_label,
            preview_max_width=preview_max_width,
            control_enabled=cfg.control_enabled,
            top_enabled=top_camera is not None,
            side_enabled=side_camera is not None,
        )
    return VisionPipeline(
        shared=shared,
        top_camera=top_camera,
        side_camera=side_camera,
        top_detection=top_detection,
        side_detection=side_detection,
        control=control,
        preview=preview,
    )


# Backward-compatible alias (legacy detection thread name)
ArucoDetectionWorker = AprilTagDetectionWorker
ArucoDetectorThread = AprilTagDetectionWorker