"""
Wielowątkowy pipeline wizji Tellcam: kamery, detekcja ArUco, sterowanie, podgląd.

- CameraReader: tylko read() @ max FPS, nadpisuje najnowszą klatkę
- ArucoDetectionWorker: detekcja @ stałej częstotliwości na najświeższej klatce
- ControlLoop (wątek główny): PID / fuzja @ control_hz, bez ArUco
- PreviewGUI: podgląd @ preview_hz, bez detekcji
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

from aruco_detector import (
    ArucoDetector,
    MarkerOverlay,
    SideCorrectionObservation,
    TopPoseObservation,
)
from config import AppConfig, CameraConfig, PipelineConfig as ConfigPipelineConfig
from state_estimator import DroneState, StateEstimator
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


@dataclass
class MarkerOverlaySlot:
    overlays: Tuple[MarkerOverlay, ...] = ()
    timestamp: float = 0.0


@dataclass
class PoseSlot:
    """Ostatnia obserwacja pozy (dane do sterowania)."""
    top: TopPoseObservation = field(default_factory=lambda: TopPoseObservation(ok=False))
    side: SideCorrectionObservation = field(default_factory=lambda: SideCorrectionObservation(ok=False))
    top_timestamp: float = 0.0
    side_timestamp: float = 0.0
    top_seq: int = 0
    side_seq: int = 0


@dataclass
class StatusSnapshot:
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
    """Współdzielony stan między wątkami (lock per slot)."""

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
        self.button_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._button_lock = threading.Lock()

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

    def snapshot_frame(self, camera_id: str) -> Tuple[Optional[np.ndarray], float, int]:
        with self._frame_locks[camera_id]:
            slot = self.frames[camera_id]
            if slot.frame is None:
                return None, 0.0, 0
            return slot.frame.copy(), slot.timestamp, slot.seq

    def peek_frame(self, camera_id: str) -> Tuple[Optional[np.ndarray], float]:
        """Podgląd: kopia klatki bez blokowania detekcji długo."""
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
    """Wątek kamery: wyłącznie read() @ max FPS, nadpisanie najnowszej klatki."""

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
            log.exception("[%s] CameraReader błąd", self.name)
        finally:
            if self._source:
                self._source.close()
            log.info("[%s] CameraReader stop (reads=%s)", self.name, self.read_count)

    def close(self) -> None:
        if self._source:
            self._source.close()


class ArucoDetectionWorker(threading.Thread):
    """
    Wątek detekcji: stała częstotliwość, zawsze najświeższa klatka z CameraReader.
    Alias specyfikacji: ArucoDetectorThread (unika kolizji z aruco_detector.ArucoDetector).
    """

    def __init__(
        self,
        name: str,
        camera_id: str,
        detection_hz: float,
        shared: VisionSharedState,
        aruco_cfg,
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
        self._detector = ArucoDetector(aruco_cfg, camera_cfg)
        self.detection_count = 0
        self.detection_hz = 0.0

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
        log.info("[%s] ArucoDetectionWorker start @ %.1f Hz", self.name, 1.0 / self._period)
        fps_n = 0
        fps_t0 = time.perf_counter()
        try:
            while not self._shared.stop.is_set():
                t0 = time.perf_counter()
                frame, _, _ = self._shared.snapshot_frame(self._camera_id)
                if frame is not None:
                    obs = self._detect(frame)
                    self._publish(obs)
                    self.detection_count += 1
                    fps_n += 1
                elapsed = time.perf_counter() - t0
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
            log.exception("[%s] ArucoDetectionWorker błąd", self.name)
        finally:
            log.info("[%s] ArucoDetectionWorker stop (detections=%s)", self.name, self.detection_count)


class ControlLoop:
    """Pętla sterowania @ control_hz — bez ArUco, tylko odczyt pozy ze współdzielonego stanu."""

    def __init__(
        self,
        shared: VisionSharedState,
        pipeline_cfg: PipelineConfig,
        estimator: StateEstimator,
        *,
        on_tick: Optional[Callable[[DroneState, TopPoseObservation, SideCorrectionObservation, float], List[str]]] = None,
        top_detection: Optional[ArucoDetectionWorker] = None,
        side_detection: Optional[ArucoDetectionWorker] = None,
        top_camera: Optional[CameraReader] = None,
        side_camera: Optional[CameraReader] = None,
    ) -> None:
        self._shared = shared
        self._period = 1.0 / max(pipeline_cfg.control_hz, 1.0)
        self._estimator = estimator
        self._on_tick = on_tick
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

            top_obs, side_obs, top_ts, side_ts = self._shared.read_poses()
            state = self._estimator.update(top_obs, side_obs)

            now = time.perf_counter()
            lines: List[str] = []
            if self._on_tick:
                lines = self._on_tick(state, top_obs, side_obs, dt)

            with self._shared._frame_locks[TOP]:
                top_frame_ts = self._shared.frames[TOP].timestamp
            with self._shared._frame_locks[SIDE]:
                side_frame_ts = self._shared.frames[SIDE].timestamp

            snap = StatusSnapshot(
                lines=lines,
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
    """Rysuje obrysy markerów na kopii klatki (wątek podglądu, bez detekcji)."""
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


class PreviewGUI(threading.Thread):
    """Podgląd @ preview_hz — resize/imshow + obrysy z ostatniej detekcji."""

    def __init__(
        self,
        shared: VisionSharedState,
        pipeline_cfg: PipelineConfig,
        *,
        window_name: str = "Tellcam — TOP | SIDE | terminal",
        top_label: str = "TOP",
        side_label: str = "SIDE",
        preview_max_width: int = 1400,
        control_enabled: bool = False,
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
        self.button_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.preview_hz = 0.0

    def _on_mouse(self, event, x, y, flags, param) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            try:
                self._shared.mouse_clicks.put_nowait((int(x), int(y)))
            except queue.Full:
                pass

    @staticmethod
    def _add_source_bar(frame: np.ndarray, label: str) -> np.ndarray:
        bar_h = 30
        bar = np.full((bar_h, frame.shape[1], 3), (45, 45, 45), dtype=np.uint8)
        cv2.putText(bar, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 2, cv2.LINE_AA)
        return cv2.vconcat([bar, frame])

    def _frame_for_preview(self, camera_id: str, live: np.ndarray) -> np.ndarray:
        if not self._show_overlay:
            return live
        overlays, overlay_ts = self._shared.peek_marker_overlay(camera_id)
        age = time.perf_counter() - overlay_ts if overlay_ts else 999.0
        return draw_marker_overlays(live, overlays, overlay_age_s=age)

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

                if top_frame is None or side_frame is None:
                    if self._shared.stop.wait(timeout=0.01):
                        break
                    continue

                top_disp = self._frame_for_preview(TOP, top_frame)
                side_disp = self._frame_for_preview(SIDE, side_frame)
                left = self._add_source_bar(top_disp, self._top_label)
                right = self._add_source_bar(side_disp, self._side_label)
                h = min(left.shape[0], right.shape[0])
                lw = cv2.resize(left, (int(left.shape[1] * h / left.shape[0]), h))
                rw = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h))
                cameras_panel = cv2.hconcat([lw, rw])

                header = [
                    f"preview={self.preview_hz:.1f}Hz control={status.control_hz:.1f}Hz",
                    (
                        f"det top={status.top_detection_hz:.1f}Hz side={status.side_detection_hz:.1f}Hz "
                        f"frame_age top={status.top_frame_age_s:.2f}s side={status.side_frame_age_s:.2f}s"
                    ),
                ]
                term_h = 120
                terminal = np.zeros((term_h, cameras_panel.shape[1], 3), dtype=np.uint8)
                all_lines = header + status.lines
                self._draw_overlay(terminal, all_lines)

                btn_w, btn_h = 180, 42
                bx0 = max(10, terminal.shape[1] - btn_w - 12)
                by0 = max(8, term_h - btn_h - 10)
                bx1, by1 = bx0 + btn_w, by0 + btn_h
                rect = (bx0, cameras_panel.shape[0] + by0, bx1, cameras_panel.shape[0] + by1)
                self.button_rect = rect
                self._shared.set_button_rect(rect)

                if self._control_enabled:
                    flight_on = any("flight=ON" in ln for ln in status.lines)
                    btn_color = (0, 180, 0) if not flight_on else (0, 80, 220)
                    cv2.rectangle(terminal, (bx0, by0), (bx1, by1), btn_color, -1)
                    cv2.rectangle(terminal, (bx0, by0), (bx1, by1), (255, 255, 255), 1)
                    label = "START" if not flight_on else "LAND"
                    cv2.putText(
                        terminal,
                        label,
                        (bx0 + 50, by0 + 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                panel = cv2.vconcat([cameras_panel, terminal])
                disp = panel
                if self._preview_max_width and disp.shape[1] > self._preview_max_width:
                    scale = self._preview_max_width / float(disp.shape[1])
                    disp = cv2.resize(
                        disp,
                        (int(disp.shape[1] * scale), int(disp.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
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
            log.exception("PreviewGUI błąd")
        finally:
            cv2.destroyWindow(self._window_name)
            log.info("PreviewGUI stop")


@dataclass
class VisionPipeline:
    """Uruchamia wątki i blokuje na ControlLoop w wątku głównym."""

    shared: VisionSharedState
    top_camera: CameraReader
    side_camera: CameraReader
    top_detection: ArucoDetectionWorker
    side_detection: ArucoDetectionWorker
    control: ControlLoop
    preview: Optional[PreviewGUI] = None

    def start_workers(self) -> None:
        self.top_camera.start()
        self.side_camera.start()
        self.top_detection.start()
        self.side_detection.start()
        if self.preview:
            self.preview.start()

    def stop(self) -> None:
        self.shared.stop.set()
        for t in (self.top_camera, self.side_camera, self.top_detection, self.side_detection):
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
    top_source_factory: Callable[[], VideoSource],
    side_source_factory: Callable[[], VideoSource],
    on_control_tick: Callable[[DroneState, TopPoseObservation, SideCorrectionObservation, float], List[str]],
    *,
    preview_enabled: bool = True,
    top_label: str = "TOP",
    side_label: str = "SIDE",
    preview_max_width: int = 1400,
) -> VisionPipeline:
    shared = VisionSharedState()
    top_camera = CameraReader("TopCamera", TOP, top_source_factory, shared)
    side_camera = CameraReader("SideCamera", SIDE, side_source_factory, shared)
    top_detection = ArucoDetectionWorker(
        "TopAruco",
        TOP,
        pipeline_cfg.detection_hz_top,
        shared,
        cfg.aruco,
        cfg.cameras,
        mode=TOP,
    )
    side_detection = ArucoDetectionWorker(
        "SideAruco",
        SIDE,
        pipeline_cfg.detection_hz_side,
        shared,
        cfg.aruco,
        cfg.cameras,
        mode=SIDE,
    )
    control = ControlLoop(
        shared,
        pipeline_cfg,
        StateEstimator(),
        on_tick=on_control_tick,
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


# Alias zgodny ze specyfikacją architektury (wątek detekcji, nie klasa ArucoDetector)
ArucoDetectorThread = ArucoDetectionWorker