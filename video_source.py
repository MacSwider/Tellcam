"""
Image sources for vision mode: USB, screen capture (mss), selected window region (PyGetWindow + mss).
Useful for AprilTag tests from browser image (window with tab) instead of USB camera.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import CameraConfig

log = logging.getLogger(__name__)


def list_window_titles(max_titles: int = 200) -> List[str]:
    """Return non-empty window titles (for --window-title selection)."""
    try:
        import pygetwindow as gw
    except ImportError as e:
        raise RuntimeError("Install PyGetWindow: pip install PyGetWindow") from e
    titles = []
    for w in gw.getAllWindows():
        t = (w.title or "").strip()
        if t:
            titles.append(t)
        if len(titles) >= max_titles:
            break
    return titles


def probe_usb_camera_indices(max_index: int = 12) -> List[int]:
    """USB camera indices that can be opened and return a frame."""
    found: List[int] = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, _ = cap.read()
            if not ok:
                for _ in range(3):
                    if cap.grab():
                        ok, _ = cap.retrieve()
                        if ok:
                            break
            if ok:
                found.append(i)
        cap.release()
    return found


def list_mss_monitors_for_ui() -> List[Tuple[int, str]]:
    """(index, label) for mss — index 0 = virtual desktop, 1+ = physical monitors."""
    import mss

    out: List[Tuple[int, str]] = []
    with mss.mss() as sct:
        for i in range(len(sct.monitors)):
            m = sct.monitors[i]
            w, h = int(m.get("width", 0)), int(m.get("height", 0))
            if i == 0:
                out.append((0, f"All screens ({w}×{h})"))
            else:
                out.append((i, f"Monitor {i} ({w}×{h})"))
    return out


def _grab_region_bgr(left: int, top: int, width: int, height: int) -> Optional[np.ndarray]:
    if width <= 0 or height <= 0:
        return None
    try:
        import mss
    except ImportError as e:
        raise RuntimeError("Install mss: pip install mss") from e
    with mss.mss() as sct:
        region = {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}
        shot = np.array(sct.grab(region))
    if shot.size == 0:
        return None
    return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)


class VideoSource(ABC):
    @abstractmethod
    def read_bgr(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return (ok, BGR frame or None)."""

    @abstractmethod
    def close(self) -> None:
        pass


class UsbCeilingSource(VideoSource):
    """Single USB camera."""

    def __init__(
        self,
        cfg: CameraConfig,
        camera_index: Optional[int] = None,
        *,
        camera_role: str = "top",
    ) -> None:
        self._cfg = cfg
        self._index = int(camera_index if camera_index is not None else cfg.index_ceiling)
        self._role = (camera_role or "top").strip().lower()
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        def _open(idx: int) -> cv2.VideoCapture:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
            cap.release()
            return cv2.VideoCapture(idx)

        self._cap = _open(self._index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open USB camera (index={self._index})")
        fourcc = (self._cfg.fourcc or "").strip().upper()
        if fourcc and len(fourcc) == 4:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        req_w, req_h = self._cfg.resolution_for(self._role)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
        buf = int(self._cfg.buffer_size)
        if buf > 0:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, buf)
        self._apply_optional_capture_props()
        self._maybe_autotune_focus()
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        log.info(
            "USB cam index=%s role=%s requested=%sx%s actual=%sx%s fps=%.1f fourcc=%s",
            self._index,
            self._role,
            req_w,
            req_h,
            actual_w,
            actual_h,
            actual_fps,
            fourcc or "auto",
        )
        if actual_w < req_w * 0.9 or actual_h < req_h * 0.9:
            log.warning(
                "USB cam index=%s (%s): driver delivered lower resolution than requested "
                "(%dx%d vs %dx%d) — check USB bandwidth / MJPEG / driver settings.",
                self._index,
                self._role,
                actual_w,
                actual_h,
                req_w,
                req_h,
            )

    def _set_capture_prop(
        self,
        cap: cv2.VideoCapture,
        prop_id: int,
        value: float,
        name: str,
    ) -> None:
        """Set a V4L2/DirectShow property and log whether the driver accepted it."""
        before = float(cap.get(prop_id))
        accepted = bool(cap.set(prop_id, float(value)))
        after = float(cap.get(prop_id))
        log.info(
            "USB cam index=%s (%s): %s requested=%.2f before=%.2f after=%.2f accepted=%s",
            self._index,
            self._role,
            name,
            float(value),
            before,
            after,
            accepted,
        )
        if not accepted or abs(after - float(value)) > max(1.0, abs(float(value)) * 0.05 + 1.0):
            log.warning(
                "USB cam index=%s (%s): %s may be unsupported or read-only on this driver "
                "(OpenCV CAP_PROP; common on fixed-focus fisheye USB cameras).",
                self._index,
                self._role,
                name,
            )

    def _apply_optional_capture_props(self) -> None:
        if not self._cap:
            return
        cap = self._cap
        ae = self._cfg.auto_exposure
        if ae is not None:
            # 0.25 = manual, 0.75 = auto (DirectShow / many UVC drivers).
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if ae else 0.25)
        exposure = self._cfg.capture_prop(self._role, "exposure")
        if exposure is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
        gain = self._cfg.capture_prop(self._role, "gain")
        if gain is not None:
            cap.set(cv2.CAP_PROP_GAIN, float(gain))
        af = self._cfg.auto_focus
        if af is not None:
            # Some DirectShow/UVC drivers use 5 for continuous AF (1 is ignored on many fisheye cams).
            self._set_capture_prop(
                cap,
                cv2.CAP_PROP_AUTOFOCUS,
                2.0 if af else 0.0,
                "AUTOFOCUS",
            )
        if self._cfg.focus is not None and not af:
            self._set_capture_prop(cap, cv2.CAP_PROP_FOCUS, float(self._cfg.focus), "FOCUS")
        brightness = self._cfg.capture_prop(self._role, "brightness")
        if brightness is not None:
            cap.set(cv2.CAP_PROP_BRIGHTNESS, float(brightness))
        contrast = self._cfg.capture_prop(self._role, "contrast")
        if contrast is not None:
            cap.set(cv2.CAP_PROP_CONTRAST, float(contrast))
        sharpness = self._cfg.capture_prop(self._role, "sharpness")
        if sharpness is not None:
            cap.set(cv2.CAP_PROP_SHARPNESS, float(sharpness))

    def _roi_from_norm(
        self, frame: np.ndarray, roi_norm: Tuple[float, float, float, float]
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = (float(v) for v in roi_norm)
        rx0 = int(np.clip(min(x0, x1), 0.0, 1.0) * w)
        ry0 = int(np.clip(min(y0, y1), 0.0, 1.0) * h)
        rx1 = int(np.clip(max(x0, x1), 0.0, 1.0) * w)
        ry1 = int(np.clip(max(y0, y1), 0.0, 1.0) * h)
        if rx1 <= rx0 + 8 or ry1 <= ry0 + 8:
            return frame
        return frame[ry0:ry1, rx0:rx1]

    @staticmethod
    def _sharpness_laplacian(roi_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _should_autotune_focus(self) -> bool:
        if not self._cfg.focus_autotune:
            return False
        roles = tuple((r or "").strip().lower() for r in self._cfg.focus_autotune_cameras)
        if "both" in roles:
            return True
        return self._role in roles

    def _maybe_autotune_focus(self) -> None:
        if not self._cap or not self._should_autotune_focus():
            return
        if self._cfg.auto_focus:
            log.info(
                "USB cam index=%s (%s): hardware auto_focus enabled — skipping software autotune",
                self._index,
                self._role,
            )
            return
        roi = self._cfg.focus_roi_norm
        if not roi or len(roi) != 4:
            roi = (0.30, 0.30, 0.70, 0.70)
        steps = max(3, int(self._cfg.focus_sweep_steps))
        lo = float(self._cfg.focus_sweep_min)
        hi = float(self._cfg.focus_sweep_max)
        settle = max(0, int(self._cfg.focus_settle_frames))
        cap = self._cap
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0.0)
        best_focus: float | None = None
        best_score = -1.0
        for focus_val in np.linspace(lo, hi, steps):
            cap.set(cv2.CAP_PROP_FOCUS, float(focus_val))
            ok = False
            frame = None
            for _ in range(settle + 1):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
            if not ok or frame is None:
                continue
            score = self._sharpness_laplacian(self._roi_from_norm(frame, roi))
            if score > best_score:
                best_score = score
                best_focus = float(focus_val)
        if best_focus is None:
            log.warning(
                "USB cam index=%s (%s): software focus autotune found no valid frame",
                self._index,
                self._role,
            )
            return
        cap.set(cv2.CAP_PROP_FOCUS, best_focus)
        log.info(
            "USB cam index=%s (%s): software focus autotune -> focus=%.1f (sharpness=%.1f)",
            self._index,
            self._role,
            best_focus,
            best_score,
        )

    def read_bgr(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._cap:
            return False, None
        drain = max(0, int(self._cfg.drain_frames_on_read))
        if drain > 0:
            for _ in range(drain):
                if not self._cap.grab():
                    break
            ok, frame = self._cap.retrieve()
        else:
            ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        return True, frame

    def close(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None


class ScreenMonitorSource(VideoSource):
    """Full capture of selected monitor (mss: 1 = first screen, 0 = virtual all)."""

    def __init__(self, monitor: int = 1) -> None:
        self._monitor = int(monitor)
        self._sct = None

    def open(self) -> None:
        try:
            import mss
        except ImportError as e:
            raise RuntimeError("Install mss: pip install mss") from e
        self._sct = mss.mss()
        if self._monitor < 0 or self._monitor >= len(self._sct.monitors):
            raise RuntimeError(
                f"monitor={self._monitor} out of range (available 0..{len(self._sct.monitors) - 1})"
            )

    def read_bgr(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._sct:
            return False, None
        mon = self._sct.monitors[self._monitor]
        shot = np.array(self._sct.grab(mon))
        if shot.size == 0:
            return False, None
        frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
        return True, frame

    def close(self) -> None:
        if self._sct:
            self._sct.close()
            self._sct = None


class WindowTitleSource(VideoSource):
    """
    Capture window rectangle whose title contains the given substring (case-insensitive).
    Window must not be minimized (on Windows usually must be visible on desktop).
    """

    def __init__(self, title_contains: str, match_index: int = 0) -> None:
        self._needle = title_contains.strip().lower()
        self._match_index = int(match_index)
        self._win = None

    def open(self) -> None:
        try:
            import pygetwindow as gw
        except ImportError as e:
            raise RuntimeError("Install PyGetWindow: pip install PyGetWindow") from e
        if not self._needle:
            raise RuntimeError("Provide non-empty --window-title (window title fragment)")
        matches = [
            w
            for w in gw.getAllWindows()
            if w.title and self._needle in w.title.strip().lower()
        ]
        if not matches:
            raise RuntimeError(f"No window with title containing: {self._needle!r}")
        if self._match_index < 0 or self._match_index >= len(matches):
            raise RuntimeError(
                f"--window-match-index={self._match_index} out of range (found {len(matches)} windows)"
            )
        self._win = matches[self._match_index]
        log.info("Bound window: %s", self._win.title)

    def read_bgr(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._win is None:
            return False, None
        try:
            if self._win.isMinimized:
                return False, None
            left, top, w, h = self._win.left, self._win.top, self._win.width, self._win.height
        except Exception:
            return False, None
        try:
            frame = _grab_region_bgr(left, top, w, h)
        except Exception:
            return False, None
        if frame is None:
            return False, None
        return True, frame

    def close(self) -> None:
        self._win = None


def create_video_source(
    mode: str,
    cfg: CameraConfig,
    *,
    camera_index: Optional[int] = None,
    camera_role: str = "top",
    screen_monitor: int = 1,
    window_title: str = "",
    window_match_index: int = 0,
) -> VideoSource:
    m = mode.strip().lower()
    if m in ("usb", "camera", "webcam"):
        src: VideoSource = UsbCeilingSource(cfg, camera_index=camera_index, camera_role=camera_role)
        src.open()
        return src
    if m in ("screen", "monitor", "desktop"):
        src = ScreenMonitorSource(monitor=screen_monitor)
        src.open()
        return src
    if m in ("window", "hwnd_title", "app"):
        src = WindowTitleSource(window_title, match_index=window_match_index)
        src.open()
        return src
    raise ValueError(f"Unknown source mode: {mode!r} (usb | screen | window)")
