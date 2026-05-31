"""
Źródła obrazu dla trybu wizji: USB, zrzut monitora (mss), region wybranego okna (PyGetWindow + mss).
Przydatne do testów AprilTag z obrazem z przeglądarki (okno z kartą) zamiast z kamery USB.
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
    """Zwraca niepuste tytuły okien (do doboru --window-title)."""
    try:
        import pygetwindow as gw
    except ImportError as e:
        raise RuntimeError("Zainstaluj PyGetWindow: pip install PyGetWindow") from e
    titles = []
    for w in gw.getAllWindows():
        t = (w.title or "").strip()
        if t:
            titles.append(t)
        if len(titles) >= max_titles:
            break
    return titles


def probe_usb_camera_indices(max_index: int = 12) -> List[int]:
    """Indeksy kamer USB, które da się otworzyć i zwracają klatkę."""
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
    """(index, etykieta) dla mss — index 0 = wirtualny pulpit, 1+ = fizyczne monitory."""
    import mss

    out: List[Tuple[int, str]] = []
    with mss.mss() as sct:
        for i in range(len(sct.monitors)):
            m = sct.monitors[i]
            w, h = int(m.get("width", 0)), int(m.get("height", 0))
            if i == 0:
                out.append((0, f"Wszystkie ekrany ({w}×{h})"))
            else:
                out.append((i, f"Monitor {i} ({w}×{h})"))
    return out


def _grab_region_bgr(left: int, top: int, width: int, height: int) -> Optional[np.ndarray]:
    if width <= 0 or height <= 0:
        return None
    try:
        import mss
    except ImportError as e:
        raise RuntimeError("Zainstaluj mss: pip install mss") from e
    with mss.mss() as sct:
        region = {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}
        shot = np.array(sct.grab(region))
    if shot.size == 0:
        return None
    return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)


class VideoSource(ABC):
    @abstractmethod
    def read_bgr(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Zwraca (ok, klatka BGR lub None)."""

    @abstractmethod
    def close(self) -> None:
        pass


class UsbCeilingSource(VideoSource):
    """Jedna kamera USB."""

    def __init__(self, cfg: CameraConfig, camera_index: Optional[int] = None) -> None:
        self._cfg = cfg
        self._index = int(camera_index if camera_index is not None else cfg.index_ceiling)
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
            raise RuntimeError(f"Nie otwarto kamery USB (index={self._index})")
        fourcc = (self._cfg.fourcc or "").strip().upper()
        if fourcc and len(fourcc) == 4:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
        buf = int(self._cfg.buffer_size)
        if buf > 0:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, buf)
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        log.info(
            "USB cam index=%s requested=%sx%s actual=%sx%s fps=%.1f fourcc=%s",
            self._index,
            self._cfg.width,
            self._cfg.height,
            actual_w,
            actual_h,
            actual_fps,
            fourcc or "auto",
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
    """Pełny zrzut wybranego monitora (mss: 1 = pierwszy ekran, 0 = wirtualny „wszystkie”)."""

    def __init__(self, monitor: int = 1) -> None:
        self._monitor = int(monitor)
        self._sct = None

    def open(self) -> None:
        try:
            import mss
        except ImportError as e:
            raise RuntimeError("Zainstaluj mss: pip install mss") from e
        self._sct = mss.mss()
        if self._monitor < 0 or self._monitor >= len(self._sct.monitors):
            raise RuntimeError(
                f"monitor={self._monitor} poza zakresem (dostępne 0..{len(self._sct.monitors) - 1})"
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
    Zrzut prostokąta okna, którego tytuł zawiera podany fragment (bez rozróżniania wielkości liter).
    Okno nie może być zminimalizowane (na Windows zwykle wymagane jest widoczne na pulpicie).
    """

    def __init__(self, title_contains: str, match_index: int = 0) -> None:
        self._needle = title_contains.strip().lower()
        self._match_index = int(match_index)
        self._win = None

    def open(self) -> None:
        try:
            import pygetwindow as gw
        except ImportError as e:
            raise RuntimeError("Zainstaluj PyGetWindow: pip install PyGetWindow") from e
        if not self._needle:
            raise RuntimeError("Podaj niepusty --window-title (fragment tytułu okna)")
        matches = [
            w
            for w in gw.getAllWindows()
            if w.title and self._needle in w.title.strip().lower()
        ]
        if not matches:
            raise RuntimeError(f"Brak okna z tytułem zawierającym: {self._needle!r}")
        if self._match_index < 0 or self._match_index >= len(matches):
            raise RuntimeError(
                f"--window-match-index={self._match_index} poza zakresem (znaleziono {len(matches)} okien)"
            )
        self._win = matches[self._match_index]
        log.info("Powiązano okno: %s", self._win.title)

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
    screen_monitor: int = 1,
    window_title: str = "",
    window_match_index: int = 0,
) -> VideoSource:
    m = mode.strip().lower()
    if m in ("usb", "camera", "webcam"):
        src: VideoSource = UsbCeilingSource(cfg, camera_index=camera_index)
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
    raise ValueError(f"Nieznany tryb źródła: {mode!r} (usb | screen | window)")
