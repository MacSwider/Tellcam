"""
Lightweight flight status panel (OpenCV) — replaces a long text terminal.
Shows: flight phase, Tello state (battery), TAG 0 / SIDE tag detection,
compact pose, and optional alert. PID details only when gui.show_pid_debug.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import cv2
import numpy as np

# BGR colors
_BG = (28, 28, 32)
_CARD = (42, 42, 48)
_BORDER = (70, 70, 78)
_TEXT = (230, 230, 235)
_MUTED = (150, 150, 158)
_OK = (80, 200, 90)
_WARN = (60, 180, 255)
_BAD = (70, 70, 240)
_ACCENT = (255, 170, 50)
_FLIGHT_ON = (90, 210, 110)
_PHASE_LABELS = {
    "IDLE": "Idle",
    "CLIMB": "Climbing",
    "HOLD": "Position hold",
    "TRAVEL": "Route flight",
    "LANDING": "Landing",
}
_TRACK_LABELS = {
    "TRACKING": "OK",
    "HOLDING_LAST": "Memory",
    "LOST": "Lost",
    "TIMED_OUT": "Timeout",
}


@dataclass
class FlightHudSnapshot:
    phase: str = "IDLE"
    phase_reason: str = ""
    preview_mode: bool = False
    tello_connected: bool = False
    tello_flying: bool = False
    battery_pct: int | None = None
    top_state: str = "LOST"
    top_tag0: bool = False
    top_tags: Tuple[int, ...] = ()
    top_quality: float = 0.0
    side_state: str = "LOST"
    side_tags: Tuple[int, ...] = ()
    side_active_tag: int | None = None
    side_quality: float = 0.0
    tracking_valid: bool = False
    active_axes: str = "-"
    pose_x_m: float = 0.0
    pose_y_m: float = 0.0
    pose_z_m: float = 0.0
    travel_progress: str = ""  # e.g. "2/6"
    geofence_active: bool = False
    geofence_ok: bool = True
    alert: str = ""
    control_hz: float = 0.0
    preview_hz: float = 0.0
    top_capture: str = ""
    side_capture: str = ""
    top_detection: str = ""
    side_detection: str = ""
    debug_lines: List[str] = field(default_factory=list)


def _track_color(state: str) -> Tuple[int, int, int]:
    if state == "TRACKING":
        return _OK
    if state == "HOLDING_LAST":
        return _WARN
    return _BAD


def _truncate(text: str, max_len: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def _draw_label(
    img: np.ndarray, text: str, x: int, y: int, color=_TEXT, scale=0.5, thick=1
) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _draw_card(img: np.ndarray, x: int, y: int, w: int, h: int, title: str) -> None:
    cv2.rectangle(img, (x, y), (x + w, y + h), _CARD, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), _BORDER, 1)
    _draw_label(img, title, x + 10, y + 20, _MUTED, 0.45, 1)


def _draw_status_dot(
    img: np.ndarray, cx: int, cy: int, color: Tuple[int, int, int], r: int = 6
) -> None:
    cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, _BORDER, 1, cv2.LINE_AA)


def _draw_battery(img: np.ndarray, x: int, y: int, pct: int | None, connected: bool) -> None:
    if not connected:
        _draw_label(img, "Tello: OFF", x, y, _BAD, 0.52, 1)
        return
    label = "Tello: OK"
    col = _OK
    if pct is not None:
        if pct <= 15:
            col = _BAD
        elif pct <= 30:
            col = _WARN
        label = f"Tello: OK  {pct}%"
    _draw_label(img, label, x, y, col, 0.52, 1)
    if pct is None:
        return
    bx, by, bw, bh = x, y + 10, 72, 10
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), _BORDER, 1)
    fill = int(bw * max(0, min(100, pct)) / 100.0)
    if fill > 0:
        cv2.rectangle(img, (bx + 1, by + 1), (bx + fill - 1, by + bh - 1), col, -1)


def draw_flight_hud(panel_w: int, hud: FlightHudSnapshot) -> np.ndarray:
    """Render a fixed, readable status panel below camera previews."""
    debug_h = 0
    if hud.debug_lines:
        debug_h = min(100, 16 * len(hud.debug_lines) + 10)
    alert_h = 22 if hud.alert else 0
    hud_h = 168 + alert_h + debug_h
    img = np.full((hud_h, panel_w, 3), _BG, dtype=np.uint8)
    # --- Header: phase + flight + battery ---
    phase_label = _PHASE_LABELS.get(hud.phase, hud.phase)
    phase_col = _ACCENT if hud.phase in ("CLIMB", "TRAVEL") else _TEXT
    if hud.phase == "HOLD" and not hud.tracking_valid:
        phase_col = _WARN
    _draw_label(img, phase_label, 14, 28, phase_col, 0.62, 2)
    reason = _truncate(hud.phase_reason, 52)
    if reason:
        _draw_label(img, reason, 14, 48, _MUTED, 0.42, 1)
    if hud.preview_mode:
        _draw_label(img, "PID PREVIEW (drone on ground)", 14, 66, _WARN, 0.42, 1)
    flight_txt = "FLYING" if hud.tello_flying else "ON GROUND"
    flight_col = _FLIGHT_ON if hud.tello_flying else _MUTED
    _draw_label(img, flight_txt, panel_w - 200, 28, flight_col, 0.55, 2)
    _draw_battery(img, panel_w - 200, 40, hud.battery_pct, hud.tello_connected)
    if hud.travel_progress:
        _draw_label(img, f"Waypoint {hud.travel_progress}", panel_w - 200, 72, _ACCENT, 0.45, 1)
    if hud.geofence_active:
        gf_txt = "Geofence: OK" if hud.geofence_ok else "Geofence: OUT"
        gf_col = _OK if hud.geofence_ok else _BAD
        _draw_label(img, gf_txt, panel_w - 200, 88 if hud.travel_progress else 72, gf_col, 0.42, 1)
    # --- TOP / SIDE / POSE cards ---
    y0 = 84
    gap = 10
    card_h = 72
    card_w = (panel_w - gap * 4) // 3
    x_top = gap
    x_side = gap * 2 + card_w
    x_pose = gap * 3 + card_w * 2
    _draw_card(img, x_top, y0, card_w, card_h, "TOP CAMERA")
    top_col = _OK if hud.top_tag0 else _BAD
    tag0_txt = "TAG 0: YES" if hud.top_tag0 else "TAG 0: NO"
    _draw_status_dot(img, x_top + 14, y0 + 38, top_col)
    _draw_label(img, tag0_txt, x_top + 26, y0 + 42, top_col, 0.48, 1)
    tags_top = ", ".join(str(t) for t in hud.top_tags) if hud.top_tags else "—"
    _draw_label(img, f"Tags: {tags_top}", x_top + 10, y0 + 56, _TEXT, 0.4, 1)
    if hud.top_capture:
        cap_txt = f"Cap {hud.top_capture}"
        if hud.top_detection and hud.top_detection != hud.top_capture:
            cap_txt += f" | Det {hud.top_detection}"
        _draw_label(img, cap_txt, x_top + 10, y0 + 72, _MUTED, 0.36, 1)
    trk = _TRACK_LABELS.get(hud.top_state, hud.top_state)
    _draw_label(img, trk, x_top + card_w - 58, y0 + 38, _track_color(hud.top_state), 0.42, 1)
    _draw_card(img, x_side, y0, card_w, card_h, "SIDE CAMERA")
    side_has = bool(hud.side_tags)
    side_col = _OK if side_has else _BAD
    _draw_status_dot(img, x_side + 14, y0 + 38, side_col)
    tags_side = ", ".join(str(t) for t in hud.side_tags) if hud.side_tags else "—"
    _draw_label(img, f"Tags: {tags_side}", x_side + 26, y0 + 42, _TEXT, 0.45, 1)
    active = str(hud.side_active_tag) if hud.side_active_tag is not None else "—"
    _draw_label(img, f"Active: {active}", x_side + 10, y0 + 56, _MUTED, 0.4, 1)
    if hud.side_capture:
        cap_txt = f"Cap {hud.side_capture}"
        if hud.side_detection and hud.side_detection != hud.side_capture:
            cap_txt += f" | Det {hud.side_detection}"
        _draw_label(img, cap_txt, x_side + 10, y0 + 72, _MUTED, 0.36, 1)
    trk_s = _TRACK_LABELS.get(hud.side_state, hud.side_state)
    _draw_label(img, trk_s, x_side + card_w - 58, y0 + 38, _track_color(hud.side_state), 0.42, 1)
    _draw_card(img, x_pose, y0, card_w, card_h, "POSE")
    _draw_label(
        img,
        f"x {hud.pose_x_m:+.2f}  y {hud.pose_y_m:+.2f}  z {hud.pose_z_m:+.2f} m",
        x_pose + 10,
        y0 + 38,
        _TEXT,
        0.42,
        1,
    )
    valid_txt = "Tracking: YES" if hud.tracking_valid else "Tracking: NO"
    valid_col = _OK if hud.tracking_valid else _BAD
    _draw_label(img, valid_txt, x_pose + 10, y0 + 52, valid_col, 0.4, 1)
    # --- Alert (single line) ---
    if hud.alert:
        ay0 = y0 + card_h + 8
        cv2.rectangle(img, (8, ay0), (panel_w - 8, ay0 + alert_h - 4), (40, 35, 30), -1)
        _draw_label(img, _truncate(hud.alert, 90), 14, ay0 + 15, _WARN, 0.42, 1)
    # --- Optional PID debug (smaller font) ---
    if hud.debug_lines:
        dy = hud_h - debug_h + 12
        for line in hud.debug_lines[:6]:
            _draw_label(img, line, 12, dy, (120, 200, 200), 0.38, 1)
            dy += 16
    return img
