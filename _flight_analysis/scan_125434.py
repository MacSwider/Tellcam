"""Scan Tellcam HUD states in 12-54-34.mkv."""
from __future__ import annotations

import cv2
import numpy as np

VIDEO = r"c:\Users\mix13\Videos\2026-06-16 12-54-34.mkv"


def tellcam_visible(frame: np.ndarray) -> bool:
    title = frame[0:36, 0:280]
    gray = cv2.cvtColor(title, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) > 25 and gray.std() > 20


def hud_layout(w: int, h: int) -> dict[str, int]:
    hy = int(h * 0.58)
    gap = 10
    card_w = (w - gap * 4) // 3
    y0 = hy + 84
    return {
        "hy": hy,
        "y0": y0,
        "x_top": gap,
        "x_side": gap * 2 + card_w,
        "card_w": card_w,
    }


def dot_green(frame: np.ndarray, x: int, y: int) -> bool:
    b, g, r = (int(v) for v in frame[y, x])
    return g > 150 and r < 120


def phase(frame: np.ndarray, hy: int) -> str:
    roi = frame[hy : hy + 40, 10:240]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 40, 255])).sum()
    blue = cv2.inRange(hsv, np.array([95, 120, 120]), np.array([125, 255, 255])).sum()
    orange = cv2.inRange(hsv, np.array([10, 120, 120]), np.array([28, 255, 255])).sum()
    if orange > max(white, blue) and orange > 500:
        return "TRAVEL"
    if white > blue and white > 800:
        return "HOLD"
    if blue > 500:
        return "CLIMB"
    return "OTHER"


def main() -> None:
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows: list[tuple] = []
    for sec in range(int(n / fps) + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
        ok, frame = cap.read()
        if not ok:
            break
        if not tellcam_visible(frame):
            rows.append((sec, "NO_UI", False, False, "-", "-"))
            continue
        h, w = frame.shape[:2]
        lay = hud_layout(w, h)
        y0 = lay["y0"]
        tag0 = dot_green(frame, lay["x_top"] + 14, y0 + 38)
        side = dot_green(frame, lay["x_side"] + 14, y0 + 38)
        # tracking label colors (OK/Memory/Lost)
        top_trk_px = lay["x_top"] + lay["card_w"] - 50
        side_trk_px = lay["x_side"] + lay["card_w"] - 50
        b, g, r = (int(v) for v in frame[y0 + 38, top_trk_px])
        top_trk = "OK" if g > 150 and r < 120 else ("MEM" if g > 100 and r > 130 else "LOST")
        b, g, r = (int(v) for v in frame[y0 + 38, side_trk_px])
        side_trk = "OK" if g > 150 and r < 120 else ("MEM" if g > 100 and r > 130 else "LOST")
        rows.append((sec, phase(frame, lay["hy"]), tag0, side, top_trk, side_trk))
    cap.release()

    print("sec  phase   TAG0  SIDEtags  top_trk side_trk")
    for row in rows:
        sec, ph, t0, sd, tt, st = row
        if ph == "NO_UI":
            print(f"{sec:3d}  (not Tellcam)")
            continue
        print(
            f"{sec:3d}  {ph:5s}  {'YES' if t0 else 'NO':3s}   "
            f"{'YES' if sd else 'NO':3s}       {tt:4s}    {st}"
        )

    vis = [r for r in rows if r[1] != "NO_UI"]
    print()
    print(f"Tellcam visible: {len(vis)}s / {len(rows)}s")
    print(f"TAG 0 detected: {sum(1 for r in vis if r[2])}s")
    print(f"SIDE tags seen: {sum(1 for r in vis if r[3])}s")
    print(f"TOP tracking OK: {sum(1 for r in vis if r[4]=='OK')}s")
    print(f"SIDE tracking OK: {sum(1 for r in vis if r[5]=='OK')}s")
    hold = [r for r in vis if r[1] == "HOLD"]
    print(f"HOLD phase: {len(hold)}s, TAG0 in HOLD: {sum(1 for r in hold if r[2])}s")


if __name__ == "__main__":
    main()
