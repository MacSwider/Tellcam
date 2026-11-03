"""
Arkusz A4 (PDF, 300 dpi) z markerami ID 0–4 do wydruku w skali 1:1.

  python generate_markers.py --family aruco_4x4_50 --size-mm 40

--size-mm to bok czarnego kwadratu (= apriltag.marker_length_m w configu).
Drukować w „rzeczywistym rozmiarze” / 100%, bez „dopasuj do strony”; sprawdzić linijką odcinek 100 mm.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from fiducial_detector import marker_dictionary

DPI = 300
A4_MM = (210, 297)
_CELLS = {"aruco_4x4_50": 6, "aruco_5x5_50": 7, "tag36h11": 8, "tag36h10": 8, "tag25h9": 7, "tag16h5": 6}


def mm(v: float) -> int:
    return int(round(v / 25.4 * DPI))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--family", default="aruco_4x4_50")
    p.add_argument("--size-mm", type=float, default=40.0)
    p.add_argument("--ids", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = p.parse_args()

    dictionary = marker_dictionary(args.family)
    cells = _CELLS.get(args.family.lower(), 8)
    side = mm(args.size_mm)
    margin = max(mm(args.size_mm / cells), mm(5))  # biała ramka: co najmniej 1 pole
    tile = side + 2 * margin

    page = np.full((mm(A4_MM[1]), mm(A4_MM[0])), 255, np.uint8)
    x0, y0 = mm(15), mm(30)
    gap = mm(10)
    x, y = x0, y0
    for mid in args.ids:
        marker = cv2.aruco.generateImageMarker(dictionary, mid, cells * 10)
        marker = cv2.resize(marker, (side, side), interpolation=cv2.INTER_NEAREST)
        if x + tile > page.shape[1] - mm(10):
            x, y = x0, y + tile + gap + mm(8)
        page[y + margin : y + margin + side, x + margin : x + margin + side] = marker
        cv2.rectangle(page, (x, y), (x + tile, y + tile), 180, 2)  # linia cięcia
        cv2.putText(page, f"ID {mid}", (x, y + tile + mm(6)), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3)
        x += tile + gap

    by = page.shape[0] - mm(30)
    cv2.line(page, (x0, by), (x0 + mm(100), by), 0, 4)
    for t in (x0, x0 + mm(100)):
        cv2.line(page, (t, by - mm(3)), (t, by + mm(3)), 0, 4)
    cv2.putText(page, "100 mm - sprawdz linijka", (x0, by - mm(5)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3)
    cv2.putText(
        page,
        f"{args.family}, czarny kwadrat {args.size_mm:g} mm, drukuj w skali 100%",
        (x0, mm(15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        0,
        3,
    )

    out = Path(__file__).resolve().parent / f"markers_{args.family}_{args.size_mm:g}mm.pdf"
    Image.fromarray(page).save(out, resolution=DPI)
    print(out)


if __name__ == "__main__":
    main()
