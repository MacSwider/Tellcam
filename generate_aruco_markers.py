"""Generuje markery ArUco ID 0–4 (zgodne z OpenCV) do wydruku."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from config import ArucoConfig, load_config


def _generate_sheet(
    dictionary_name: str,
    marker_ids: tuple[int, ...],
    marker_mm: float,
    dpi: int,
    quiet_mm: float,
) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    px_per_mm = dpi / 25.4
    side_px = max(40, int(round(marker_mm * px_per_mm)))
    quiet_px = max(8, int(round(quiet_mm * px_per_mm)))
    cell = side_px + 2 * quiet_px
    cols = min(3, len(marker_ids))
    rows = (len(marker_ids) + cols - 1) // cols
    sheet = np.ones((rows * cell, cols * cell), dtype=np.uint8) * 255

    for idx, mid in enumerate(marker_ids):
        inner = cv2.aruco.generateImageMarker(dictionary, mid, side_px, borderBits=1)
        r, c = divmod(idx, cols)
        y0, x0 = r * cell + quiet_px, c * cell + quiet_px
        sheet[y0 : y0 + side_px, x0 : x0 + side_px] = inner
        label_y = r * cell + cell - 8
        label_x = c * cell + 8
        cv2.putText(
            sheet,
            f"ID {mid}",
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            0,
            1,
            cv2.LINE_AA,
        )
    return sheet


def main() -> int:
    p = argparse.ArgumentParser(description="Generuj arkusz markerów ArUco 0–4 (OpenCV)")
    p.add_argument("--config", default=None, help="JSON config (sekcja aruco)")
    p.add_argument("--output", default="aruco_markers_print.png", help="Plik wyjściowy PNG")
    p.add_argument("--dpi", type=int, default=300, help="Rozdzielczość wydruku")
    p.add_argument("--quiet-mm", type=float, default=3.0, help="Biała ramka wokół markera [mm]")
    args = p.parse_args()

    cfg = load_config(args.config)
    ar = cfg.aruco
    sheet = _generate_sheet(
        ar.dictionary_name,
        tuple(int(x) for x in ar.marker_ids),
        ar.marker_length_m * 1000.0,
        args.dpi,
        args.quiet_mm,
    )
    out = Path(args.output)
    cv2.imwrite(str(out), sheet)
    print(f"Zapisano: {out.resolve()}")
    print(f"Słownik: {ar.dictionary_name}, bok markera: {ar.marker_length_m * 1000:.0f} mm")
    print("Wydrukuj w skali 1:1 (bez skalowania w drukarce).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
