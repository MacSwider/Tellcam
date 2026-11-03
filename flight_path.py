"""
Trasa lotu z pliku JSON: punkty główne (x, y, z) + automatycznie wstawiane punkty pomocnicze.

Układ współrzędnych:
- x, y [m] — płaszczyzna kamery TOP (x w prawo, y w dół obrazu), (0, 0) = punkt pod osią optyczną kamery,
- z [m] — wysokość nad podłogą.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

Point3 = Tuple[float, float, float]


@dataclass(frozen=True)
class PathPoint:
    xyz: Point3
    is_main: bool
    main_index: int  # indeks punktu głównego, do którego zmierza ten odcinek


@dataclass
class FlightPath:
    name: str
    main_points: List[Point3]
    points: List[PathPoint]
    loop: bool
    acceptance_radius_m: float
    helper_acceptance_radius_m: float
    hold_at_main_s: float


def _parse_point(raw, idx: int) -> Point3:
    if isinstance(raw, dict):
        try:
            return float(raw["x"]), float(raw["y"]), float(raw["z"])
        except KeyError as e:
            raise ValueError(f"Punkt {idx}: brak współrzędnej {e}") from e
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        return float(raw[0]), float(raw[1]), float(raw[2])
    raise ValueError(f"Punkt {idx}: oczekiwano {{x, y, z}} lub [x, y, z], jest {raw!r}")


def densify(main_points: List[Point3], spacing_m: float, closed: bool) -> List[PathPoint]:
    """Między kolejnymi punktami głównymi wstawia punkty pomocnicze co ~spacing_m."""
    if not main_points:
        return []
    out: List[PathPoint] = [PathPoint(main_points[0], True, 0)]
    segments = list(range(1, len(main_points)))
    if closed and len(main_points) > 2:
        segments.append(0)
    for target_idx in segments:
        a = out[-1].xyz
        b = main_points[target_idx]
        dist = math.dist(a, b)
        n = int(dist // spacing_m) if spacing_m > 0 else 0
        if n > 0 and dist - n * spacing_m < 0.5 * spacing_m:
            n -= 1
        for k in range(1, n + 1):
            t = k * spacing_m / dist
            out.append(
                PathPoint(tuple(a[i] + (b[i] - a[i]) * t for i in range(3)), False, target_idx)  # type: ignore[arg-type]
            )
        out.append(PathPoint(b, True, target_idx))
    return out


def load_path(path: str | Path) -> FlightPath:
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    raw_points = data.get("points") or []
    main = [_parse_point(r, i) for i, r in enumerate(raw_points)]
    if not main:
        raise ValueError(f"{p}: trasa nie zawiera punktów")
    loop = bool(data.get("loop", False))
    closed = bool(data.get("closed", loop))
    spacing = float(data.get("helper_spacing_m", 0.15))
    return FlightPath(
        name=str(data.get("name", p.stem)),
        main_points=main,
        points=densify(main, spacing, closed),
        loop=loop,
        acceptance_radius_m=float(data.get("acceptance_radius_m", 0.08)),
        helper_acceptance_radius_m=float(data.get("helper_acceptance_radius_m", 0.15)),
        hold_at_main_s=float(data.get("hold_at_main_s", 1.0)),
    )


class PathFollower:
    """Wybiera bieżący punkt docelowy; przechodzi dalej po wejściu w promień akceptacji."""

    def __init__(self, path: FlightPath) -> None:
        self.path = path
        self.index = 0
        self.finished = False
        self._hold_left_s: Optional[float] = None

    def reset(self) -> None:
        self.index = 0
        self.finished = False
        self._hold_left_s = None

    @property
    def target(self) -> PathPoint:
        return self.path.points[self.index]

    def update(self, x_m: float, y_m: float, alt_m: float, dt: float) -> PathPoint:
        if self.finished:
            return self.target
        tgt = self.target
        radius = self.path.acceptance_radius_m if tgt.is_main else self.path.helper_acceptance_radius_m
        if math.dist((x_m, y_m, alt_m), tgt.xyz) > radius:
            self._hold_left_s = None
            return tgt
        if tgt.is_main and self.path.hold_at_main_s > 0:
            if self._hold_left_s is None:
                self._hold_left_s = self.path.hold_at_main_s
            self._hold_left_s -= dt
            if self._hold_left_s > 0:
                return tgt
        self._hold_left_s = None
        if self.index + 1 < len(self.path.points):
            self.index += 1
        elif self.path.loop:
            pts = self.path.points
            self.index = 1 if len(pts) > 1 and pts[-1].xyz == pts[0].xyz else 0
        else:
            self.finished = True
        return self.target
