"""
Nawigacja po waypointach (na razie scaffolding pod przyszłe loty po ścieżce).

Domyślnie pętla lotu działa w trybie HOLD (dron trzyma jeden punkt — TOP+SIDE).
Ten moduł dostarcza:
  - `TrajectoryPoint`  – pojedynczy CEL ABSOLUTNY w układzie kamery TOP [m, rad],
  - `FlightPathStep`   – pojedynczy KROK WZGLĘDNY w układzie ciała drona,
  - `load_flight_path` – wczytanie listy kroków z pliku JSON,
  - `steps_to_waypoints` – zamiana kroków względnych na ciąg celów absolutnych,
  - `FutureTrajectoryPlanner` – kolejka waypointów + logika „osiągnięto cel”.

Reprezentacja kroku: wybrano typowany `@dataclass(slots=True)` zamiast „gołych”
krotek. Zajmuje tyle samo pamięci co krotka (dzięki `slots`), ale pola są
nazwane, walidowalne i mają wartości domyślne — czytelniejsze i odporniejsze na
pomyłki kolejności niż `(0.0, 0.0, 0.0, 90.0)`.

Konwencja jednostek: w pliku ścieżki wartość 1.0 oznacza 1 metr (skala
`units_to_meter`). Kąty obrotu podajemy w STOPNIACH (turn_deg), dodatnie = CCW
(w lewo, patrząc z góry) — spójnie z konwencją yaw w detektorze TOP.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class TrajectoryPoint:
    """Cel absolutny w układzie kamery TOP."""
    x_m: float
    y_m: float
    z_m: float
    yaw_rad: float = 0.0


@dataclass(frozen=True, slots=True)
class FlightPathStep:
    """
    Pojedynczy ruch WZGLĘDNY w układzie CIAŁA drona (patrząc z góry):
      forward_m – do przodu (+) / do tyłu (−),
      left_m    – w lewo (+) / w prawo (−),
      up_m      – w górę (+) / w dół (−),
      turn_deg  – obrót yaw, dodatni = CCW (w lewo),
      hold_s    – opcjonalny postój nad punktem po dotarciu [s].
    Wartość 1.0 w forward/left/up = 1 metr (po przeskalowaniu units_to_meter).
    """
    forward_m: float = 0.0
    left_m: float = 0.0
    up_m: float = 0.0
    turn_deg: float = 0.0
    hold_s: float = 0.0

    @classmethod
    def from_mapping(cls, data: dict) -> "FlightPathStep":
        return cls(
            forward_m=float(data.get("forward_m", data.get("forward", 0.0))),
            left_m=float(data.get("left_m", data.get("left", 0.0))),
            up_m=float(data.get("up_m", data.get("up", 0.0))),
            turn_deg=float(data.get("turn_deg", data.get("turn", 0.0))),
            hold_s=float(data.get("hold_s", data.get("hold", 0.0))),
        )


def load_flight_path(path: str | Path) -> List[FlightPathStep]:
    """Wczytaj ścieżkę z JSON.

    Format pliku (lista obiektów):
        [
          {"forward_m": 1.0, "turn_deg": 0.0},
          {"turn_deg": 90.0},
          {"forward_m": 2.0, "up_m": 0.5, "hold_s": 1.0}
        ]
    Akceptujemy też klucz "steps": [...] na najwyższym poziomie.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Plik ścieżki lotu nie istnieje: {p}")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("steps", [])
    if not isinstance(data, list):
        raise ValueError("Ścieżka lotu musi być listą kroków (lub {'steps': [...]}).")
    return [FlightPathStep.from_mapping(item) for item in data]


def steps_to_waypoints(
    start: TrajectoryPoint,
    steps: Sequence[FlightPathStep],
    units_to_meter: float = 1.0,
) -> List[TrajectoryPoint]:
    """
    Zintegruj kroki WZGLĘDNE (układ ciała) do listy celów ABSOLUTNYCH (układ TOP).

    Transformacja zgodna z kontrolerem (control_frame="top"): oś +Y to „przód”,
    oś +X to „lewo/bok”, kurs yaw obraca ciało w płaszczyźnie obrazu. Najpierw
    aplikujemy obrót (turn_deg), potem ruch w nowym kursie.
    """
    waypoints: List[TrajectoryPoint] = []
    x, y, z, yaw = start.x_m, start.y_m, start.z_m, start.yaw_rad
    for step in steps:
        yaw = math.atan2(math.sin(yaw + math.radians(step.turn_deg)),
                         math.cos(yaw + math.radians(step.turn_deg)))
        fwd = step.forward_m * units_to_meter
        lat = step.left_m * units_to_meter
        c, s = math.cos(yaw), math.sin(yaw)
        # (dx, dy) = R(yaw) @ (lat, fwd): lat -> oś X (bok), fwd -> oś Y (przód).
        dx = c * lat - s * fwd
        dy = s * lat + c * fwd
        x += dx
        y += dy
        z += step.up_m * units_to_meter
        waypoints.append(TrajectoryPoint(x_m=x, y_m=y, z_m=z, yaw_rad=yaw))
    return waypoints


class FutureTrajectoryPlanner:
    """
    Kolejka waypointów + bieżący cel. Domyślnie zwraca pojedynczy statyczny cel
    (tryb HOLD). Po wczytaniu ścieżki działa jako prosty sekwencer waypointów.
    """

    def __init__(self, initial_target: TrajectoryPoint) -> None:
        self._queue: "Queue[TrajectoryPoint]" = Queue()
        self._current = initial_target

    def set_static_target(self, target: TrajectoryPoint) -> None:
        self._current = target
        self.clear()

    def enqueue_waypoints(self, trajectory: Iterable[TrajectoryPoint]) -> None:
        for point in trajectory:
            self._queue.put(point)

    def load_path(
        self,
        start: TrajectoryPoint,
        steps: Sequence[FlightPathStep],
        units_to_meter: float = 1.0,
    ) -> List[TrajectoryPoint]:
        """Zamień kroki względne na waypointy i wstaw je do kolejki."""
        waypoints = steps_to_waypoints(start, steps, units_to_meter)
        self.clear()
        self.enqueue_waypoints(waypoints)
        if waypoints:
            self._current = waypoints[0]
        return waypoints

    def load_path_from_file(
        self,
        start: TrajectoryPoint,
        path: str | Path,
        units_to_meter: float = 1.0,
    ) -> List[TrajectoryPoint]:
        return self.load_path(start, load_flight_path(path), units_to_meter)

    def peek_target(self) -> TrajectoryPoint:
        if not self._queue.empty():
            self._current = self._queue.queue[0]
        return self._current

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    def reached(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        yaw_rad: float,
        radius_m: float = 0.15,
        yaw_tol_deg: float = 8.0,
    ) -> bool:
        """Czy dron jest w tolerancji bieżącego waypointu (pozycja + kurs)?"""
        tgt = self.peek_target()
        dist = math.dist((x_m, y_m, z_m), (tgt.x_m, tgt.y_m, tgt.z_m))
        dyaw = abs(math.atan2(math.sin(yaw_rad - tgt.yaw_rad),
                              math.cos(yaw_rad - tgt.yaw_rad)))
        return dist <= radius_m and math.degrees(dyaw) <= yaw_tol_deg

    def advance_if_reached(self) -> None:
        if not self._queue.empty():
            self._current = self._queue.get()

    def clear(self) -> None:
        while not self._queue.empty():
            self._queue.get()
