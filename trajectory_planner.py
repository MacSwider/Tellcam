"""
Waypoint navigation for TRAVEL phase.

Path file formats (JSON):
  1. Absolute 3D points (preferred) — lab / TOP camera frame:
       {"coordinate_unit": "cm", "points": [{"X": 0, "Y": 0, "Z": 100}, ...]}
     X = lateral (left +), Y = forward (+), Z = up (+). Scaled to metres for control.

  2. Legacy relative body steps:
       {"steps": [{"forward_m": 0.5, "left_m": -0.5}, ...]}

Unit convention: `coordinate_unit` in file ("cm" | "m") or `units_to_meter` from config.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Iterable, List, Sequence


_UNIT_TO_METRE = {"cm": 0.01, "m": 1.0, "mm": 0.001}


@dataclass(frozen=True)
class TrajectoryPoint:
    """Absolute target in TOP camera frame [m, rad]."""
    x_m: float
    y_m: float
    z_m: float
    yaw_rad: float = 0.0


@dataclass(frozen=True, slots=True)
class FlightPathStep:
    """
    Single RELATIVE move in drone BODY frame (legacy format):
      forward_m, left_m, up_m, turn_deg, hold_s.
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


@dataclass
class LoadedFlightPath:
    """Parsed route from JSON — absolute waypoints and/or legacy relative steps."""

    absolute: bool = True
    waypoints: List[TrajectoryPoint] = field(default_factory=list)
    steps: List[FlightPathStep] = field(default_factory=list)
    coordinate_unit: str = "cm"

    @property
    def non_empty(self) -> bool:
        return bool(self.waypoints or self.steps)


def _scale_for_unit(unit: str, units_to_meter: float) -> float:
    key = (unit or "").strip().lower()
    if key in _UNIT_TO_METRE:
        return _UNIT_TO_METRE[key]
    return float(units_to_meter)


def _point_from_mapping(data: dict, scale: float, default_yaw_rad: float) -> TrajectoryPoint:
    x = float(data.get("X", data.get("x", data.get("x_m", 0.0))))
    y = float(data.get("Y", data.get("y", data.get("y_m", 0.0))))
    z = float(data.get("Z", data.get("z", data.get("z_m", 0.0))))
    if "yaw_rad" in data:
        yaw = float(data["yaw_rad"])
    elif "yaw_deg" in data or "Yaw" in data:
        yaw = math.radians(float(data.get("yaw_deg", data.get("Yaw", 0.0))))
    else:
        yaw = default_yaw_rad
    s = float(scale)
    return TrajectoryPoint(x_m=x * s, y_m=y * s, z_m=z * s, yaw_rad=yaw)


def square_clockwise_steps(side_m: float) -> List[FlightPathStep]:
    """Closed square in body frame, clockwise when viewed from above (legacy preset)."""
    s = float(side_m)
    return [
        FlightPathStep(forward_m=s),
        FlightPathStep(left_m=-s),
        FlightPathStep(forward_m=-s),
        FlightPathStep(left_m=s),
    ]


def cumulative_body_offsets(
    steps: Sequence[FlightPathStep],
    units_to_meter: float = 1.0,
) -> List[tuple[float, float]]:
    """Cumulative (forward_m, left_m) in body frame after each step."""
    fwd = 0.0
    lat = 0.0
    scale = float(units_to_meter)
    out: List[tuple[float, float]] = []
    for step in steps:
        fwd += float(step.forward_m) * scale
        lat += float(step.left_m) * scale
        out.append((fwd, lat))
    return out


def world_delta_to_body(dx_m: float, dy_m: float, yaw_rad: float) -> tuple[float, float]:
    """TOP-frame (dx, dy) -> body (forward, left) at given heading."""
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    fwd = c * dy_m + s * dx_m
    lat = -s * dy_m + c * dx_m
    return fwd, lat


def load_flight_path(
    path: str | Path,
    *,
    default_square_side_m: float = 0.5,
    units_to_meter: float = 0.01,
    default_yaw_rad: float = 0.0,
) -> LoadedFlightPath:
    """Load route from JSON (absolute points or legacy relative steps)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Flight path file does not exist: {p}")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        preset = (data.get("type") or "").strip().lower()
        if preset == "square_clockwise":
            side_m = float(data.get("side_m", default_square_side_m))
            return LoadedFlightPath(
                absolute=False,
                steps=square_clockwise_steps(side_m),
                coordinate_unit="m",
            )

        unit = str(data.get("coordinate_unit", data.get("unit", "cm")))
        scale = _scale_for_unit(unit, units_to_meter)
        raw_points = data.get("points", data.get("waypoints"))
        if raw_points is not None:
            if not isinstance(raw_points, list):
                raise ValueError("'points' must be a list of {X,Y,Z} objects.")
            waypoints = [
                _point_from_mapping(item, scale, default_yaw_rad) for item in raw_points
            ]
            return LoadedFlightPath(absolute=True, waypoints=waypoints, coordinate_unit=unit)

        raw_steps = data.get("steps", [])
        if not isinstance(raw_steps, list):
            raise ValueError("'steps' must be a list of relative move objects.")
        return LoadedFlightPath(
            absolute=False,
            steps=[FlightPathStep.from_mapping(item) for item in raw_steps],
            coordinate_unit=unit,
        )

    if isinstance(data, list):
        if not data:
            return LoadedFlightPath()
        first = data[0]
        if isinstance(first, dict) and any(k in first for k in ("X", "x", "x_m")):
            scale = float(units_to_meter)
            waypoints = [_point_from_mapping(item, scale, default_yaw_rad) for item in data]
            return LoadedFlightPath(absolute=True, waypoints=waypoints)
        return LoadedFlightPath(
            absolute=False,
            steps=[FlightPathStep.from_mapping(item) for item in data],
        )

    raise ValueError("Flight path must be {'points': [...]} or {'steps': [...]}.")


def steps_to_waypoints(
    start: TrajectoryPoint,
    steps: Sequence[FlightPathStep],
    units_to_meter: float = 1.0,
) -> List[TrajectoryPoint]:
    """Integrate RELATIVE steps (body frame) into ABSOLUTE targets (TOP frame)."""
    waypoints: List[TrajectoryPoint] = []
    x, y, z, yaw = start.x_m, start.y_m, start.z_m, start.yaw_rad
    for step in steps:
        yaw = math.atan2(
            math.sin(yaw + math.radians(step.turn_deg)),
            math.cos(yaw + math.radians(step.turn_deg)),
        )
        fwd = step.forward_m * units_to_meter
        lat = step.left_m * units_to_meter
        c, s = math.cos(yaw), math.sin(yaw)
        dx = c * lat - s * fwd
        dy = s * lat + c * fwd
        x += dx
        y += dy
        z += step.up_m * units_to_meter
        waypoints.append(TrajectoryPoint(x_m=x, y_m=y, z_m=z, yaw_rad=yaw))
    return waypoints


class FutureTrajectoryPlanner:
    """Waypoint queue + current target."""

    def __init__(self, initial_target: TrajectoryPoint) -> None:
        self._queue: "Queue[TrajectoryPoint]" = Queue()
        self._current = initial_target

    def set_static_target(self, target: TrajectoryPoint) -> None:
        self._current = target
        self.clear()

    def enqueue_waypoints(self, trajectory: Iterable[TrajectoryPoint]) -> None:
        for point in trajectory:
            self._queue.put(point)

    def load_absolute_waypoints(self, waypoints: Sequence[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """Enqueue fixed lab-frame targets (no offset from current pose)."""
        wps = list(waypoints)
        self.clear()
        self.enqueue_waypoints(wps)
        if wps:
            self._current = wps[0]
        return wps

    def load_path(
        self,
        start: TrajectoryPoint,
        steps: Sequence[FlightPathStep],
        units_to_meter: float = 1.0,
    ) -> List[TrajectoryPoint]:
        """Convert relative steps to waypoints and enqueue them."""
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
        loaded = load_flight_path(path, units_to_meter=units_to_meter)
        if loaded.absolute:
            return self.load_absolute_waypoints(loaded.waypoints)
        return self.load_path(start, loaded.steps, units_to_meter)

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
        tgt = self.peek_target()
        dist = math.dist((x_m, y_m, z_m), (tgt.x_m, tgt.y_m, tgt.z_m))
        dyaw = abs(
            math.atan2(math.sin(yaw_rad - tgt.yaw_rad), math.cos(yaw_rad - tgt.yaw_rad))
        )
        return dist <= radius_m and math.degrees(dyaw) <= yaw_tol_deg

    def reached_side_pnp(
        self,
        side_x_m: float,
        side_z_m: float,
        target_x_m: float,
        target_z_m: float,
        radius_m: float = 0.15,
    ) -> bool:
        return math.dist((side_x_m, side_z_m), (target_x_m, target_z_m)) <= radius_m

    def advance_if_reached(self) -> None:
        if not self._queue.empty():
            self._current = self._queue.get()

    def clear(self) -> None:
        while not self._queue.empty():
            self._queue.get()
