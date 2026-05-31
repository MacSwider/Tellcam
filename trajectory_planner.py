"""Placeholder architektury pod przyszłą nawigację po waypointach."""
from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from typing import Iterable


@dataclass(frozen=True)
class TrajectoryPoint:
    x_m: float
    y_m: float
    z_m: float
    yaw_rad: float = 0.0


class FutureTrajectoryPlanner:
    """
    Na razie dostarcza pojedynczy statyczny cel.
    W przyszłości może obsługiwać kolejkę waypointów i planowanie trajektorii.
    """

    def __init__(self, initial_target: TrajectoryPoint) -> None:
        self._queue: Queue[TrajectoryPoint] = Queue()
        self._current = initial_target

    def set_static_target(self, target: TrajectoryPoint) -> None:
        self._current = target
        self.clear()

    def enqueue_waypoints(self, trajectory: Iterable[TrajectoryPoint]) -> None:
        for point in trajectory:
            self._queue.put(point)

    def peek_target(self) -> TrajectoryPoint:
        if not self._queue.empty():
            self._current = self._queue.queue[0]
        return self._current

    def advance_if_reached(self) -> None:
        if not self._queue.empty():
            self._current = self._queue.get()

    def clear(self) -> None:
        while not self._queue.empty():
            self._queue.get()
