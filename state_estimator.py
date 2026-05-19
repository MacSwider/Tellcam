"""Fuzja TOP (pełna poza) i SIDE (korekta yaw/z) dla markerów ArUco."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aruco_detector import SideCorrectionObservation, TopPoseObservation


@dataclass
class DroneState:
    valid: bool
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0
    yaw_rad: float = 0.0
    pitch_rad: float = 0.0
    roll_rad: float = 0.0
    top_ok: bool = False
    side_ok: bool = False


class StateEstimator:
    def __init__(self, side_yaw_alpha: float = 0.25, side_z_alpha: float = 0.4) -> None:
        self._yaw_alpha = float(np.clip(side_yaw_alpha, 0.0, 1.0))
        self._z_alpha = float(np.clip(side_z_alpha, 0.0, 1.0))

    def update(self, top: TopPoseObservation, side: SideCorrectionObservation) -> DroneState:
        top_ok = bool(top.ok)
        side_ok = bool(side.ok)
        if not top_ok:
            return DroneState(valid=False, top_ok=top_ok, side_ok=side_ok)

        yaw = self._unwrap_yaw(top.yaw_rad)
        z = top.z_m
        if side_ok:
            yaw = self._blend_angles(yaw, side.yaw_correction_rad, self._yaw_alpha)
            z = (1.0 - self._z_alpha) * z + self._z_alpha * side.z_from_marker_m

        return DroneState(
            valid=True,
            x_m=top.x_m,
            y_m=top.y_m,
            z_m=z,
            yaw_rad=self._unwrap_yaw(yaw),
            pitch_rad=top.pitch_rad,
            roll_rad=top.roll_rad,
            top_ok=top_ok,
            side_ok=side_ok,
        )

    @staticmethod
    def _unwrap_yaw(yaw: float) -> float:
        return float(np.arctan2(np.sin(yaw), np.cos(yaw)))

    @staticmethod
    def _blend_angles(a: float, b: float, alpha: float) -> float:
        da = np.arctan2(np.sin(b - a), np.cos(b - a))
        return float(a + alpha * da)
