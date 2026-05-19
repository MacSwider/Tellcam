"""Regulatory PID → komendy RC dla Tello (roll, pitch, throttle, yaw)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from config import ControllerConfig, PIDGains, TargetConfig


class PID:
    def __init__(self, gains: PIDGains) -> None:
        self.g = gains
        self._i = 0.0
        self._prev_e = 0.0

    def reset(self) -> None:
        self._i = 0.0
        self._prev_e = 0.0

    def step(self, error: float, dt: float) -> float:
        if dt <= 0:
            dt = 1e-3
        p = self.g.kp * error
        self._i += self.g.ki * error * dt
        lim = self.g.integral_limit
        self._i = float(np.clip(self._i, -lim, lim))
        d = self.g.kd * (error - self._prev_e) / dt
        self._prev_e = error
        u = p + self._i + d
        return float(np.clip(u, -self.g.output_limit, self.g.output_limit))


@dataclass
class RCCommand:
    roll: int
    pitch: int
    throttle: int
    yaw: int


class DroneController:
    """
    Błędy: ex = target.x - x, itd.
    Znaki wyjść można dopasować do układu współrzędnych / montażu kamer.
    """

    def __init__(self, cfg: ControllerConfig, target: TargetConfig) -> None:
        self._cfg = cfg
        self._target = target
        self.pid_x = PID(cfg.pid_x)
        self.pid_y = PID(cfg.pid_y)
        self.pid_z = PID(cfg.pid_z)
        self.pid_yaw = PID(cfg.pid_yaw)
        # Znaki: strojenie pod rzeczywisty układ Tello + kamery
        self.sign_roll = 1
        self.sign_pitch = 1
        self.sign_throttle = 1
        self.sign_yaw_cmd = 1

    def set_target(self, target: TargetConfig) -> None:
        self._target = target

    def reset(self) -> None:
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()
        self.pid_yaw.reset()

    def compute(self, x_m: float, y_m: float, z_m: float, yaw_rad: float, dt: float) -> RCCommand:
        ex = self._target.x_m - x_m
        ey = self._target.y_m - y_m
        ez = self._target.z_m - z_m
        eyaw = self._angle_diff(self._target.yaw_rad, yaw_rad)

        if abs(ex) < self._cfg.deadzone_m:
            ex = 0.0
        if abs(ey) < self._cfg.deadzone_m:
            ey = 0.0
        if abs(ez) < self._cfg.deadzone_m:
            ez = 0.0
        if abs(eyaw) < self._cfg.deadzone_yaw_rad:
            eyaw = 0.0

        roll = self.sign_roll * self.pid_x.step(ex, dt)
        pitch = self.sign_pitch * self.pid_y.step(ey, dt)
        throttle = self.sign_throttle * self.pid_z.step(ez, dt)
        yaw = self.sign_yaw_cmd * self.pid_yaw.step(eyaw, dt)

        def q(v: float) -> int:
            return int(np.clip(round(v), -100, 100))

        return RCCommand(roll=q(roll), pitch=q(pitch), throttle=q(throttle), yaw=q(yaw))

    @staticmethod
    def _angle_diff(target: float, current: float) -> float:
        return float(np.arctan2(np.sin(target - current), np.cos(target - current)))
