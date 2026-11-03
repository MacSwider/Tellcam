"""Regulatory PID → komendy RC dla Tello (roll, pitch, throttle, yaw)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import ControllerConfig, PIDGains


class PID:
    def __init__(self, gains: PIDGains) -> None:
        self.g = gains
        self._i = 0.0
        self._prev_e: float | None = None

    def reset(self) -> None:
        self._i = 0.0
        self._prev_e = None

    def step(self, error: float, dt: float, rate: float | None = None) -> float:
        """rate: zmierzona prędkość w kierunku błędu (z filtra); wtedy D = -kd * rate."""
        if dt <= 0:
            dt = 1e-3
        p = self.g.kp * error
        self._i += self.g.ki * error * dt
        lim = self.g.integral_limit
        self._i = float(np.clip(self._i, -lim, lim))
        if rate is not None:
            d = -self.g.kd * rate
        else:
            d = 0.0 if self._prev_e is None else self.g.kd * (error - self._prev_e) / dt
        self._prev_e = error
        u = p + self._i + d
        return float(np.clip(u, -self.g.output_limit, self.g.output_limit))


@dataclass
class RCCommand:
    roll: int
    pitch: int
    throttle: int
    yaw: int


ZERO_RC = RCCommand(0, 0, 0, 0)


class DroneController:
    """
    XY i yaw w układzie kamery TOP (x w prawo, y w dół obrazu, yaw rośnie zgodnie z ruchem
    wskazówek zegara patrząc z góry), wysokość = wysokość nad podłogą.
    RC Tello: roll + = w prawo, pitch + = do przodu, throttle + = w górę, yaw + = w prawo.
    """

    def __init__(self, cfg: ControllerConfig) -> None:
        self._cfg = cfg
        self.pid_x = PID(cfg.pid_x)
        self.pid_y = PID(cfg.pid_y)
        self.pid_z = PID(cfg.pid_z)
        self.pid_yaw = PID(cfg.pid_yaw)

    def reset(self) -> None:
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()
        self.pid_yaw.reset()

    def _deadzone(self, e: float, dz: float) -> float:
        return 0.0 if abs(e) < dz else e

    def altitude_only(self, target_alt_m: float, alt_m: float, dt: float) -> RCCommand:
        ez = self._deadzone(target_alt_m - alt_m, self._cfg.deadzone_m)
        throttle = self._cfg.sign_throttle * self.pid_z.step(ez, dt)
        return RCCommand(0, 0, _q(throttle), 0)

    def compute(
        self,
        target_xyz: tuple[float, float, float],
        target_yaw_rad: float | None,
        x_m: float,
        y_m: float,
        alt_m: float,
        yaw_rad: float,
        dt: float,
        vel_xy: tuple[float, float] | None = None,
        basis: tuple[tuple[float, float], tuple[float, float]] | None = None,
    ) -> RCCommand:
        """basis: zmierzone kierunki (przód, prawo) drona w układzie kamery — ma pierwszeństwo przed yaw."""
        cfg = self._cfg
        ex = target_xyz[0] - x_m
        ey = target_xyz[1] - y_m
        ez = self._deadzone(target_xyz[2] - alt_m, cfg.deadzone_m)

        if basis is not None:
            (fx, fy), (rx, ry) = basis

            def to_body(ax: float, ay: float) -> tuple[float, float]:
                return ax * fx + ay * fy, ax * rx + ay * ry
        elif cfg.body_frame:
            heading = yaw_rad + cfg.heading_offset_rad
            c, s = np.cos(heading), np.sin(heading)

            def to_body(ax: float, ay: float) -> tuple[float, float]:
                return ax * c + ay * s, -ax * s + ay * c
        else:

            def to_body(ax: float, ay: float) -> tuple[float, float]:
                return -ay, ax

        e_fwd, e_right = to_body(ex, ey)
        v_fwd, v_right = to_body(*vel_xy) if vel_xy is not None else (None, None)

        if np.hypot(e_fwd, e_right) < cfg.deadzone_m:
            e_fwd = e_right = 0.0

        # Zmierzona baza już zawiera znaki osi — sign_roll/sign_pitch tylko dla trybu z yaw.
        sr, sp = (1, 1) if basis is not None else (cfg.sign_roll, cfg.sign_pitch)
        roll = sr * self.pid_x.step(e_right, dt, v_right)
        pitch = sp * self.pid_y.step(e_fwd, dt, v_fwd)
        throttle = cfg.sign_throttle * self.pid_z.step(ez, dt)
        yaw = 0.0
        if target_yaw_rad is not None:
            eyaw = self._deadzone(_angle_diff(target_yaw_rad, yaw_rad), cfg.deadzone_yaw_rad)
            yaw = cfg.sign_yaw * self.pid_yaw.step(eyaw, dt)

        return RCCommand(roll=_q(roll), pitch=_q(pitch), throttle=_q(throttle), yaw=_q(yaw))


def _q(v: float) -> int:
    return int(np.clip(round(v), -100, 100))


def _angle_diff(target: float, current: float) -> float:
    return float(np.arctan2(np.sin(target - current), np.cos(target - current)))
