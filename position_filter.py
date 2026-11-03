"""
Filtr Kalmana (model stałej prędkości) dla pozycji XY drona w układzie kamery TOP.

- pomiar pozycji: każda nowa detekcja z kamery TOP,
- opcjonalny pomiar prędkości: vgx/vgy z telemetrii Tello (obrócone o yaw do układu kamery),
- między pomiarami predykcja; po max_predict_s bez detekcji filtr uznaje tracking za utracony.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from config import FilterConfig


@dataclass
class FilteredState:
    valid: bool
    x_m: float = 0.0
    y_m: float = 0.0
    vx_mps: float = 0.0
    vy_mps: float = 0.0
    predicted_s: float = 0.0  # czas od ostatniej detekcji


class _Axis:
    """1D [pozycja, prędkość]."""

    def __init__(self) -> None:
        self.x = np.zeros(2)
        self.P = np.eye(2)

    def reset(self, pos: float) -> None:
        self.x = np.array([pos, 0.0])
        self.P = np.diag([0.01, 1.0])

    def predict(self, dt: float, accel_sigma: float) -> None:
        F = np.array([[1.0, dt], [0.0, 1.0]])
        G = np.array([0.5 * dt * dt, dt])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + np.outer(G, G) * accel_sigma**2

    def update(self, z: float, row: int, sigma: float) -> None:
        H = np.zeros(2)
        H[row] = 1.0
        S = H @ self.P @ H + sigma**2
        K = self.P @ H / S
        self.x = self.x + K * (z - H @ self.x)
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P


class PositionFilter:
    def __init__(self, cfg: FilterConfig) -> None:
        self._cfg = cfg
        self._ax = _Axis()
        self._ay = _Axis()
        self._initialized = False
        self._since_meas = float("inf")
        self._last_meas_ts: Optional[float] = None
        self._rejected = 0
        self._confirmed = 0

    def reset(self) -> None:
        self._initialized = False
        self._since_meas = float("inf")
        self._last_meas_ts = None
        self._rejected = 0
        self._confirmed = 0

    def step(
        self,
        dt: float,
        top_ok: bool,
        top_xy: Tuple[float, float],
        top_ts: float,
        tello_vel_cam: Optional[Tuple[float, float]] = None,
    ) -> FilteredState:
        cfg = self._cfg
        if self._initialized:
            self._ax.predict(dt, cfg.accel_sigma)
            self._ay.predict(dt, cfg.accel_sigma)
            self._since_meas += dt

        new_meas = top_ok and top_ts != self._last_meas_ts
        if new_meas:
            self._last_meas_ts = top_ts
            stale = not self._initialized or self._since_meas > cfg.max_predict_s
            jump = 0.0 if stale else float(np.hypot(top_xy[0] - self._ax.x[0], top_xy[1] - self._ay.x[0]))
            if stale:
                self._ax.reset(top_xy[0])
                self._ay.reset(top_xy[1])
                self._initialized = True
                self._since_meas = 0.0
                self._rejected = 0
                self._confirmed = 1
            elif jump > cfg.gate_m and self._rejected + 1 < cfg.gate_reset_count:
                self._rejected += 1
            else:
                if jump > cfg.gate_m:
                    self._ax.reset(top_xy[0])
                    self._ay.reset(top_xy[1])
                    self._confirmed = 1
                else:
                    self._ax.update(top_xy[0], 0, cfg.position_sigma_m)
                    self._ay.update(top_xy[1], 0, cfg.position_sigma_m)
                    self._confirmed += 1
                self._since_meas = 0.0
                self._rejected = 0

        if self._initialized and tello_vel_cam is not None:
            self._ax.update(tello_vel_cam[0], 1, cfg.tello_velocity_sigma_mps)
            self._ay.update(tello_vel_cam[1], 1, cfg.tello_velocity_sigma_mps)

        valid = (
            self._initialized
            and self._since_meas <= cfg.max_predict_s
            and self._confirmed >= cfg.confirm_count
        )
        return FilteredState(
            valid=valid,
            x_m=float(self._ax.x[0]),
            y_m=float(self._ay.x[0]),
            vx_mps=float(self._ax.x[1]),
            vy_mps=float(self._ay.x[1]),
            predicted_s=self._since_meas if self._initialized else 0.0,
        )


def tello_velocity_to_camera(
    vgx: float, vgy: float, yaw_rad: float, cfg: FilterConfig, heading_offset_rad: float
) -> Tuple[float, float]:
    """vgx = przód, vgy = prawo (po korekcie znaków z configu) → układ kamery TOP."""
    v_fwd = cfg.tello_vel_sign_x * vgx * cfg.tello_velocity_scale
    v_right = cfg.tello_vel_sign_y * vgy * cfg.tello_velocity_scale
    h = yaw_rad + heading_offset_rad
    c, s = np.cos(h), np.sin(h)
    return float(v_fwd * c - v_right * s), float(v_fwd * s + v_right * c)
