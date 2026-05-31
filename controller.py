"""PID + stabilizacja pozycji TOP/SIDE -> komendy RC dla Tello."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import ControllerConfig, PIDGains, SafetyConfig, StabilizationConfig, TargetConfig


@dataclass
class RCCommand:
    left_right: int
    forward_back: int
    up_down: int
    yaw: int

    @property
    def roll(self) -> int:
        return self.left_right

    @property
    def pitch(self) -> int:
        return self.forward_back

    @property
    def throttle(self) -> int:
        return self.up_down

    @staticmethod
    def zero() -> "RCCommand":
        return RCCommand(0, 0, 0, 0)


@dataclass
class AxisDebug:
    error: float = 0.0
    output: float = 0.0


@dataclass
class StabilizationDebug:
    lateral: AxisDebug = field(default_factory=AxisDebug)
    vertical: AxisDebug = field(default_factory=AxisDebug)
    distance: AxisDebug = field(default_factory=AxisDebug)
    yaw: AxisDebug = field(default_factory=AxisDebug)
    target: TargetConfig = field(default_factory=TargetConfig)


class PIDController:
    def __init__(self, gains: PIDGains) -> None:
        self.gains = gains
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_output = 0.0

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_output = 0.0

    def step(self, error: float, dt: float, *, freeze_integrator: bool = False) -> float:
        if dt <= 0.0:
            dt = 1e-3
        if abs(error) < float(self.gains.deadzone):
            error = 0.0

        if not freeze_integrator:
            self._integral += self.gains.ki * error * dt
            limit = float(self.gains.integral_limit)
            self._integral = float(np.clip(self._integral, -limit, limit))

        p_term = self.gains.kp * error
        d_term = self.gains.kd * (error - self._prev_error) / dt
        raw = p_term + self._integral + d_term
        output = float(np.clip(raw, -self.gains.output_limit, self.gains.output_limit))

        if self.gains.slew_rate > 0.0:
            max_delta = float(self.gains.slew_rate) * dt
            low = self._prev_output - max_delta
            high = self._prev_output + max_delta
            output = float(np.clip(output, low, high))

        self._prev_error = error
        self._prev_output = output
        return output


class StabilizationController:
    """
    `control_frame="top"`:
    - x_m -> left/right nad podłogą
    - y_m -> forward/back nad podłogą
    - z_m -> wysokość

    `control_frame="side"`:
    - x_m -> left/right
    - y_m -> wysokość
    - z_m -> forward/back
    """

    def __init__(
        self,
        cfg: ControllerConfig,
        target: TargetConfig,
        stabilization_cfg: StabilizationConfig,
        safety_cfg: SafetyConfig,
    ) -> None:
        self._cfg = cfg
        self._target = target
        self._stabilization_cfg = stabilization_cfg
        self._safety_cfg = safety_cfg
        self.pid_x = PIDController(cfg.pid_x)
        self.pid_y = PIDController(cfg.pid_y)
        self.pid_z = PIDController(cfg.pid_z)
        self.pid_yaw = PIDController(cfg.pid_yaw)
        self.last_debug = StabilizationDebug(target=target)

    def set_target(self, target: TargetConfig) -> None:
        self._target = target
        self.last_debug.target = target

    def reset(self) -> None:
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()
        self.pid_yaw.reset()

    def compute(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        yaw_rad: float,
        dt: float,
        *,
        freeze_integrators: bool = False,
    ) -> RCCommand:
        ex = self._target.x_m - x_m
        ey = self._target.y_m - y_m
        ez = self._target.z_m - z_m
        eyaw = self._angle_diff(self._target.yaw_rad, yaw_rad)

        frame = self._stabilization_cfg.control_frame.strip().lower()
        if frame == "top":
            left_right = self._axis_output("x", self.pid_x, ex, dt, freeze_integrators)
            forward_back = self._axis_output("y", self.pid_y, ey, dt, freeze_integrators)
            up_down = self._axis_output("z", self.pid_z, ez, dt, freeze_integrators)
        else:
            left_right = self._axis_output("x", self.pid_x, ex, dt, freeze_integrators)
            up_down = self._axis_output("y", self.pid_y, ey, dt, freeze_integrators)
            forward_back = self._axis_output("z", self.pid_z, ez, dt, freeze_integrators)

        yaw_cmd = 0.0
        if self._cfg.use_yaw and self._stabilization_cfg.use_yaw:
            yaw_cmd = self.pid_yaw.step(eyaw, dt, freeze_integrator=freeze_integrators)
        else:
            self.pid_yaw.reset()

        self.last_debug = StabilizationDebug(
            lateral=AxisDebug(error=ex, output=left_right),
            vertical=AxisDebug(error=ey, output=up_down),
            distance=AxisDebug(error=ez, output=forward_back),
            yaw=AxisDebug(error=eyaw, output=yaw_cmd),
            target=self._target,
        )
        return RCCommand(
            left_right=self._quantize(left_right),
            forward_back=self._quantize(forward_back),
            up_down=self._quantize(up_down),
            yaw=self._quantize(yaw_cmd),
        )

    def _axis_output(
        self,
        axis_name: str,
        pid: PIDController,
        error: float,
        dt: float,
        freeze_integrators: bool,
    ) -> float:
        if axis_name not in self._stabilization_cfg.enabled_axes:
            pid.reset()
            return 0.0
        return pid.step(error, dt, freeze_integrator=freeze_integrators)

    def _quantize(self, value: float) -> int:
        limit = min(int(self._cfg.max_rc_abs), int(self._safety_cfg.max_rc_abs), 100)
        return int(np.clip(round(value), -limit, limit))

    @staticmethod
    def _angle_diff(target: float, current: float) -> float:
        return float(np.arctan2(np.sin(target - current), np.cos(target - current)))


class DroneController(StabilizationController):
    """Alias zgodny z dotychczasowym kodem."""
