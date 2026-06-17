"""PID + TOP/SIDE position stabilization -> RC commands for Tello."""

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
    # Labels follow drone commands (independent of control_frame).
    left_right: AxisDebug = field(default_factory=AxisDebug)
    forward_back: AxisDebug = field(default_factory=AxisDebug)
    up_down: AxisDebug = field(default_factory=AxisDebug)
    yaw: AxisDebug = field(default_factory=AxisDebug)
    side_u: AxisDebug = field(default_factory=AxisDebug)
    side_depth: AxisDebug = field(default_factory=AxisDebug)
    top_u: AxisDebug = field(default_factory=AxisDebug)
    top_v: AxisDebug = field(default_factory=AxisDebug)
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

    def step(
        self,
        error: float,
        dt: float,
        *,
        freeze_integrator: bool = False,
        measurement_rate: float | None = None,
    ) -> float:
        if dt <= 0.0:
            dt = 1e-3
        active = abs(error) >= float(self.gains.deadzone)
        if not active:
            error = 0.0
        if not freeze_integrator:
            self._integral += self.gains.ki * error * dt
            limit = float(self.gains.integral_limit)
            self._integral = float(np.clip(self._integral, -limit, limit))
        p_term = self.gains.kp * error
        if measurement_rate is not None:
            # Derivative from measured velocity (d(error)/dt = -v for fixed target) —
            # smoother and faster damping of drone acceleration (drift).
            d_term = -self.gains.kd * float(measurement_rate)
        else:
            d_term = self.gains.kd * (error - self._prev_error) / dt
        raw = p_term + self._integral + d_term
        output = float(np.clip(raw, -self.gains.output_limit, self.gains.output_limit))
        if self.gains.slew_rate > 0.0:
            max_delta = float(self.gains.slew_rate) * dt
            low = self._prev_output - max_delta
            high = self._prev_output + max_delta
            output = float(np.clip(output, low, high))
        # Minimum effective command — overcome Tello RC dead zone.
        min_cmd = float(self.gains.min_command)
        if active and min_cmd > 0.0 and 0.0 < abs(output) < min_cmd:
            output = float(np.copysign(min_cmd, output))
        self._prev_error = error
        self._prev_output = output
        return output


class StabilizationController:
    """
    `control_frame="top"`:
    - x_m -> left/right over floor
    - y_m -> forward/back over floor
    - z_m -> altitude
    `control_frame="side"`:
    - x_m -> left/right
    - y_m -> altitude
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
        self.pid_side_u = PIDController(cfg.pid_side_u)
        self.pid_side_depth = PIDController(cfg.pid_side_depth)
        self.pid_side_x = PIDController(cfg.pid_side_x_m)
        self.pid_top_u = PIDController(cfg.pid_top_u)
        self.pid_top_v = PIDController(cfg.pid_top_v)
        self.last_debug = StabilizationDebug(target=target)

    def set_target(self, target: TargetConfig) -> None:
        self._target = target
        self.last_debug.target = target

    def reset(self) -> None:
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()
        self.pid_yaw.reset()
        self.pid_side_u.reset()
        self.pid_side_depth.reset()
        self.pid_side_x.reset()
        self.pid_top_u.reset()
        self.pid_top_v.reset()

    def compute(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        yaw_rad: float,
        dt: float,
        *,
        freeze_integrators: bool = False,
        active_axes: tuple[str, ...] | None = None,
        freeze_axes: tuple[str, ...] = (),
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
        heading_rad: float = 0.0,
        heading_valid: bool = False,
        side_u_px: float = 0.0,
        side_v_px: float = 0.0,
        side_area_px: float = 0.0,
        side_z_m: float = 0.0,
        side_vu_px_s: float = 0.0,
        side_vv_px_s: float = 0.0,
        side_vz_m_s: float = 0.0,
        side_x_m: float = 0.0,
        side_vx_m_s: float = 0.0,
        top_u_px: float = 0.0,
        top_v_px: float = 0.0,
        top_vu_px_s: float = 0.0,
        top_vv_px_s: float = 0.0,
        apply_side_hold: bool = True,
        apply_side_pnp_travel: bool = False,
    ) -> RCCommand:
        ex = self._target.x_m - x_m
        ey = self._target.y_m - y_m
        ez = self._target.z_m - z_m
        eyaw = self._angle_diff(self._target.yaw_rad, yaw_rad)

        def _freeze(axis: str) -> bool:
            return freeze_integrators or (axis in freeze_axes)

        roll_sign = float(self._cfg.roll_sign)
        pitch_sign = float(self._cfg.pitch_sign)
        throttle_sign = float(self._cfg.throttle_sign)
        frame = self._stabilization_cfg.control_frame.strip().lower()
        if frame == "top":
            # World: x=left, y=forward. Yaw=0 => drone faces +y (forward in lab).
            if self._stabilization_cfg.body_frame_control and heading_valid:
                c, s = float(np.cos(heading_rad)), float(np.sin(heading_rad))
                fwd_e = -s * ex + c * ey
                lat_e = c * ex + s * ey
                fwd_v = -s * vx + c * vy
                lat_v = c * vx + s * vy
            else:
                fwd_e, lat_e = ey, ex
                fwd_v, lat_v = vy, vx
            # Align yaw first: at large heading error limit horizontal motion.
            trans_scale = 1.0
            if heading_valid:
                thr = float(np.radians(self._stabilization_cfg.yaw_align_deg_for_translation))
                if abs(eyaw) > thr:
                    trans_scale = float(max(0.2, 1.0 - (abs(eyaw) - thr) / (np.pi / 2.0)))
            forward_back = (
                self._axis_output("y", self.pid_y, fwd_e, dt, _freeze("y"), active_axes, fwd_v)
                * trans_scale
                * pitch_sign
            )
            left_right = (
                self._axis_output("x", self.pid_x, lat_e, dt, _freeze("x"), active_axes, lat_v)
                * trans_scale
                * roll_sign
            )
            up_down = (
                self._axis_output("z", self.pid_z, ez, dt, _freeze("z"), active_axes, vz)
                * throttle_sign
            )
            lr_e, fb_e, ud_e = lat_e, fwd_e, ez
            if apply_side_hold and not apply_side_pnp_travel:
                top_gain = float(getattr(self._stabilization_cfg, "top_hold_gain", 1.0))
                top_xy_active = active_axes is None or (
                    "x" in active_axes and "y" in active_axes
                )
                if top_gain != 1.0 and top_xy_active:
                    left_right *= top_gain
                    forward_back *= top_gain
        else:
            left_right = (
                self._axis_output("x", self.pid_x, ex, dt, _freeze("x"), active_axes, vx)
                * roll_sign
            )
            up_down = (
                self._axis_output("y", self.pid_y, ey, dt, _freeze("y"), active_axes, vy)
                * throttle_sign
            )
            forward_back = (
                self._axis_output("z", self.pid_z, ez, dt, _freeze("z"), active_axes, vz)
                * pitch_sign
            )
            lr_e, fb_e, ud_e = ex, ez, ey
        yaw_active = active_axes is None or "yaw" in active_axes
        yaw_cmd = 0.0
        if self._cfg.use_yaw and self._stabilization_cfg.use_yaw and yaw_active:
            yaw_cmd = float(self._cfg.yaw_sign) * self.pid_yaw.step(
                eyaw, dt, freeze_integrator=_freeze("yaw")
            )
        else:
            self.pid_yaw.reset()
        depth_rc_axis = (
            getattr(self._stabilization_cfg, "side_depth_rc_axis", "pitch") or "pitch"
        ).strip().lower()

        def _add_depth_rc(lr: float, fb: float, out_d: float, blend: float) -> tuple[float, float]:
            if depth_rc_axis == "roll":
                return lr + blend * out_d, fb
            return lr, fb + blend * out_d

        if apply_side_pnp_travel and self._target.side_x_m is not None and self._target.side_z_m is not None:
            travel_blend = float(getattr(self._stabilization_cfg, "side_travel_blend", 0.85))
            top_ff = 1.0 - travel_blend
            ex_s = float(self._target.side_x_m) - float(side_x_m)
            ez_s = float(self._target.side_z_m) - float(side_z_m)
            side_x_active = active_axes is None or "side_x" in active_axes
            side_d_active = active_axes is None or "side_depth" in active_axes
            lat_out = 0.0
            depth_out = 0.0
            if side_x_active:
                lat_out = float(self._cfg.roll_sign) * self.pid_side_x.step(
                    ex_s, dt, freeze_integrator=_freeze("side_x"), measurement_rate=side_vx_m_s
                )
            else:
                self.pid_side_x.reset()
            if side_d_active:
                depth_out = float(self._cfg.side_depth_sign) * self.pid_side_depth.step(
                    ez_s, dt, freeze_integrator=_freeze("side_depth"), measurement_rate=side_vz_m_s
                )
            else:
                self.pid_side_depth.reset()
            left_right = travel_blend * lat_out + top_ff * left_right
            if depth_rc_axis == "roll":
                left_right = travel_blend * depth_out + top_ff * left_right
                forward_back = top_ff * forward_back
            else:
                forward_back = travel_blend * depth_out + top_ff * forward_back
            lr_e, fb_e = ex_s, ez_s
        side_u_dbg = AxisDebug()
        side_depth_dbg = AxisDebug()
        pnp_blend = float(getattr(self._stabilization_cfg, "side_pnp_hold_blend", 0.0))
        top_horiz_lost = (
            active_axes is not None and "x" not in active_axes and "y" not in active_axes
        )
        top_primary = bool(getattr(self._stabilization_cfg, "top_primary_horiz", True))
        if (
            apply_side_hold
            and not apply_side_pnp_travel
            and getattr(self._stabilization_cfg, "side_pnp_hold", False)
            and pnp_blend > 0.0
            and self._target.side_x_m is not None
            and (active_axes is None or "side_x" in active_axes)
        ):
            if top_primary and not top_horiz_lost:
                pnp_blend = 0.0
            elif top_horiz_lost:
                pnp_blend = min(1.0, pnp_blend * 1.15)
            ex_s = float(self._target.side_x_m) - float(side_x_m)
            out_x = float(self._cfg.roll_sign) * self.pid_side_x.step(
                ex_s, dt, freeze_integrator=_freeze("side_x"), measurement_rate=side_vx_m_s
            )
            left_right += pnp_blend * out_x
        elif apply_side_hold and not apply_side_pnp_travel:
            self.pid_side_x.reset()
        top_u_dbg = AxisDebug()
        top_v_dbg = AxisDebug()
        top_img_blend = 0.0
        if (
            apply_side_hold
            and not apply_side_pnp_travel
            and getattr(self._stabilization_cfg, "top_image_hold", False)
        ):
            top_img_blend = float(
                np.clip(float(getattr(self._stabilization_cfg, "top_image_hold_blend", 0.0)), 0.0, 1.0)
            )
        if (
            top_img_blend > 0.0
            and self._target.top_u_px is not None
            and self._target.top_v_px is not None
        ):
            u_axis = (getattr(self._stabilization_cfg, "top_u_rc_axis", "roll") or "roll").strip().lower()
            v_axis = (getattr(self._stabilization_cfg, "top_v_rc_axis", "pitch") or "pitch").strip().lower()
            img_lr = 0.0
            img_fb = 0.0
            if active_axes is None or "top_u" in active_axes:
                eu = float(self._target.top_u_px) - float(top_u_px)
                out_u = float(self._cfg.top_u_sign) * self.pid_top_u.step(
                    eu, dt, freeze_integrator=_freeze("top_u"), measurement_rate=top_vu_px_s
                )
                if u_axis == "roll":
                    img_lr += out_u
                else:
                    img_fb += out_u
                top_u_dbg = AxisDebug(error=eu, output=out_u)
            else:
                self.pid_top_u.reset()
            if active_axes is None or "top_v" in active_axes:
                ev = float(self._target.top_v_px) - float(top_v_px)
                out_v = float(self._cfg.top_v_sign) * self.pid_top_v.step(
                    ev, dt, freeze_integrator=_freeze("top_v"), measurement_rate=top_vv_px_s
                )
                if v_axis == "roll":
                    img_lr += out_v
                else:
                    img_fb += out_v
                top_v_dbg = AxisDebug(error=ev, output=out_v)
            else:
                self.pid_top_v.reset()
            pnp_lr, pnp_fb = left_right, forward_back
            left_right = (1.0 - top_img_blend) * pnp_lr + top_img_blend * img_lr
            forward_back = (1.0 - top_img_blend) * pnp_fb + top_img_blend * img_fb
        elif apply_side_hold and not apply_side_pnp_travel:
            self.pid_top_u.reset()
            self.pid_top_v.reset()
        side_img_blend = 0.0
        side_area_blend = 0.0
        if (
            apply_side_hold
            and not apply_side_pnp_travel
            and getattr(self._stabilization_cfg, "side_image_hold", False)
        ):
            side_img_blend = float(
                np.clip(float(getattr(self._stabilization_cfg, "side_image_hold_blend", 0.0)), 0.0, 1.0)
            )
            side_area_blend = float(
                np.clip(float(getattr(self._stabilization_cfg, "side_image_area_blend", 0.0)), 0.0, 1.0)
            )
        if side_img_blend > 0.0 and self._target.side_u_px is not None:
            u_axis = (getattr(self._stabilization_cfg, "side_u_rc_axis", "roll") or "roll").strip().lower()
            area_axis = (
                getattr(self._stabilization_cfg, "side_area_rc_axis", "pitch") or "pitch"
            ).strip().lower()
            img_lr = 0.0
            img_fb = 0.0
            if active_axes is None or "side_u" in active_axes:
                eu = float(self._target.side_u_px) - float(side_u_px)
                out_u = float(self._cfg.side_u_sign) * self.pid_side_u.step(
                    eu, dt, freeze_integrator=_freeze("side_u"), measurement_rate=side_vu_px_s
                )
                if u_axis == "roll":
                    img_lr += out_u
                else:
                    img_fb += out_u
                side_u_dbg = AxisDebug(error=eu, output=out_u)
            else:
                self.pid_side_u.reset()
            if (
                side_area_blend > 0.0
                and getattr(self._stabilization_cfg, "side_image_area_hold", False)
                and self._target.side_area_px is not None
                and (active_axes is None or "side_depth" in active_axes)
            ):
                ed = (float(self._target.side_area_px) - float(side_area_px)) / max(
                    float(self._target.side_area_px), 100.0
                )
                out_d = float(self._cfg.side_depth_sign) * self.pid_side_depth.step(
                    ed, dt, freeze_integrator=_freeze("side_depth"), measurement_rate=side_vz_m_s
                )
                if area_axis == "roll":
                    img_lr += side_area_blend * out_d
                else:
                    img_fb += side_area_blend * out_d
                side_depth_dbg = AxisDebug(error=ed, output=out_d)
            else:
                self.pid_side_depth.reset()
            base_lr, base_fb = left_right, forward_back
            left_right = (1.0 - side_img_blend) * base_lr + side_img_blend * img_lr
            forward_back = (1.0 - side_img_blend) * base_fb + side_img_blend * img_fb
        elif apply_side_hold and not apply_side_pnp_travel:
            self.pid_side_u.reset()
            if not getattr(self._stabilization_cfg, "side_image_area_hold", False):
                self.pid_side_depth.reset()
        self.last_debug = StabilizationDebug(
            left_right=AxisDebug(error=lr_e, output=left_right),
            forward_back=AxisDebug(error=fb_e, output=forward_back),
            up_down=AxisDebug(error=ud_e, output=up_down),
            yaw=AxisDebug(error=eyaw, output=yaw_cmd),
            side_u=side_u_dbg,
            side_depth=side_depth_dbg,
            top_u=top_u_dbg,
            top_v=top_v_dbg,
            target=self._target,
        )
        return RCCommand(
            left_right=self._quantize(left_right),
            forward_back=self._quantize(forward_back),
            up_down=self._quantize(up_down),
            yaw=self._quantize(yaw_cmd),
        )

    def compute_yaw_command(
        self,
        yaw_rad: float,
        target_yaw_rad: float,
        dt: float,
        *,
        freeze_integrator: bool = False,
    ) -> int:
        """Yaw correction only (e.g. during climb) — no roll/pitch/throttle.
        Stabilizes heading before full position loop runs so the drone does not
        spin up (anti-spin). Returns quantized RC yaw command.
        """
        if not (self._cfg.use_yaw and self._stabilization_cfg.use_yaw):
            self.pid_yaw.reset()
            return 0
        eyaw = self._angle_diff(target_yaw_rad, yaw_rad)
        yaw_cmd = float(self._cfg.yaw_sign) * self.pid_yaw.step(
            eyaw, dt, freeze_integrator=freeze_integrator
        )
        self.last_debug.yaw = AxisDebug(error=eyaw, output=yaw_cmd)
        return self._quantize(yaw_cmd)

    def _axis_output(
        self,
        axis_name: str,
        pid: PIDController,
        error: float,
        dt: float,
        freeze_integrator: bool,
        active_axes: tuple[str, ...] | None,
        measurement_rate: float = 0.0,
    ) -> float:
        if axis_name not in self._stabilization_cfg.enabled_axes:
            pid.reset()
            return 0.0
        # No fresh source for this axis -> hover on that axis (zero and reset PID).
        if active_axes is not None and axis_name not in active_axes:
            pid.reset()
            return 0.0
        return pid.step(
            error, dt, freeze_integrator=freeze_integrator, measurement_rate=measurement_rate
        )

    def _quantize(self, value: float) -> int:
        limit = min(int(self._cfg.max_rc_abs), int(self._safety_cfg.max_rc_abs), 100)
        return int(np.clip(round(value), -limit, limit))

    @staticmethod
    def _angle_diff(target: float, current: float) -> float:
        return float(np.arctan2(np.sin(target - current), np.cos(target - current)))


class DroneController(StabilizationController):
    """Alias compatible with existing code."""
