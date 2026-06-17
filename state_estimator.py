"""
TOP+SIDE pose fusion with alpha-beta filter (position + velocity) and prediction.
Optional complementary fusion with Tello body velocities between AprilTag updates.
Physical assumptions (two external USB cameras observing a tagged drone):
- TOP camera best measures horizontal position: x (left/right), y (forward/back), yaw.
- SIDE camera best measures drone ALTITUDE (vertical position = side.y_m).
  When face_to_face at takeoff, SIDE depth/area also reflects approach toward the camera (pitch).
Each axis (x, y from TOP; altitude from SIDE) has an alpha-beta filter that:
- estimates velocity from successive detections,
- predicts position between detections and in short gaps (up to max_predict_s),
- blends Tello odometry velocity between vision frames when enabled,
reducing control delay and damping drift earlier.
Source states: TRACKING -> HOLDING_LAST -> LOST -> TIMED_OUT (per camera).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from apriltag_detector import SideCorrectionObservation, TopPoseObservation
from config import OdometryFusionConfig, StabilizationConfig, TrackingConfig

TRACKING = "TRACKING"
HOLDING_LAST = "HOLDING_LAST"
LOST = "LOST"
TIMED_OUT = "TIMED_OUT"


@dataclass
class DroneState:
    valid: bool
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0
    yaw_rad: float = 0.0
    pitch_rad: float = 0.0
    roll_rad: float = 0.0
    vx_m_s: float = 0.0
    vy_m_s: float = 0.0
    vz_m_s: float = 0.0
    # Drone heading (for body-frame transform): TOP, fallback aligned SIDE.
    heading_rad: float = 0.0
    heading_valid: bool = False
    # Raw heading from side camera (tags 1–4 layout) — for diagnostics/preview.
    side_heading_rad: float = 0.0
    side_heading_valid: bool = False
    top_ok: bool = False
    side_ok: bool = False
    top_state: str = LOST
    side_state: str = LOST
    tracking_state: str = LOST
    active_axes: tuple[str, ...] = ()
    freeze_axes: tuple[str, ...] = ()
    pose_age_s: float = 0.0
    measurement_age_s: float = 0.0
    marker_id: int | None = None
    hint: str | None = None
    # SIDE camera measurements for image point/distance hold.
    side_u_px: float = 0.0
    side_v_px: float = 0.0
    side_area_px: float = 0.0
    side_z_m: float = 0.0
    side_x_m: float = 0.0
    side_vu_px_s: float = 0.0
    side_vv_px_s: float = 0.0
    side_vz_m_s: float = 0.0
    side_vx_m_s: float = 0.0
    side_marker_id: int | None = None
    top_u_px: float = 0.0
    top_v_px: float = 0.0
    top_vu_px_s: float = 0.0
    top_vv_px_s: float = 0.0
    odom_fusion_active: bool = False
    odom_vx_m_s: float = 0.0
    odom_vy_m_s: float = 0.0
    odom_vz_m_s: float = 0.0


@dataclass
class OdometrySample:
    """Body-frame velocity sample for complementary fusion."""

    vx_m_s: float = 0.0
    vy_m_s: float = 0.0
    vz_m_s: float = 0.0
    valid: bool = False
    flying: bool = False


class AlphaBetaAxis:
    """Alpha-beta filter: estimates position and velocity, predicts over time."""

    def __init__(self, alpha: float, beta: float, max_speed: float) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_speed = float(max_speed)
        self.x: float | None = None
        self.v: float = 0.0
        self.t: float = 0.0

    def reset(self) -> None:
        self.x = None
        self.v = 0.0
        self.t = 0.0

    def update(self, meas: float, t: float) -> None:
        if self.x is None:
            self.x = float(meas)
            self.v = 0.0
            self.t = t
            return
        dt = t - self.t
        if dt <= 1e-4:
            dt = 1e-3
        x_pred = self.x + self.v * dt
        residual = float(meas) - x_pred
        self.x = x_pred + self.alpha * residual
        self.v = self.v + (self.beta / dt) * residual
        self.v = float(np.clip(self.v, -self.max_speed, self.max_speed))
        self.t = t

    def predict(self, t: float, max_horizon: float) -> float:
        """Position predicted at time t (velocity extrapolation, horizon limited)."""
        if self.x is None:
            return 0.0
        horizon = min(max(0.0, t - self.t), max_horizon)
        return self.x + self.v * horizon


class PoseEstimator:
    def __init__(self, cfg: TrackingConfig, stab_cfg: StabilizationConfig | None = None) -> None:
        self._cfg = cfg
        self._stab = stab_cfg or StabilizationConfig()
        a, b, vmax = cfg.filter_alpha, cfg.filter_beta, cfg.max_speed_m_s
        a_xy = float(getattr(cfg, "filter_alpha_xy", a))
        b_xy = float(getattr(cfg, "filter_beta_xy", b))
        self._fx = AlphaBetaAxis(a_xy, b_xy, vmax)
        self._fy = AlphaBetaAxis(a_xy, b_xy, vmax)
        self._falt = AlphaBetaAxis(a, b, vmax)
        self._ftop_z = AlphaBetaAxis(a, b, vmax)
        self._fside_u = AlphaBetaAxis(a, b, vmax * 500.0)  # px/s
        self._fside_v = AlphaBetaAxis(a, b, vmax * 500.0)
        self._fside_area = AlphaBetaAxis(a, b, vmax * 1e6)
        self._fside_z = AlphaBetaAxis(a, b, vmax)
        self._fside_x = AlphaBetaAxis(a, b, vmax)
        self._top_image_hold = bool(getattr(self._stab, "top_image_hold", False))
        self._ftop_u = AlphaBetaAxis(a, b, vmax * 500.0)
        self._ftop_v = AlphaBetaAxis(a, b, vmax * 500.0)
        self._side_image_hold = bool(getattr(self._stab, "side_image_hold", False))
        self._top_ts = 0.0
        self._side_ts = 0.0
        self._top_meas_ts = 0.0
        self._side_meas_ts = 0.0
        self._yaw = 0.0
        self._pitch = 0.0
        self._roll = 0.0
        self._side_facing = 0.0
        self._side_heading = 0.0  # continuous heading from side tag layout
        self._has_side_heading = False
        self._side_to_top_offset = 0.0  # SIDE->TOP heading alignment
        self._has_offset = False
        self._top_n = 0
        self._yaw_source = (self._stab.yaw_source or "off").strip().lower()
        self._use_yaw = bool(self._stab.use_yaw)
        self._side_yaw_sign = float(getattr(self._stab, "side_yaw_sign", 1.0))
        self._side_yaw_offset = np.radians(float(getattr(self._stab, "side_yaw_offset_deg", 0.0)))
        # Outward azimuths of side tags [rad], keys as int.
        az = getattr(self._stab, "side_tag_azimuths_deg", {}) or {}
        self._side_azimuth = {int(k): np.radians(float(v)) for k, v in az.items()}
        self._has_top = False
        self._has_side = False
        self._odom_cfg: OdometryFusionConfig = cfg.odometry

    def _side_image_enabled(
        self, top_usable: bool, side_usable: bool, side_state: str
    ) -> bool:
        if not self._side_image_hold:
            return False
        side_live = side_usable and side_state in (TRACKING, HOLDING_LAST)
        if bool(getattr(self._stab, "side_primary_horiz", False)):
            return side_live
        if bool(getattr(self._stab, "top_primary_horiz", True)):
            return side_live and not top_usable
        return side_live

    def _top_image_enabled(
        self,
        top_usable: bool,
        side_usable: bool,
        top_state: str,
        side_state: str,
    ) -> bool:
        if not self._top_image_hold:
            return False
        top_live = top_usable and top_state in (TRACKING, HOLDING_LAST)
        if bool(getattr(self._stab, "side_primary_horiz", False)):
            side_live = (
                self._side_image_hold
                and side_usable
                and side_state in (TRACKING, HOLDING_LAST)
            )
            if side_live:
                return False
            return top_live
        return top_live

    def reset(self) -> None:
        for f in (
            self._fx,
            self._fy,
            self._falt,
            self._ftop_z,
            self._fside_u,
            self._fside_v,
            self._fside_area,
            self._fside_z,
            self._fside_x,
            self._ftop_u,
            self._ftop_v,
        ):
            f.reset()
        self._top_ts = self._side_ts = 0.0
        self._top_meas_ts = self._side_meas_ts = 0.0
        self._has_top = self._has_side = False

    def _state_for_age(self, age: float) -> str:
        if age <= self._cfg.hold_last_pose_s:
            return HOLDING_LAST
        if age <= self._cfg.lost_timeout_s:
            return LOST
        return TIMED_OUT

    def update(
        self,
        top: TopPoseObservation,
        side: SideCorrectionObservation,
        *,
        top_ts: float = 0.0,
        side_ts: float = 0.0,
        now: float = 0.0,
        odometry: OdometrySample | None = None,
    ) -> DroneState:
        predict = bool(self._cfg.predict_enabled)
        max_h = float(self._cfg.max_predict_s)
        # --- TOP: new measurement only when publish timestamp changes ---
        if top.ok and top_ts > self._top_ts:
            self._fx.update(float(top.x_m), now)
            self._fy.update(float(top.y_m), now)
            self._ftop_z.update(float(top.z_m), now)
            if self._top_image_hold:
                self._ftop_u.update(float(top.centroid_u_px), now)
                self._ftop_v.update(float(top.centroid_v_px), now)
            self._yaw = self._unwrap_yaw(float(top.yaw_rad))
            self._pitch = float(top.pitch_rad)
            self._roll = float(top.roll_rad)
            self._top_n = len(top.markers_seen)
            self._top_ts = top_ts
            self._top_meas_ts = now
            self._has_top = True
        top_state = (
            TRACKING
            if (top.ok and now - self._top_meas_ts <= 1e-6)
            else (self._state_for_age(now - self._top_meas_ts) if self._has_top else LOST)
        )
        # --- SIDE: altitude + heading + (optional) image point/distance ---
        if side.ok and side_ts > self._side_ts:
            self._falt.update(float(side.y_m), now)
            self._fside_x.update(float(side.x_m), now)
            self._fside_z.update(float(side.z_m), now)
            if self._side_image_hold:
                self._fside_u.update(float(side.centroid_u_px), now)
                self._fside_v.update(float(side.centroid_v_px), now)
                self._fside_area.update(float(side.image_area_px), now)
            self._side_facing = self._lowpass_angle(self._side_facing, float(side.yaw_facing_rad))
            # Continuous heading: tag yaw_facing corrected by layout azimuth.
            # Avoids ~90° jump when the visible tag switches.
            mid = int(side.marker_id) if side.marker_id is not None else -1
            az = self._side_azimuth.get(mid)
            if az is not None:
                raw_heading = self._unwrap_yaw(
                    self._side_yaw_sign * (float(side.yaw_facing_rad) - az) + self._side_yaw_offset
                )
                if self._has_side_heading:
                    self._side_heading = self._lowpass_angle(self._side_heading, raw_heading)
                else:
                    self._side_heading = raw_heading
                    self._has_side_heading = True
            self._side_ts = side_ts
            self._side_meas_ts = now
            self._has_side = True
        side_state = (
            TRACKING
            if (side.ok and now - self._side_meas_ts <= 1e-6)
            else (self._state_for_age(now - self._side_meas_ts) if self._has_side else LOST)
        )
        top_usable = top_state in (TRACKING, HOLDING_LAST)
        side_usable = side_state in (TRACKING, HOLDING_LAST)
        alt_from_side = self._stab.altitude_source.strip().lower() == "side"
        top_predicting = (now - self._top_meas_ts) <= max_h
        z_meas_ts = self._side_meas_ts if alt_from_side else self._top_meas_ts
        z_predicting = (now - z_meas_ts) <= max_h
        # --- heading (for body-frame odometry and control) ---
        top_yaw_usable = top_usable and self._top_n >= int(self._stab.yaw_top_min_markers)
        side_yaw_usable = side_usable and self._has_side_heading
        if top_yaw_usable and side_yaw_usable:
            target_off = self._unwrap_yaw(self._yaw - self._side_heading)
            if self._has_offset:
                self._side_to_top_offset = self._lowpass_angle(self._side_to_top_offset, target_off)
            else:
                self._side_to_top_offset = target_off
                self._has_offset = True
        aligned_side = self._unwrap_yaw(self._side_heading + self._side_to_top_offset)
        if top_yaw_usable:
            heading_for_body = self._yaw
            heading_valid = True
        elif side_yaw_usable and self._has_offset:
            heading_for_body = aligned_side
            heading_valid = True
        else:
            heading_for_body = self._yaw
            heading_valid = False
        # --- complementary odometry (optional; default: D-term only, not position) ---
        odom_active, odom_vx, odom_vy, odom_vz = self._fuse_odometry(
            odometry,
            heading_rad=heading_for_body,
            heading_valid=heading_valid,
        )
        top_age = now - self._top_meas_ts if self._has_top else 0.0
        side_age = now - self._side_meas_ts if self._has_side else 0.0
        if odom_active and self._odom_cfg.affect_filter:
            if self._has_top and top_usable:
                w = self._odom_blend_for_age(top_age)
                self._blend_axis_velocity(self._fx, odom_vx, w, top_age)
                self._blend_axis_velocity(self._fy, odom_vy, w, top_age)
            if self._odom_cfg.fuse_altitude:
                if alt_from_side and self._has_side and side_usable:
                    wz = self._odom_blend_for_age(side_age)
                    self._blend_axis_velocity(self._falt, odom_vz, wz, side_age)
                elif not alt_from_side and self._has_top and top_usable:
                    wz = self._odom_blend_for_age(top_age)
                    self._blend_axis_velocity(self._ftop_z, odom_vz, wz, top_age)
        # Predict position at current time (reduces delay, catches drift earlier).
        use_predict_ctrl = bool(predict and getattr(self._cfg, "predict_for_control", False))
        if self._has_top:
            if use_predict_ctrl:
                x_m = self._fx.predict(now, max_h)
                y_m = self._fy.predict(now, max_h)
                top_z = self._ftop_z.predict(now, max_h)
            else:
                x_m = self._fx.x if self._fx.x is not None else 0.0
                y_m = self._fy.x if self._fy.x is not None else 0.0
                top_z = self._ftop_z.x if self._ftop_z.x is not None else 0.0
            vx, vy = self._fx.v, self._fy.v
        else:
            x_m = y_m = vx = vy = top_z = 0.0
        if alt_from_side and self._has_side:
            if use_predict_ctrl:
                z_m = self._falt.predict(now, max_h)
            else:
                z_m = self._falt.x if self._falt.x is not None else 0.0
            vz = self._falt.v
            z_state, z_usable = side_state, side_usable
        elif not alt_from_side and self._has_top:
            z_m, vz = top_z, self._ftop_z.v
            z_state, z_usable = top_state, top_usable
        else:
            z_m, vz = 0.0, 0.0
            z_state, z_usable = side_state if alt_from_side else top_state, False
        active: list[str] = []
        freeze: list[str] = []
        if top_usable:
            active.extend(["x", "y"])
            if top_state != TRACKING:
                freeze.extend(["x", "y"])
                vx = vy = 0.0
            elif not top_predicting:
                freeze.extend(["x", "y"])
                vx = vy = 0.0
        if z_usable:
            active.append("z")
            z_track = side_state if alt_from_side else top_state
            if z_track != TRACKING:
                freeze.append("z")
                vz = 0.0
            elif not z_predicting:
                freeze.append("z")
                vz = 0.0
        if (
            odom_active
            and self._odom_cfg.use_for_derivative
            and not self._odom_cfg.affect_filter
        ):
            w_cap = float(self._odom_cfg.derivative_blend_max)
            if self._has_top and top_usable and top_state == TRACKING and top_age > 1e-6:
                w = min(self._odom_blend_for_age(top_age), w_cap)
                vx = (1.0 - w) * vx + w * odom_vx
                vy = (1.0 - w) * vy + w * odom_vy
            if self._odom_cfg.fuse_altitude:
                if alt_from_side and self._has_side and side_usable and side_age > 1e-6:
                    wz = min(self._odom_blend_for_age(side_age), w_cap)
                    vz = (1.0 - wz) * vz + wz * odom_vz
                elif not alt_from_side and self._has_top and top_usable and top_age > 1e-6:
                    wz = min(self._odom_blend_for_age(top_age), w_cap)
                    vz = (1.0 - wz) * vz + wz * odom_vz
        side_u = side_v = side_area = side_z = side_x = 0.0
        side_vu = side_vv = side_vz = side_vx = 0.0
        top_u = top_v = 0.0
        top_vu = top_vv = 0.0
        side_predicting = (now - self._side_meas_ts) <= max_h
        side_img = self._side_image_enabled(top_usable, side_usable, side_state)
        top_img = self._top_image_enabled(top_usable, side_usable, top_state, side_state)
        if bool(getattr(self._stab, "side_primary_horiz", False)) and (side_img or top_img):
            active = [a for a in active if a not in ("x", "y")]
            freeze = [a for a in freeze if a not in ("x", "y")]
        if self._has_side:
            side_x = self._fside_x.predict(now, max_h) if predict else (self._fside_x.x or 0.0)
            side_z = self._fside_z.predict(now, max_h) if predict else (self._fside_z.x or 0.0)
            side_vx, side_vz = self._fside_x.v, self._fside_z.v
            if side_usable and side_img and getattr(self._stab, "side_pnp_hold", False):
                active.append("side_x")
                if bool(getattr(self._stab, "side_depth_hold", False)):
                    active.append("side_depth")
                if side_state != TRACKING or not side_predicting:
                    freeze.append("side_x")
                    if "side_depth" in active:
                        freeze.append("side_depth")
                    side_vx = side_vz = 0.0
        if self._top_image_hold and self._has_top:
            top_u = self._ftop_u.x if self._ftop_u.x is not None else 0.0
            top_v = self._ftop_v.x if self._ftop_v.x is not None else 0.0
            top_vu, top_vv = self._ftop_u.v, self._ftop_v.v
            if top_usable and top_img:
                active.extend(["top_u", "top_v"])
                if top_state != TRACKING:
                    freeze.extend(["top_u", "top_v"])
                    top_vu = top_vv = 0.0
        if self._side_image_hold and self._has_side:
            side_u = self._fside_u.x if self._fside_u.x is not None else 0.0
            side_v = self._fside_v.x if self._fside_v.x is not None else 0.0
            side_area = self._fside_area.x if self._fside_area.x is not None else 0.0
            side_vu, side_vv = self._fside_u.v, self._fside_v.v
            if side_usable and side_img:
                if getattr(self._stab, "side_hold_u", True):
                    active.append("side_u")
                    if side_state != TRACKING:
                        freeze.append("side_u")
                        side_vu = 0.0
                if getattr(self._stab, "side_image_area_hold", False):
                    active.append("side_depth")
                    if side_state != TRACKING:
                        freeze.append("side_depth")
                if getattr(self._stab, "side_hold_v", False):
                    active.append("side_v")
                    if side_state != TRACKING:
                        freeze.append("side_v")
                        side_vv = 0.0
        # Yaw control input per selected source.
        yaw_meas = self._yaw
        if self._use_yaw and self._yaw_source != "off":
            if self._yaw_source == "side":
                yaw_meas = self._side_heading
                yaw_usable = side_yaw_usable
                yaw_predicting = (now - self._side_meas_ts) <= max_h
            elif self._yaw_source == "auto":
                if top_yaw_usable:
                    yaw_meas = self._yaw
                    yaw_usable = True
                    yaw_predicting = top_predicting
                elif side_yaw_usable:
                    yaw_meas = aligned_side if self._has_offset else self._side_heading
                    yaw_usable = True
                    yaw_predicting = (now - self._side_meas_ts) <= max_h
                else:
                    yaw_usable = False
                    yaw_predicting = False
            else:  # "top"
                yaw_meas = self._yaw
                yaw_usable = top_yaw_usable
                yaw_predicting = top_predicting
            if yaw_usable:
                active.append("yaw")
                if not yaw_predicting:
                    freeze.append("yaw")
        valid = bool(active)
        tracking_state = self._overall_state(top_state, side_state if alt_from_side else top_state)
        age = now - self._top_meas_ts if self._has_top else 0.0
        hints = [h for h in (top.hint, side.hint) if h]
        return DroneState(
            valid=valid,
            x_m=x_m,
            y_m=y_m,
            z_m=z_m,
            yaw_rad=yaw_meas,
            pitch_rad=self._pitch,
            roll_rad=self._roll,
            vx_m_s=vx,
            vy_m_s=vy,
            vz_m_s=vz,
            heading_rad=heading_for_body,
            heading_valid=bool(heading_valid),
            side_heading_rad=self._side_heading,
            side_heading_valid=bool(side_yaw_usable),
            top_ok=bool(top.ok),
            side_ok=bool(side.ok),
            top_state=top_state,
            side_state=side_state,
            tracking_state=tracking_state,
            active_axes=tuple(dict.fromkeys(active)),
            freeze_axes=tuple(dict.fromkeys(freeze)),
            pose_age_s=age,
            measurement_age_s=age,
            marker_id=top.reference_marker_id,
            hint=" | ".join(hints) or None,
            side_u_px=side_u,
            side_v_px=side_v,
            side_area_px=side_area,
            side_z_m=side_z,
            side_x_m=side_x,
            side_vu_px_s=side_vu,
            side_vv_px_s=side_vv,
            side_vz_m_s=side_vz,
            side_vx_m_s=side_vx,
            side_marker_id=side.marker_id,
            top_u_px=top_u,
            top_v_px=top_v,
            top_vu_px_s=top_vu,
            top_vv_px_s=top_vv,
            odom_fusion_active=odom_active,
            odom_vx_m_s=odom_vx,
            odom_vy_m_s=odom_vy,
            odom_vz_m_s=odom_vz,
        )

    def _odom_blend_for_age(self, meas_age_s: float) -> float:
        cfg = self._odom_cfg
        if meas_age_s <= 1e-6:
            return 0.0
        t = min(1.0, meas_age_s / max(float(cfg.stale_ramp_s), 1e-3))
        lo = float(cfg.velocity_blend)
        hi = float(cfg.velocity_blend_max)
        return lo + (hi - lo) * t

    @staticmethod
    def _blend_axis_velocity(axis: AlphaBetaAxis, v_odom: float, blend: float, meas_age_s: float) -> None:
        if axis.x is None or blend <= 0.0 or meas_age_s <= 1e-6:
            return
        w = float(np.clip(blend, 0.0, 1.0))
        axis.v = float(np.clip((1.0 - w) * axis.v + w * v_odom, -axis.max_speed, axis.max_speed))

    def _fuse_odometry(
        self,
        odometry: OdometrySample | None,
        *,
        heading_rad: float,
        heading_valid: bool,
    ) -> tuple[bool, float, float, float]:
        cfg = self._odom_cfg
        if not cfg.enabled or odometry is None or not odometry.valid:
            return False, 0.0, 0.0, 0.0
        if cfg.require_flying and not odometry.flying:
            return False, 0.0, 0.0, 0.0
        if cfg.require_heading and not heading_valid:
            return False, 0.0, 0.0, 0.0
        vmax = min(float(cfg.max_speed_m_s), float(self._cfg.max_speed_m_s))
        fwd = float(np.clip(cfg.speed_x_sign * odometry.vx_m_s, -vmax, vmax))
        lat = float(np.clip(cfg.speed_y_sign * odometry.vy_m_s, -vmax, vmax))
        vz = float(np.clip(cfg.speed_z_sign * odometry.vz_m_s, -vmax, vmax))
        db = max(0.0, float(cfg.deadband_m_s))
        if max(abs(fwd), abs(lat), abs(vz)) < db:
            return False, 0.0, 0.0, 0.0
        c, s = float(np.cos(heading_rad)), float(np.sin(heading_rad))
        # Match controller.py world<->body (yaw=0: forward=+y, lateral=+x).
        vx_w = -s * fwd + c * lat
        vy_w = c * fwd + s * lat
        return True, vx_w, vy_w, vz

    @staticmethod
    def _overall_state(a: str, b: str) -> str:
        order = {TRACKING: 0, HOLDING_LAST: 1, LOST: 2, TIMED_OUT: 3}
        return min(a, b, key=lambda s: order.get(s, 3))

    def _lowpass_angle(self, prev: float, meas: float) -> float:
        alpha = float(np.clip(self._cfg.pose_lowpass_alpha, 0.0, 1.0))
        if alpha <= 0.0:
            return self._unwrap_yaw(meas)
        delta = np.arctan2(np.sin(meas - prev), np.cos(meas - prev))
        return self._unwrap_yaw(prev + (1.0 - alpha) * delta)

    @staticmethod
    def _unwrap_yaw(yaw: float) -> float:
        return float(np.arctan2(np.sin(yaw), np.cos(yaw)))


StateEstimator = PoseEstimator
