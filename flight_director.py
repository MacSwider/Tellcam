"""
Flight state machine: start -> climb to ~1.0 m -> hold in place
                      -> (TRAVEL: path flight) -> landing.
- CLIMB: open loop (gentle up) until cameras acquire markers; height from Tello
  sensor with time limit (failsafe).
- HOLD: closed loop on TOP+SIDE fusion; target = pose captured on entry
  (hold current position). Brief marker loss handled via active_axes/freeze_axes
  from the estimator.
- TRAVEL: sequential waypoint flight. On TRACKING LOSS the maneuver pauses —
  drone holds position — and resumes when tracking returns.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from config import (
    ClimbConfig,
    GeofenceConfig,
    NavigationConfig,
    SafetyConfig,
    StabilizationConfig,
    TargetConfig,
)
from controller import RCCommand, StabilizationController
from state_estimator import DroneState
from trajectory_planner import (
    FutureTrajectoryPlanner,
    LoadedFlightPath,
    TrajectoryPoint,
    cumulative_body_offsets,
    world_delta_to_body,
)

log = logging.getLogger(__name__)


def _angle_diff(a: float, b: float) -> float:
    """Shortest angle difference a-b in (-pi, pi]."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


IDLE = "IDLE"
CLIMB = "CLIMB"
HOLD = "HOLD"
TRAVEL = "TRAVEL"
LANDING = "LANDING"


@dataclass
class _GeofenceBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


class FlightDirector:
    def __init__(
        self,
        controller: StabilizationController,
        stab_cfg: StabilizationConfig,
        climb_cfg: ClimbConfig,
        safety_cfg: SafetyConfig,
        nav_cfg: NavigationConfig | None = None,
        planner: FutureTrajectoryPlanner | None = None,
        flight_path: LoadedFlightPath | None = None,
    ) -> None:
        self._controller = controller
        self._stab = stab_cfg
        self._climb = climb_cfg
        self._safety = safety_cfg
        self._nav = nav_cfg or NavigationConfig()
        self._planner = planner
        self._flight_path = flight_path or LoadedFlightPath()
        self.phase = IDLE
        self.reason = ""
        self._climb_t0 = 0.0
        self._hold_t0 = 0.0
        self._target_captured = False
        # --- Anti-spin: locked reference heading + spin watchdog ---
        self._yaw_lock_rad: float | None = None
        self._yaw_offset = math.radians(float(getattr(stab_cfg, "yaw_reference_offset_deg", 0.0)))
        self._prev_yaw: float | None = None
        self._prev_yaw_t = 0.0
        self.spin_rate_deg_s = 0.0
        self.spin_active = False
        # Camera is high — TAG 0 becomes visible only after climb.
        # Lock reference heading after CLIMB target height is reached.
        self._climb_height_reached = False
        # --- TRAVEL: waypoint flight + pause on tracking loss ---
        self._travel_total = 0
        self._travel_paused = False
        self._wp_reached_t: float | None = None
        self._waypoints: list[TrajectoryPoint] = []
        self._travel_start: TrajectoryPoint | None = None
        self._side_travel_anchor: tuple[float, float] | None = None
        self._side_wp_targets: list[tuple[float, float]] = []
        # --- Geofence: safe zone relative to reference point ---
        self._geofence_bounds: _GeofenceBounds | None = None
        self.geofence_breached = False

    @property
    def geofence_enabled(self) -> bool:
        return bool(self._safety.geofence.enabled)

    @property
    def geofence_bounds(self) -> _GeofenceBounds | None:
        return self._geofence_bounds

    def start_flight(self, now: float) -> None:
        """Call after physical takeoff()."""
        self.phase = CLIMB
        self.reason = "start: climbing"
        self._climb_t0 = now
        self._target_captured = False
        self._yaw_lock_rad = None
        self._prev_yaw = None
        self.spin_active = False
        self._climb_height_reached = False
        self._geofence_bounds = None
        self.geofence_breached = False
        self._controller.reset()
        log.info("FlightDirector: CLIMB to %.2f m", self._climb.target_height_m)

    def request_land(self) -> None:
        self.phase = LANDING
        self.reason = "landing"

    @property
    def has_path(self) -> bool:
        return self._flight_path.non_empty and self._planner is not None

    @property
    def is_traveling(self) -> bool:
        return self.phase == TRAVEL

    @property
    def travel_done(self) -> int:
        """Number of reached waypoints (for preview)."""
        if self._planner is None:
            return 0
        return max(0, self._travel_total - self._planner.pending)

    @property
    def travel_waypoints(self) -> list[TrajectoryPoint]:
        """Full waypoint list for current route (for preview drawing)."""
        return self._waypoints

    @property
    def travel_start(self) -> TrajectoryPoint | None:
        """Route start point (pose when TRAVEL was pressed)."""
        return self._travel_start

    @property
    def travel_current_index(self) -> int:
        """0-based index of the waypoint currently being executed."""
        return min(self.travel_done, max(0, self._travel_total - 1))

    def request_travel(self, state: DroneState, now: float) -> bool:
        """Start path flight from HOLD.

        Absolute paths (`points` with X/Y/Z): fly to fixed lab-frame coordinates [m].
        Legacy paths (`steps`): integrate relative moves from current hover pose.
        """
        if not self.has_path:
            log.warning("TRAVEL: no flight path loaded — skipping.")
            return False
        pose_ok = state.valid and self._acquired_for_travel(state)
        if self.phase != HOLD or not pose_ok:
            log.warning(
                "TRAVEL: allowed only in HOLD with full pose (phase=%s, valid=%s, guidance=%s).",
                self.phase,
                state.valid,
                self._nav.travel_guidance,
            )
            return False
        start = TrajectoryPoint(
            x_m=state.x_m, y_m=state.y_m, z_m=state.z_m, yaw_rad=self._desired_yaw(state)
        )
        if self._flight_path.absolute:
            waypoints = self._planner.load_absolute_waypoints(self._flight_path.waypoints)
            log.info(
                "TRAVEL: absolute path — %d points (unit=%s)",
                len(waypoints),
                self._flight_path.coordinate_unit,
            )
        else:
            waypoints = self._planner.load_path(
                start, self._flight_path.steps, self._nav.units_to_meter
            )
        if not waypoints:
            log.warning("TRAVEL: path empty after conversion — skipping.")
            return False
        self._capture_geofence(start.x_m, start.y_m, start.z_m)
        if self._geofence_cfg().block_travel_outside and self._geofence_bounds is not None:
            for i, wp in enumerate(waypoints):
                if not self._point_in_geofence(wp.x_m, wp.y_m, wp.z_m):
                    viol = self._geofence_violation(wp.x_m, wp.y_m, wp.z_m)
                    log.warning("TRAVEL: waypoint %d outside geofence (%s) — blocked.", i + 1, viol)
                    return False
        self._waypoints = waypoints
        self._travel_start = start
        self._travel_total = len(waypoints)
        self._travel_paused = False
        self._wp_reached_t = None
        if self._travel_uses_side():
            anchor = (float(state.side_x_m), float(state.side_z_m))
            self._side_travel_anchor = anchor
            if self._flight_path.absolute:
                self._side_wp_targets = []
                for wp in waypoints:
                    dx, dy = wp.x_m - start.x_m, wp.y_m - start.y_m
                    fwd, lat = world_delta_to_body(dx, dy, start.yaw_rad)
                    self._side_wp_targets.append(
                        self._body_to_side_target(anchor[0], anchor[1], fwd, lat)
                    )
            else:
                offsets = cumulative_body_offsets(
                    self._flight_path.steps, self._nav.units_to_meter
                )
                self._side_wp_targets = [
                    self._body_to_side_target(anchor[0], anchor[1], fwd, lat)
                    for fwd, lat in offsets
                ]
            log.info(
                "TRAVEL: SIDE PnP anchor (x=%.3f, z=%.3f), %d side targets",
                anchor[0],
                anchor[1],
                len(self._side_wp_targets),
            )
        else:
            self._side_travel_anchor = None
            self._side_wp_targets = []
        self.phase = TRAVEL
        self._controller.reset()
        self.reason = f"travel: start ({self._travel_total} wp)"
        log.info(
            "TRAVEL: route start — %d waypoints from (%.2f, %.2f, %.2f)",
            self._travel_total,
            start.x_m,
            start.y_m,
            start.z_m,
        )
        return True

    def reset(self) -> None:
        self.phase = IDLE
        self.reason = ""
        self._target_captured = False
        self._yaw_lock_rad = None
        self._prev_yaw = None
        self.spin_active = False
        self._climb_height_reached = False
        self._travel_paused = False
        self._wp_reached_t = None
        self._waypoints = []
        self._travel_start = None
        self._travel_total = 0
        self._side_travel_anchor = None
        self._side_wp_targets = []
        self._geofence_bounds = None
        self.geofence_breached = False
        self._controller.reset()

    @property
    def in_flight_sequence(self) -> bool:
        return self.phase in (CLIMB, HOLD, TRAVEL)

    def _acquired(self, state: DroneState) -> bool:
        need = {"x", "y", "z"}
        return need.issubset(set(state.active_axes))

    def _travel_uses_side(self) -> bool:
        return (self._nav.travel_guidance or "top").strip().lower() == "side"

    def _acquired_for_travel(self, state: DroneState) -> bool:
        if self._travel_uses_side():
            need = {"z", "side_x", "side_depth"}
            return need.issubset(set(state.active_axes))
        return self._acquired(state)

    def _body_to_side_target(
        self, anchor_x: float, anchor_z: float, fwd_m: float, lat_m: float
    ) -> tuple[float, float]:
        tx = anchor_x + float(self._nav.side_travel_lat_sign) * float(lat_m)
        tz = anchor_z + float(self._nav.side_travel_fwd_sign) * float(fwd_m)
        return tx, tz

    def _maybe_lock_yaw(self, state: DroneState) -> None:
        """Lock reference heading from TAG 0 — only after CLIMB target height.
        The camera is high enough that TAG 0 is not visible near the floor,
        so we lock after target height (~1.0 m) when the tag is in frame and
        heading is trustworthy.
        """
        if not getattr(self._stab, "yaw_lock_on_takeoff", True):
            return
        if not self._climb_height_reached:
            return
        if self._yaw_lock_rad is not None or not state.heading_valid:
            return
        self._yaw_lock_rad = _angle_diff(state.yaw_rad + self._yaw_offset, 0.0)
        log.info(
            "FlightDirector: locked reference heading yaw=%.1f deg (anti-spin)",
            math.degrees(self._yaw_lock_rad),
        )

    def _desired_yaw(self, state: DroneState) -> float:
        """Target heading: locked start orientation when available."""
        yaw_source = (self._stab.yaw_source or "off").strip().lower()
        if self._yaw_lock_rad is not None and yaw_source in ("top", "auto"):
            return self._yaw_lock_rad
        # For yaw "side", target is 0 (tag facing the side camera).
        return 0.0 if yaw_source != "top" else state.yaw_rad

    def _update_spin_watchdog(self, state: DroneState, now: float) -> None:
        """Estimate heading angular rate and set anti-spin flag."""
        limit = float(getattr(self._stab, "spin_rate_limit_deg_s", 0.0))
        if not state.heading_valid or limit <= 0.0:
            self.spin_active = False
            self._prev_yaw = state.yaw_rad if state.heading_valid else None
            self._prev_yaw_t = now
            return
        if self._prev_yaw is not None:
            dt = max(1e-3, now - self._prev_yaw_t)
            rate = math.degrees(_angle_diff(state.yaw_rad, self._prev_yaw)) / dt
            # Light smoothing so a single spike does not trigger the mode.
            self.spin_rate_deg_s = 0.6 * self.spin_rate_deg_s + 0.4 * rate
            spinning = abs(self.spin_rate_deg_s) > limit
            if spinning and not self.spin_active:
                log.warning(
                    "FlightDirector: ANTI-SPIN — detected rotation %.0f deg/s (>%.0f). "
                    "Damping horizontal motion. If spin worsens, check yaw_sign!",
                    self.spin_rate_deg_s,
                    limit,
                )
            self.spin_active = spinning
        self._prev_yaw = state.yaw_rad
        self._prev_yaw_t = now

    def _geofence_cfg(self) -> GeofenceConfig:
        return self._safety.geofence

    def _capture_geofence(self, x_m: float, y_m: float, z_m: float) -> None:
        """Set rectangular fence around reference point."""
        gf = self._geofence_cfg()
        if not gf.enabled:
            self._geofence_bounds = None
            return
        self._geofence_bounds = _GeofenceBounds(
            x_min=float(x_m) - float(gf.half_x_m),
            x_max=float(x_m) + float(gf.half_x_m),
            y_min=float(y_m) - float(gf.half_y_m),
            y_max=float(y_m) + float(gf.half_y_m),
            z_min=float(z_m) - float(gf.z_below_m),
            z_max=float(z_m) + float(gf.z_above_m),
        )
        log.info(
            "Geofence: x[%.2f..%.2f] y[%.2f..%.2f] z[%.2f..%.2f]",
            self._geofence_bounds.x_min,
            self._geofence_bounds.x_max,
            self._geofence_bounds.y_min,
            self._geofence_bounds.y_max,
            self._geofence_bounds.z_min,
            self._geofence_bounds.z_max,
        )

    def _point_in_geofence(self, x_m: float, y_m: float, z_m: float) -> bool:
        b = self._geofence_bounds
        if b is None:
            return True
        return (
            b.x_min <= float(x_m) <= b.x_max
            and b.y_min <= float(y_m) <= b.y_max
            and b.z_min <= float(z_m) <= b.z_max
        )

    def _geofence_violation(self, x_m: float, y_m: float, z_m: float) -> str:
        b = self._geofence_bounds
        if b is None:
            return ""
        parts: list[str] = []
        if float(x_m) < b.x_min:
            parts.append("x-")
        elif float(x_m) > b.x_max:
            parts.append("x+")
        if float(y_m) < b.y_min:
            parts.append("y-")
        elif float(y_m) > b.y_max:
            parts.append("y+")
        if float(z_m) < b.z_min:
            parts.append("z-")
        elif float(z_m) > b.z_max:
            parts.append("z+")
        return "/".join(parts)

    def _enforce_geofence(self, state: DroneState, *, context: str) -> RCCommand | None:
        """Return zero RC when drone is outside zone; None = OK, continue."""
        if not self._geofence_cfg().enabled or self._geofence_bounds is None or not state.valid:
            return None
        if self._point_in_geofence(state.x_m, state.y_m, state.z_m):
            if self.geofence_breached:
                self.geofence_breached = False
                log.info("Geofence: returned to safe zone.")
            return None
        viol = self._geofence_violation(state.x_m, state.y_m, state.z_m)
        if not self.geofence_breached:
            log.warning("Geofence: OUTSIDE ZONE (%s) — hover until return.", viol)
        self.geofence_breached = True
        self.reason = f"{context}: GEOFENCE ({viol})"
        return RCCommand.zero()

    def _has_hold_control(self, state: DroneState) -> bool:
        """True when HOLD can command at least horizontal correction."""
        axes = set(state.active_axes)
        has_horiz = bool(axes & {"x", "y", "top_u", "top_v", "side_u", "side_x", "side_depth"})
        if not has_horiz:
            return False
        if state.tracking_state == "TIMED_OUT":
            return False
        return state.valid or state.tracking_state == "HOLDING_LAST"

    def _side_hold_targets(self, state: DroneState) -> dict[str, float | None]:
        """SIDE hold targets (captured on HOLD entry)."""
        if not getattr(self._stab, "side_image_hold", False) or state.side_marker_id is None:
            return {}
        out: dict[str, float | None] = {}
        if getattr(self._stab, "side_hold_u", True):
            out["side_u_px"] = float(state.side_u_px)
        if getattr(self._stab, "side_hold_v", False):
            out["side_v_px"] = float(state.side_v_px)
        if getattr(self._stab, "side_image_area_hold", False):
            out["side_area_px"] = float(state.side_area_px)
        depth_src = (getattr(self._stab, "side_depth_source", "area") or "area").strip().lower()
        if depth_src in ("z_m", "both") and getattr(self._stab, "side_depth_hold", False):
            out["side_z_m"] = float(state.side_z_m)
        if getattr(self._stab, "side_pnp_hold", False):
            out["side_x_m"] = float(state.side_x_m)
        return out

    def _top_hold_targets(self, state: DroneState) -> dict[str, float | None]:
        if not getattr(self._stab, "top_image_hold", False):
            return {}
        return {
            "top_u_px": float(state.top_u_px),
            "top_v_px": float(state.top_v_px),
        }

    def _capture_hold_target(self, state: DroneState) -> None:
        if not self._stab.hold_capture_on_enter:
            return
        self._maybe_lock_yaw(state)
        yaw_target = self._desired_yaw(state)
        self._controller.set_target(
            TargetConfig(
                x_m=state.x_m,
                y_m=state.y_m,
                z_m=state.z_m,
                yaw_rad=yaw_target,
                **self._top_hold_targets(state),
                **self._side_hold_targets(state),
            )
        )
        self._capture_geofence(state.x_m, state.y_m, state.z_m)
        self._target_captured = True
        side = self._controller.last_debug.target
        log.info(
            "FlightDirector: HOLD — target x=%.2f y=%.2f z=%.2f yaw=%.2f | SIDE u=%s area=%s z=%s",
            state.x_m,
            state.y_m,
            state.z_m,
            yaw_target,
            side.side_u_px,
            side.side_area_px,
            side.side_z_m,
        )

    def update(self, state: DroneState, height_m: float, dt: float, now: float) -> RCCommand:
        if self.phase == IDLE or self.phase == LANDING:
            return RCCommand.zero()
        # Spin watchdog runs in every flight phase (anti-spin).
        self._update_spin_watchdog(state, now)
        if self.phase == CLIMB:
            height_ok = height_m >= float(self._climb.target_height_m)
            # After target height TAG 0 is in frame -> lock reference heading
            # and enable yaw correction (anti-spin).
            if height_ok:
                self._climb_height_reached = True
            self._maybe_lock_yaw(state)
            acquired = self._acquired(state)
            timed_out = (now - self._climb_t0) >= float(self._climb.timeout_s)
            ready = height_ok and (acquired or not self._climb.require_top_and_side)
            if ready or timed_out:
                self.phase = HOLD
                self._hold_t0 = now
                # In HOLD we are hovering — allow heading lock even after timeout.
                self._climb_height_reached = True
                self.reason = "hold (cameras)" if not timed_out else "hold (climb timeout)"
                self._controller.reset()
                if acquired:
                    self._capture_hold_target(state)
                return RCCommand.zero()
            # Gentle climb; slow down near target height to avoid overshoot.
            up = int(self._climb.rc_up if not height_ok else max(0, self._climb.rc_up // 3))
            # Anti-spin: correct yaw only when reference heading is locked (after
            # climb and TAG 0 detection). Near floor the drone only climbs.
            yaw_cmd = 0
            if (
                getattr(self._stab, "yaw_control_during_climb", True)
                and self._yaw_lock_rad is not None
                and state.heading_valid
            ):
                yaw_cmd = self._controller.compute_yaw_command(
                    state.yaw_rad, self._yaw_lock_rad, dt
                )
            spin = " [ANTI-SPIN]" if self.spin_active else ""
            self.reason = f"climbing h={height_m:.2f}/{self._climb.target_height_m:.2f}m{spin}"
            return RCCommand(0, 0, up, yaw_cmd)
        if self.phase == TRAVEL:
            return self._update_travel(state, dt, now)
        # HOLD
        if not self._target_captured:
            if self._acquired(state):
                self._capture_hold_target(state)
            else:
                self.reason = "hold: waiting for full pose (TOP+SIDE)"
                return RCCommand.zero()
        if not self._has_hold_control(state):
            self.reason = f"hold: no pose ({state.tracking_state} axes={state.active_axes})"
            return RCCommand.zero()
        gf_cmd = self._enforce_geofence(state, context="hold")
        if gf_cmd is not None:
            return gf_cmd
        self.reason = f"hold: {state.tracking_state}"
        return self._position_command(state, dt, hold_reason="hold", apply_hold_creep=True)

    def _position_command(
        self,
        state: DroneState,
        dt: float,
        *,
        hold_reason: str,
        apply_side_hold: bool = True,
        apply_side_pnp_travel: bool = False,
        apply_hold_creep: bool = False,
    ) -> RCCommand:
        """Full position control (roll/pitch/throttle/yaw) to current target."""
        cmd = self._controller.compute(
            state.x_m,
            state.y_m,
            state.z_m,
            state.yaw_rad,
            dt,
            active_axes=state.active_axes,
            freeze_axes=state.freeze_axes,
            vx=state.vx_m_s,
            vy=state.vy_m_s,
            vz=state.vz_m_s,
            heading_rad=state.heading_rad,
            heading_valid=state.heading_valid,
            side_u_px=state.side_u_px,
            side_v_px=state.side_v_px,
            side_area_px=state.side_area_px,
            side_z_m=state.side_z_m,
            side_vu_px_s=state.side_vu_px_s,
            side_vv_px_s=state.side_vv_px_s,
            side_vz_m_s=state.side_vz_m_s,
            side_x_m=state.side_x_m,
            side_vx_m_s=state.side_vx_m_s,
            top_u_px=state.top_u_px,
            top_v_px=state.top_v_px,
            top_vu_px_s=state.top_vu_px_s,
            top_vv_px_s=state.top_vv_px_s,
            apply_side_hold=apply_side_hold,
            apply_side_pnp_travel=apply_side_pnp_travel,
        )
        # Anti-spin: when spinning, align heading first — damp horizontal motion
        # so the drone does not arc away. Keep altitude (up_down) and yaw correction.
        if self.spin_active:
            self.reason = f"{hold_reason}: [ANTI-SPIN {self.spin_rate_deg_s:+.0f}deg/s]"
            return RCCommand(0, 0, cmd.up_down, cmd.yaw)
        cmd = self._apply_side_trim(cmd)
        if apply_hold_creep:
            cmd = self._apply_hold_creep(cmd, state)
        return cmd

    def _apply_hold_creep(self, cmd: RCCommand, state: DroneState) -> RCCommand:
        """Add constant lab-frame velocity feedforward (counter steady drift)."""
        if not bool(getattr(self._stab, "hold_creep_enabled", False)):
            return cmd
        vx = float(getattr(self._stab, "hold_creep_vx_m_s", 0.0) or 0.0)
        vy = float(getattr(self._stab, "hold_creep_vy_m_s", 0.0) or 0.0)
        vz = float(getattr(self._stab, "hold_creep_vz_m_s", 0.0) or 0.0)
        if abs(vx) < 1e-6 and abs(vy) < 1e-6 and abs(vz) < 1e-6:
            return cmd
        gain = float(getattr(self._stab, "hold_creep_rc_per_m_s", 90.0) or 90.0)
        if gain <= 0.0:
            return cmd
        limit = min(int(self._safety.max_rc_abs), 100)
        heading = float(state.heading_rad)
        heading_ok = bool(state.heading_valid)
        if bool(getattr(self._stab, "body_frame_control", True)) and heading_ok:
            c, s = math.cos(heading), math.sin(heading)
            fwd_v = -s * vx + c * vy
            lat_v = c * vx + s * vy
        else:
            fwd_v, lat_v = vy, vx
        roll_sign = float(getattr(self._controller._cfg, "roll_sign", 1.0))
        pitch_sign = float(getattr(self._controller._cfg, "pitch_sign", 1.0))
        throttle_sign = float(getattr(self._controller._cfg, "throttle_sign", 1.0))
        lr = int(cmd.left_right) + int(round(roll_sign * lat_v * gain))
        fb = int(cmd.forward_back) + int(round(pitch_sign * fwd_v * gain))
        ud = int(cmd.up_down) + int(round(throttle_sign * vz * gain))
        lr = max(-limit, min(limit, lr))
        fb = max(-limit, min(limit, fb))
        ud = max(-limit, min(limit, ud))
        if lr == cmd.left_right and fb == cmd.forward_back and ud == cmd.up_down:
            return cmd
        return RCCommand(lr, fb, ud, cmd.yaw)

    def _apply_side_trim(self, cmd: RCCommand) -> RCCommand:
        limit = min(int(self._safety.max_rc_abs), 100)
        lr = int(cmd.left_right)
        fb = int(cmd.forward_back)
        lat_trim = int(getattr(self._stab, "side_lateral_trim_rc", 0) or 0)
        fwd_trim = int(getattr(self._stab, "side_forward_trim_rc", 0) or 0)
        if lat_trim != 0:
            lr = max(-limit, min(limit, lr + lat_trim))
        if fwd_trim != 0:
            fb = max(-limit, min(limit, fb + fwd_trim))
        if lr == cmd.left_right and fb == cmd.forward_back:
            return cmd
        return RCCommand(lr, fb, cmd.up_down, cmd.yaw)

    def _update_travel(self, state: DroneState, dt: float, now: float) -> RCCommand:
        assert self._planner is not None
        side_travel = self._travel_uses_side()
        # --- Safety: tracking loss -> PAUSE, drone holds position ---
        if not (state.valid and self._acquired_for_travel(state)):
            if not self._travel_paused:
                self._travel_paused = True
                self._controller.reset()  # no integrator jump on resume
                log.warning(
                    "TRAVEL: PAUSED — tracking lost (%s). Drone holding position.",
                    state.tracking_state,
                )
            self._wp_reached_t = None
            self.reason = f"travel: PAUSED (tracking lost: {state.tracking_state})"
            return RCCommand.zero()
        if self._travel_paused:
            self._travel_paused = False
            self._controller.reset()
            log.info("TRAVEL: resumed after tracking recovered.")
        tgt = self._planner.peek_target()
        if (
            self._geofence_cfg().block_travel_outside
            and self._geofence_bounds is not None
            and not self._point_in_geofence(tgt.x_m, tgt.y_m, tgt.z_m)
        ):
            viol = self._geofence_violation(tgt.x_m, tgt.y_m, tgt.z_m)
            log.warning("TRAVEL: target outside geofence (%s) — aborting, HOLD.", viol)
            self.phase = HOLD
            self._target_captured = False
            self._capture_hold_target(state)
            self.reason = f"travel: GEOFENCE target ({viol}) -> HOLD"
            return RCCommand.zero()
        gf_cmd = self._enforce_geofence(state, context="travel")
        if gf_cmd is not None:
            return gf_cmd
        prev = self._controller.last_debug.target
        side_tx = prev.side_x_m
        side_tz = prev.side_z_m
        wp_idx = self.travel_done
        if side_travel and wp_idx < len(self._side_wp_targets):
            side_tx, side_tz = self._side_wp_targets[wp_idx]
        self._controller.set_target(
            TargetConfig(
                x_m=tgt.x_m,
                y_m=tgt.y_m,
                z_m=tgt.z_m,
                yaw_rad=tgt.yaw_rad,
                side_u_px=prev.side_u_px,
                side_v_px=prev.side_v_px,
                side_area_px=prev.side_area_px,
                side_z_m=side_tz if side_travel else prev.side_z_m,
                side_x_m=side_tx if side_travel else prev.side_x_m,
            )
        )
        if side_travel and side_tx is not None and side_tz is not None:
            reached = self._planner.reached_side_pnp(
                state.side_x_m,
                state.side_z_m,
                float(side_tx),
                float(side_tz),
                radius_m=float(self._nav.waypoint_radius_m),
            )
            speed = math.hypot(state.side_vx_m_s, state.side_vz_m_s)
        else:
            reached = self._planner.reached(
                state.x_m,
                state.y_m,
                state.z_m,
                state.yaw_rad,
                radius_m=float(self._nav.waypoint_radius_m),
                yaw_tol_deg=float(self._nav.waypoint_yaw_tol_deg),
            )
            speed = math.hypot(state.vx_m_s, state.vy_m_s)
        # Speed gate: drone must stabilize (low speed) before advancing.
        speed_gate = float(getattr(self._nav, "waypoint_speed_gate_m_s", 0.0))
        settled = reached and (speed_gate <= 0.0 or speed <= speed_gate)
        if settled:
            if self._wp_reached_t is None:
                self._wp_reached_t = now
            elif (now - self._wp_reached_t) >= float(self._nav.waypoint_settle_s):
                self._planner.advance_if_reached()
                self._wp_reached_t = None
                if self._planner.pending == 0:
                    # Route complete -> HOLD at end point.
                    self.phase = HOLD
                    self._target_captured = False
                    self._capture_hold_target(state)
                    self.reason = "travel: COMPLETE -> HOLD"
                    log.info("TRAVEL: route complete — entering HOLD.")
                    return RCCommand.zero()
        elif not reached:
            self._wp_reached_t = None
        done = self.travel_done
        if side_travel and side_tx is not None and side_tz is not None:
            dist = math.dist((state.side_x_m, state.side_z_m), (float(side_tx), float(side_tz)))
            self.reason = f"travel: SIDE wp {done + 1}/{self._travel_total} d={dist:.2f}m"
        else:
            dist = math.dist((state.x_m, state.y_m, state.z_m), (tgt.x_m, tgt.y_m, tgt.z_m))
            self.reason = f"travel: wp {done + 1}/{self._travel_total} d={dist:.2f}m"
        return self._position_command(
            state,
            dt,
            hold_reason="travel",
            apply_side_hold=False,
            apply_side_pnp_travel=side_travel,
        )
