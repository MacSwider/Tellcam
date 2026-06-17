"""Tellcam — AprilTag tag36h11 navigation (TOP + SIDE) for Ryze Tello."""

from __future__ import annotations

import argparse
import logging
import math
import queue
import sys
import time
import cv2
import numpy as np
from config import AppConfig, load_config
from controller import DroneController
from flight_director import FlightDirector
from state_estimator import DroneState, OdometrySample
from tello_interface import TelloConfig, TelloController
from trajectory_planner import FutureTrajectoryPlanner, LoadedFlightPath, TrajectoryPoint, load_flight_path
from video_source import create_video_source
from flight_hud import FlightHudSnapshot
from vision_pipeline import PipelineConfig, TrajectoryOverlay, build_vision_pipeline

log = logging.getLogger(__name__)


def _project_top_points(K, dist, pts_xy, depth_m: float) -> list[tuple[int, int]]:
    """Project world points (TOP camera) to pixels.
    Waypoints have x,y in meters in the CAMERA FRAME (same as drone pose from PnP),
    so we project them with zero rvec/tvec at the given depth (depth_m = current
    distance from the ceiling camera). This gives correct meter->pixel scale
    in the drone plane.
    """
    obj = np.array([[float(x), float(y), float(depth_m)] for (x, y) in pts_xy], dtype=np.float64)
    img, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K, dist)
    img = img.reshape(-1, 2)
    return [(int(round(p[0])), int(round(p[1]))) for p in img]


def _short_alert(
    state: DroneState,
    top_ref_seen: bool,
    top_obs,
    side_obs,
    *,
    reference_tag_id: int,
    geofence_breached: bool = False,
) -> str:
    """One short hint instead of long hints in the panel."""
    if geofence_breached:
        return "Outside safe zone — drone waiting to return"
    if state.top_state == "TIMED_OUT" or state.side_state == "TIMED_OUT":
        return "Tracking lost — drone holding position"
    if not side_obs.markers_seen:
        return "No tags 0–4 in SIDE frame"
    if reference_tag_id == 0 and not top_ref_seen:
        return "TAG 0 not visible — horizontal stabilization limited"
    if not top_obs.markers_seen:
        return "No tags in TOP frame"
    hint = state.hint or top_obs.hint or side_obs.hint or ""
    if "No tags" in hint:
        return "Check focus, lighting, and marker size (40 mm)"
    return ""


def _publish_trajectory_overlay(shared, director, state, top_obs, top_detection) -> None:
    """Build and publish route preview (TRAVEL phase only with valid TOP pose)."""
    if not (
        director.is_traveling
        and top_detection is not None
        and top_obs.ok
        and float(top_obs.z_m) > 1e-3
        and director.travel_waypoints
    ):
        shared.publish_trajectory(None)
        return
    try:
        K = top_detection.camera_matrix
        dist = top_detection.dist_coeffs
        depth = float(top_obs.z_m)
        wps_px = _project_top_points(
            K, dist, [(w.x_m, w.y_m) for w in director.travel_waypoints], depth
        )
        start = director.travel_start
        start_px = (
            _project_top_points(K, dist, [(start.x_m, start.y_m)], depth)[0] if start else None
        )
        drone_px = (
            _project_top_points(K, dist, [(state.x_m, state.y_m)], depth)[0]
            if state.valid
            else None
        )
        shared.publish_trajectory(
            TrajectoryOverlay(
                waypoints_px=wps_px,
                start_px=start_px,
                drone_px=drone_px,
                current_index=director.travel_current_index,
                total=len(wps_px),
            )
        )
    except Exception:  # preview must not crash the control loop
        log.debug("Route preview: projection failed", exc_info=True)
        shared.publish_trajectory(None)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tellcam — AprilTag TOP/SIDE navigation + (optional) Tello"
    )
    p.add_argument("--config", type=str, default=None, help="JSON config overrides")
    p.add_argument(
        "--no-control", action="store_true", help="Disable RC to drone (detection/preview only)"
    )
    p.add_argument(
        "--connect-tello",
        action="store_true",
        help="Connect to Tello at startup (default: only after START click)",
    )
    p.add_argument(
        "--mock-tello", action="store_true", help="Simulate Tello (on START, no physical drone)"
    )
    p.add_argument("--no-windows", action="store_true", help="Disable OpenCV windows")
    for prefix in ("top", "side"):
        p.add_argument(
            f"--{prefix}-capture", type=str, default="usb", choices=("usb", "screen", "window")
        )
        p.add_argument(f"--{prefix}-camera-index", type=int, default=None)
        p.add_argument(f"--{prefix}-screen-monitor", type=int, default=1)
        p.add_argument(f"--{prefix}-window-title", type=str, default="")
        p.add_argument(f"--{prefix}-window-match-index", type=int, default=0)
    p.add_argument(
        "--preview-max-width",
        type=int,
        default=None,
        help="Max OpenCV preview width (0 = no scaling; default from config pipeline.preview_max_width)",
    )
    p.add_argument(
        "--gui",
        action="store_true",
        help="Launcher with TOP and SIDE source selection",
    )
    return p.parse_args()


def _source_label(kind: str, args: argparse.Namespace) -> str:
    cap = getattr(args, f"{kind}_capture")
    if cap == "usb":
        idx = getattr(args, f"{kind}_camera_index")
        if idx is None:
            return f"{kind.upper()} | USB (default)"
        return f"{kind.upper()} | USB cam={idx}"
    if cap == "screen":
        mon = getattr(args, f"{kind}_screen_monitor")
        return f"{kind.upper()} | SCREEN mon={mon}"
    title = getattr(args, f"{kind}_window_title") or "(no title)"
    return f"{kind.upper()} | WINDOW {title[:28]}"


def run_vision_stack(cfg: AppConfig, args: argparse.Namespace) -> int:
    if args.no_control:
        cfg.control_enabled = False
    if args.no_windows:
        cfg.debug_windows = False
    if not cfg.apriltag.calibration_path:
        cfg.apriltag.intrinsics_from_frame_size = True
    cfg.controller.use_yaw = bool(cfg.stabilization.use_yaw)
    reference_tag_id = int(cfg.stabilization.top_reference_tag_id)
    top_camera_index = (
        args.top_camera_index
        if args.top_camera_index is not None
        else int(cfg.cameras.index_ceiling)
    )
    side_camera_index = (
        args.side_camera_index
        if args.side_camera_index is not None
        else int(cfg.cameras.index_side)
    )

    def top_factory():
        return create_video_source(
            args.top_capture,
            cfg.cameras,
            camera_index=top_camera_index,
            camera_role="top",
            screen_monitor=args.top_screen_monitor,
            window_title=args.top_window_title,
            window_match_index=args.top_window_match_index,
        )

    top_factory_ref = None if cfg.stabilization.demo_side_only else top_factory

    def side_factory():
        return create_video_source(
            args.side_capture,
            cfg.cameras,
            camera_index=side_camera_index,
            camera_role="side",
            screen_monitor=args.side_screen_monitor,
            window_title=args.side_window_title,
            window_match_index=args.side_window_match_index,
        )

    pipeline_cfg = PipelineConfig(
        control_hz=cfg.pipeline.control_hz,
        detection_hz_top=cfg.pipeline.detection_hz_top,
        detection_hz_side=cfg.pipeline.detection_hz_side,
        preview_hz=cfg.pipeline.preview_hz,
    )
    static_target = cfg.stabilization.target_pose_m
    planner = FutureTrajectoryPlanner(
        TrajectoryPoint(
            x_m=static_target.x_m,
            y_m=static_target.y_m,
            z_m=static_target.z_m,
            yaw_rad=static_target.yaw_rad,
        )
    )
    # --- Flight path -------------------------------------------------------
    # Absolute 3D points (X/Y/Z) in lab frame, or legacy relative steps.
    # TRAVEL starts only after the Travel button.
    nav = cfg.navigation
    flight_path = LoadedFlightPath()
    if nav.flight_path_path:
        try:
            flight_path = load_flight_path(
                nav.flight_path_path,
                default_square_side_m=float(nav.square_side_m),
                units_to_meter=float(nav.units_to_meter),
            )
            if flight_path.absolute:
                log.info(
                    "Loaded flight path: %d absolute points from %s (unit=%s)",
                    len(flight_path.waypoints),
                    nav.flight_path_path,
                    flight_path.coordinate_unit,
                )
            else:
                log.info(
                    "Loaded flight path: %d relative steps from %s (1 unit=%.3f m)",
                    len(flight_path.steps),
                    nav.flight_path_path,
                    nav.units_to_meter,
                )
        except (FileNotFoundError, ValueError) as e:
            log.error("Flight path: %s — TRAVEL inactive, HOLD only.", e)
    controller = DroneController(cfg.controller, static_target, cfg.stabilization, cfg.safety)
    director = FlightDirector(
        controller,
        cfg.stabilization,
        cfg.climb,
        cfg.safety,
        nav_cfg=nav,
        planner=planner,
        flight_path=flight_path,
    )
    tello = TelloController(
        TelloConfig(
            mock=args.mock_tello,
            rc_send_hz=max(float(cfg.safety.rc_send_hz), float(cfg.pipeline.control_hz)),
        )
    )

    def _ensure_tello() -> bool:
        if tello.connected:
            return True
        try:
            tello.connect()
            log.info("Connected to Tello")
            return True
        except Exception as e:
            log.error("Tello: %s", e)
            return False

    if cfg.control_enabled and args.connect_tello:
        if not _ensure_tello():
            log.warning("No Tello at startup — START will retry.")

    def _odometry_sample() -> OdometrySample | None:
        if not tello.connected:
            return None
        vel = tello.get_body_velocity_m_s()
        snap = tello.get_snapshot()
        return OdometrySample(
            vx_m_s=vel.vx_m_s,
            vy_m_s=vel.vy_m_s,
            vz_m_s=vel.vz_m_s,
            valid=vel.valid,
            flying=snap.flying,
        )

    def on_control_tick(state: DroneState, top_obs, side_obs, dt: float) -> FlightHudSnapshot:
        now = time.perf_counter()
        telemetry = tello.get_snapshot()
        top_ref_seen = reference_tag_id in top_obs.markers_seen
        pipeline = on_control_tick._pipeline  # type: ignore[attr-defined]
        shared = pipeline.shared
        while True:
            try:
                action = shared.ui_events.get_nowait()
            except queue.Empty:
                break
            if not cfg.control_enabled:
                log.info("Control disabled — click %s ignored", action)
                continue
            if action == "takeoff":
                if _ensure_tello() and not tello.flying:
                    tello.takeoff()
                    director.start_flight(time.perf_counter())
                    log.info("GUI: takeoff -> climbing")
            elif action == "travel":
                if tello.connected and tello.flying:
                    if director.request_travel(state, time.perf_counter()):
                        log.info("GUI: TRAVEL -> path flight")
                    else:
                        log.info(
                            "GUI: TRAVEL skipped (phase=%s, valid=%s)", director.phase, state.valid
                        )
                else:
                    log.info("GUI: TRAVEL skipped — drone not flying")
            elif action == "land":
                if tello.connected and tello.flying:
                    director.request_land()
                    tello.send_rc_zero(force=True, hover_reason="land")
                    tello.land()
                    director.reset()
                    log.info("GUI: land")
            elif action == "emergency":
                if tello.connected:
                    tello.emergency_stop()
                    director.reset()
                    log.warning("GUI: emergency stop")
        preview_mode = False
        if telemetry.flying and cfg.control_enabled and tello.connected:
            height_m = tello.get_height_m()
            cmd = director.update(state, height_m, dt, now)
            tello.send_rc(cmd, hover_reason=director.reason)
        elif cfg.control_enabled and state.valid and not telemetry.flying:
            # GROUND CALIBRATION: compute command relative to target (frame center),
            # without sending to drone — only to refresh preview.
            preview_mode = True
            controller.set_target(static_target)
            controller.compute(
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
                side_vz_m_s=state.side_vz_m_s,
                top_u_px=state.top_u_px,
                top_v_px=state.top_v_px,
                top_vu_px_s=state.top_vu_px_s,
                top_vv_px_s=state.top_vv_px_s,
            )
        telemetry = tello.get_snapshot()
        debug = controller.last_debug
        travel_progress = ""
        if director.is_traveling and director.travel_waypoints:
            travel_progress = f"{director.travel_done + 1}/{len(director.travel_waypoints)}"
        debug_lines: list[str] = []
        if cfg.gui.show_pid_debug:
            tag = "PID (preview)" if preview_mode else "PID"
            tgt = debug.target
            debug_lines.append(
                f"{tag} lr={debug.left_right.output:+.0f} fb={debug.forward_back.output:+.0f} "
                f"ud={debug.up_down.output:+.0f} yaw={debug.yaw.output:+.0f}"
            )
            debug_lines.append(
                f"err x={tgt.x_m - state.x_m:+.2f} y={tgt.y_m - state.y_m:+.2f} "
                f"z={tgt.z_m - state.z_m:+.2f} | TOP={state.top_state} SIDE={state.side_state} "
                f"axes={'/'.join(state.active_axes) or '-'}"
            )
            if getattr(cfg.stabilization, "top_image_hold", False):
                debug_lines.append(
                    f"TOP img u={debug.top_u.output:+.1f} v={debug.top_v.output:+.1f} "
                    f"blend={cfg.stabilization.top_image_hold_blend:.2f}"
                )
            if cfg.stabilization.side_image_hold:
                debug_lines.append(
                    f"SIDE img u={debug.side_u.output:+.1f} area={debug.side_depth.output:+.1f} "
                    f"blend={cfg.stabilization.side_image_hold_blend:.2f}/"
                    f"{cfg.stabilization.side_image_area_blend:.2f}"
                )
            if state.odom_fusion_active:
                odom_mode = "d-term" if not cfg.tracking.odometry.affect_filter else "filter"
                debug_lines.append(
                    f"ODOM {odom_mode} vx={state.odom_vx_m_s:+.2f} vy={state.odom_vy_m_s:+.2f} "
                    f"vz={state.odom_vz_m_s:+.2f} m/s"
                )
            trim_lr = int(getattr(cfg.stabilization, "side_lateral_trim_rc", 0) or 0)
            trim_fb = int(getattr(cfg.stabilization, "side_forward_trim_rc", 0) or 0)
            if trim_lr != 0 or trim_fb != 0:
                debug_lines.append(f"SIDE trim lr={trim_lr:+d} fb={trim_fb:+d}")
            if getattr(cfg.stabilization, "hold_creep_enabled", False):
                creep_vx = float(getattr(cfg.stabilization, "hold_creep_vx_m_s", 0.0) or 0.0)
                creep_vy = float(getattr(cfg.stabilization, "hold_creep_vy_m_s", 0.0) or 0.0)
                creep_vz = float(getattr(cfg.stabilization, "hold_creep_vz_m_s", 0.0) or 0.0)
                creep_gain = float(getattr(cfg.stabilization, "hold_creep_rc_per_m_s", 90.0) or 90.0)
                layout = (
                    getattr(cfg.stabilization, "side_camera_layout", "face_to_face") or "face_to_face"
                ).strip().lower()
                debug_lines.append(
                    f"HOLD creep lab=({creep_vx:+.2f},{creep_vy:+.2f},{creep_vz:+.2f}) m/s "
                    f"gain={creep_gain:.0f} SIDE={layout}"
                )
            depth_axis = getattr(cfg.stabilization, "side_depth_rc_axis", "pitch")
            debug_lines.append(f"SIDE depth -> {depth_axis}")
        _publish_trajectory_overlay(shared, director, state, top_obs, pipeline.top_detection)

        def _wh(slot_w: int, slot_h: int) -> str:
            return f"{slot_w}×{slot_h}" if slot_w > 0 and slot_h > 0 else ""

        top_cap = _wh(shared.frames["top"].width, shared.frames["top"].height)
        side_cap = _wh(shared.frames["side"].width, shared.frames["side"].height)
        top_det = ""
        side_det = ""
        if pipeline.top_detection is not None:
            td = pipeline.top_detection
            top_det = _wh(td.last_input_width, td.last_input_height)
        if pipeline.side_detection is not None:
            sd = pipeline.side_detection
            side_det = _wh(sd.last_input_width, sd.last_input_height)

        return FlightHudSnapshot(
            phase=director.phase,
            phase_reason=director.reason,
            preview_mode=preview_mode,
            tello_connected=telemetry.connected,
            tello_flying=telemetry.flying,
            battery_pct=telemetry.battery,
            top_state=state.top_state,
            top_tag0=top_ref_seen,
            top_tags=tuple(top_obs.markers_seen),
            top_quality=float(top_obs.pose_quality),
            side_state=state.side_state,
            side_tags=tuple(side_obs.markers_seen),
            side_active_tag=side_obs.marker_id,
            side_quality=float(side_obs.pose_quality),
            tracking_valid=state.valid,
            active_axes="/".join(state.active_axes) or "-",
            pose_x_m=state.x_m,
            pose_y_m=state.y_m,
            pose_z_m=state.z_m,
            travel_progress=travel_progress,
            geofence_active=director.geofence_enabled and director.geofence_bounds is not None,
            geofence_ok=not director.geofence_breached,
            alert=_short_alert(
                state,
                top_ref_seen,
                top_obs,
                side_obs,
                reference_tag_id=reference_tag_id,
                geofence_breached=director.geofence_breached,
            ),
            top_capture=top_cap,
            side_capture=side_cap,
            top_detection=top_det,
            side_detection=side_det,
            debug_lines=debug_lines,
        )

    try:
        pipeline = build_vision_pipeline(
            cfg,
            pipeline_cfg,
            top_factory_ref,
            side_factory,
            on_control_tick,
            preview_enabled=cfg.debug_windows,
            top_label=(
                "TOP | disabled" if cfg.stabilization.demo_side_only else _source_label("top", args)
            ),
            side_label=_source_label("side", args),
            preview_max_width=(
                int(args.preview_max_width)
                if args.preview_max_width is not None
                else int(cfg.pipeline.preview_max_width)
            ),
            odometry_provider=_odometry_sample,
        )
    except Exception as e:
        log.error("TOP/SIDE sources: %s", e)
        return 1
    on_control_tick._pipeline = pipeline  # type: ignore[attr-defined]
    if cfg.control_enabled:
        log.info(
            "Pipeline: control=%.0f Hz det_top=%.0f Hz det_side=%.0f Hz | demo_side_only=%s | Q quits",
            pipeline_cfg.control_hz,
            pipeline_cfg.detection_hz_top,
            pipeline_cfg.detection_hz_side,
            cfg.stabilization.demo_side_only,
        )
    else:
        log.info("Detection pipeline — Q quits | no Tello")
    try:
        pipeline.run()
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        if tello.connected:
            tello.send_rc_zero(force=True, hover_reason="shutdown")
            if tello.flying:
                try:
                    tello.land()
                except Exception:
                    pass
        tello.disconnect()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    if args.gui:
        from device_bar import run_device_bar

        return run_device_bar()
    cfg: AppConfig = load_config(args.config)
    if args.no_windows:
        cfg.debug_windows = False
    return run_vision_stack(cfg, args)


# Backward-compatible alias
run_aruco_stack = run_vision_stack
if __name__ == "__main__":
    sys.exit(main())
