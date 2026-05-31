"""Tellcam — nawigacja AprilTag tag36h11 (TOP + SIDE) dla Ryze Tello."""
from __future__ import annotations

import argparse
import logging
import queue
import sys
import time
import cv2

from config import AppConfig, load_config
from controller import DroneController
from state_estimator import DroneState
from tello_interface import TelloConfig, TelloController
from trajectory_planner import FutureTrajectoryPlanner, TrajectoryPoint
from video_source import create_video_source
from vision_pipeline import PipelineConfig, build_vision_pipeline

log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tellcam — nawigacja AprilTag TOP/SIDE + (opcjonalnie) Tello")
    p.add_argument("--config", type=str, default=None, help="JSON z nadpisaniami configu")
    p.add_argument("--no-control", action="store_true", help="Wyłącz RC do drona (tylko detekcja / podgląd)")
    p.add_argument(
        "--connect-tello",
        action="store_true",
        help="Połącz z Tello już przy starcie (domyślnie: tylko po kliknięciu START)",
    )
    p.add_argument("--mock-tello", action="store_true", help="Symulacja Tello (przy START, bez fizycznego drona)")
    p.add_argument("--no-windows", action="store_true", help="Bez okien OpenCV")
    for prefix in ("top", "side"):
        p.add_argument(f"--{prefix}-capture", type=str, default="usb", choices=("usb", "screen", "window"))
        p.add_argument(f"--{prefix}-camera-index", type=int, default=None)
        p.add_argument(f"--{prefix}-screen-monitor", type=int, default=1)
        p.add_argument(f"--{prefix}-window-title", type=str, default="")
        p.add_argument(f"--{prefix}-window-match-index", type=int, default=0)
    p.add_argument(
        "--preview-max-width",
        type=int,
        default=1400,
        help="Maks. szerokość podglądu OpenCV (0 = bez skalowania)",
    )
    p.add_argument(
        "--gui",
        action="store_true",
        help="Launcher z wyborem źródeł TOP i SIDE",
    )
    return p.parse_args()


def _source_label(kind: str, args: argparse.Namespace) -> str:
    cap = getattr(args, f"{kind}_capture")
    if cap == "usb":
        idx = getattr(args, f"{kind}_camera_index")
        if idx is None:
            return f"{kind.upper()} | USB (domyslna)"
        return f"{kind.upper()} | USB cam={idx}"
    if cap == "screen":
        mon = getattr(args, f"{kind}_screen_monitor")
        return f"{kind.upper()} | SCREEN mon={mon}"
    title = getattr(args, f"{kind}_window_title") or "(brak tytulu)"
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

    top_camera_index = args.top_camera_index if args.top_camera_index is not None else int(cfg.cameras.index_ceiling)
    side_camera_index = args.side_camera_index if args.side_camera_index is not None else int(cfg.cameras.index_side)

    def top_factory():
        return create_video_source(
            args.top_capture,
            cfg.cameras,
            camera_index=top_camera_index,
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
    controller = DroneController(cfg.controller, static_target, cfg.stabilization, cfg.safety)
    tello = TelloController(
        TelloConfig(
            mock=args.mock_tello,
            rc_send_hz=cfg.safety.rc_send_hz,
        )
    )

    def _ensure_tello() -> bool:
        if tello.connected:
            return True
        try:
            tello.connect()
            log.info("Połączono z Tello")
            return True
        except Exception as e:
            log.error("Tello: %s", e)
            return False

    if cfg.control_enabled and args.connect_tello:
        if not _ensure_tello():
            log.warning("Brak Tello przy starcie — START spróbuje ponownie.")

    def on_control_tick(state: DroneState, top_obs, side_obs, dt: float) -> list[str]:
        telemetry = tello.get_snapshot()
        planner_target = planner.peek_target()
        top_ref_seen = reference_tag_id in top_obs.markers_seen
        controller.set_target(
            type(static_target)(
                x_m=planner_target.x_m,
                y_m=planner_target.y_m,
                z_m=planner_target.z_m,
                yaw_rad=planner_target.yaw_rad,
            )
        )

        pipeline = on_control_tick._pipeline  # type: ignore[attr-defined]
        shared = pipeline.shared
        while True:
            try:
                action = shared.ui_events.get_nowait()
            except queue.Empty:
                break
            if not cfg.control_enabled and action != "connect":
                log.info("Sterowanie wyłączone — klik %s pominięty", action)
                continue
            if action == "connect":
                if _ensure_tello():
                    log.info("GUI: połączono z Tello")
            elif action == "takeoff":
                if _ensure_tello() and not tello.flying:
                    tello.takeoff()
                    controller.reset()
                    log.info("GUI: takeoff")
            elif action == "land":
                if tello.connected and tello.flying:
                    tello.send_rc_zero(force=True, hover_reason="land")
                    tello.land()
                    controller.reset()
                    log.info("GUI: land")
            elif action == "emergency":
                if tello.connected:
                    tello.emergency_stop()
                    controller.reset()
                    log.warning("GUI: emergency stop")

        if state.valid and telemetry.flying and cfg.control_enabled and tello.connected:
            cmd = controller.compute(
                state.x_m,
                state.y_m,
                state.z_m,
                state.yaw_rad,
                dt,
                freeze_integrators=(state.tracking_state != "TRACKING"),
            )
            tello.send_rc(cmd)
        elif telemetry.flying and cfg.control_enabled and tello.connected and cfg.safety.zero_rc_on_loss:
            tello.send_rc_zero(hover_reason=f"tracking_{state.tracking_state.lower()}")
            if state.tracking_state in {"LOST", "TIMED_OUT"}:
                controller.reset()

        telemetry = tello.get_snapshot()
        status_lines = [
            (
                f"tracking={state.tracking_state} valid={state.valid} "
                f"ref_tag={reference_tag_id} top_ref={'YES' if top_ref_seen else 'NO'}"
            ),
            (
                f"pose x={state.x_m:+.2f}m y={state.y_m:+.2f}m "
                f"z={state.z_m:.2f}m yaw={state.yaw_rad:+.2f}rad age={state.pose_age_s:.2f}s"
            ),
            (
                f"target x={planner_target.x_m:+.2f}m y={planner_target.y_m:+.2f}m "
                f"z={planner_target.z_m:.2f}m yaw={planner_target.yaw_rad:+.2f}rad"
            ),
            (
                f"tello={'OK' if telemetry.connected else 'OFF'} flight={'ON' if telemetry.flying else 'OFF'} "
                f"battery={telemetry.battery if telemetry.battery is not None else '--'}%"
            ),
            (
                f"rc lr={telemetry.current_rc.left_right:+d} fb={telemetry.current_rc.forward_back:+d} "
                f"ud={telemetry.current_rc.up_down:+d} yaw={telemetry.current_rc.yaw:+d}"
            ),
            (
                f"top_tag_ids={list(top_obs.markers_seen)} q={top_obs.pose_quality:.2f} "
                f"side_tag_ids={list(side_obs.markers_seen)} q={side_obs.pose_quality:.2f}"
            ),
            f"control={cfg.control_enabled} frame={cfg.stabilization.control_frame} side_only={cfg.stabilization.demo_side_only}",
        ]

        debug = controller.last_debug
        if cfg.gui.show_pid_debug:
            if cfg.stabilization.control_frame.strip().lower() == "top":
                pid_line = (
                    f"pid lr={debug.lateral.output:+.1f}({debug.lateral.error:+.2f}) "
                    f"fb={debug.vertical.output:+.1f}({debug.vertical.error:+.2f}) "
                    f"ud={debug.distance.output:+.1f}({debug.distance.error:+.2f}) "
                    f"yaw={debug.yaw.output:+.1f}({debug.yaw.error:+.2f})"
                )
            else:
                pid_line = (
                    f"pid lr={debug.lateral.output:+.1f}({debug.lateral.error:+.2f}) "
                    f"ud={debug.vertical.output:+.1f}({debug.vertical.error:+.2f}) "
                    f"fb={debug.distance.output:+.1f}({debug.distance.error:+.2f}) "
                    f"yaw={debug.yaw.output:+.1f}({debug.yaw.error:+.2f})"
                )
            status_lines.append(
                pid_line
            )
        if state.hint:
            status_lines.append(f"tracking_hint={state.hint}")
        if top_obs.hint and not cfg.stabilization.demo_side_only:
            status_lines.append(f"top_hint={top_obs.hint}")
        if side_obs.hint:
            status_lines.append(f"side_hint={side_obs.hint}")

        _, _, top_ts, side_ts = shared.read_poses()
        now = time.perf_counter()
        top_pose_age = now - top_ts if top_ts else 0.0
        side_pose_age = now - side_ts if side_ts else 0.0
        status_lines.append(
            f"pose_age top={top_pose_age:.2f}s side={side_pose_age:.2f}s "
            f"det top={pipeline.top_detection.detection_hz if pipeline.top_detection else 0.0:.1f}Hz "
            f"side={pipeline.side_detection.detection_hz if pipeline.side_detection else 0.0:.1f}Hz"
        )
        if pipeline.top_camera:
            status_lines.append(f"cam_top={pipeline.top_camera.read_hz:.1f}Hz")
        if pipeline.side_camera:
            status_lines.append(f"cam_side={pipeline.side_camera.read_hz:.1f}Hz")
        return status_lines

    try:
        pipeline = build_vision_pipeline(
            cfg,
            pipeline_cfg,
            top_factory_ref,
            side_factory,
            on_control_tick,
            preview_enabled=cfg.debug_windows,
            top_label="TOP | disabled" if cfg.stabilization.demo_side_only else _source_label("top", args),
            side_label=_source_label("side", args),
            preview_max_width=args.preview_max_width,
        )
    except Exception as e:
        log.error("Źródła TOP/SIDE: %s", e)
        return 1

    on_control_tick._pipeline = pipeline  # type: ignore[attr-defined]

    if cfg.control_enabled:
        log.info(
            "Pipeline: control=%.0f Hz det_top=%.0f Hz det_side=%.0f Hz | demo_side_only=%s | Q kończy",
            pipeline_cfg.control_hz,
            pipeline_cfg.detection_hz_top,
            pipeline_cfg.detection_hz_side,
            cfg.stabilization.demo_side_only,
        )
    else:
        log.info("Pipeline detekcji — Q kończy | bez Tello")

    try:
        pipeline.run()
    except KeyboardInterrupt:
        log.info("Przerwano przez użytkownika")
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


# Alias wsteczny
run_aruco_stack = run_vision_stack


if __name__ == "__main__":
    sys.exit(main())
