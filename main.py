"""Tellcam — nawigacja AprilTag tag36h11 (TOP + SIDE) dla Ryze Tello."""
from __future__ import annotations

import argparse
import logging
import math
import queue
import sys
import time
import cv2

from config import AppConfig, load_config
from controller import DroneController
from flight_director import FlightDirector
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
    # --- Ścieżka lotu (scaffolding) -----------------------------------------
    # Domyślnie HOLD: dron trzyma jeden punkt (TOP+SIDE). Gdy w configu podano
    # plik ścieżki, wczytujemy go i przygotowujemy waypointy w planerze. Samo
    # lecenie po ścieżce nie jest jeszcze podpięte do FlightDirectora (HOLD nadal
    # rządzi) — to miejsce na przyszłe rozszerzenie nawigacji po waypointach.
    nav = cfg.navigation
    if nav.flight_path_path:
        try:
            waypoints = planner.load_path_from_file(
                planner.peek_target(), nav.flight_path_path, nav.units_to_meter
            )
            log.info(
                "Wczytano ścieżkę lotu: %d waypointów z %s (mode=%s, 1 jednostka=%.2f m)",
                len(waypoints), nav.flight_path_path, nav.mode, nav.units_to_meter,
            )
            if nav.mode.strip().lower() != "path":
                log.info("navigation.mode=%s -> lot pozostaje w trybie HOLD (ścieżka tylko wczytana).", nav.mode)
        except (FileNotFoundError, ValueError) as e:
            log.error("Ścieżka lotu: %s — pozostaję w trybie HOLD.", e)
    controller = DroneController(cfg.controller, static_target, cfg.stabilization, cfg.safety)
    director = FlightDirector(controller, cfg.stabilization, cfg.climb, cfg.safety)
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
            if not cfg.control_enabled and action != "connect":
                log.info("Sterowanie wyłączone — klik %s pominięty", action)
                continue
            if action == "connect":
                if _ensure_tello():
                    log.info("GUI: połączono z Tello")
            elif action == "takeoff":
                if _ensure_tello() and not tello.flying:
                    tello.takeoff()
                    director.start_flight(time.perf_counter())
                    log.info("GUI: takeoff -> wznoszenie")
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
            # KALIBRACJA NA ZIEMI: licz komendę względem celu (środek kadru),
            # bez wysyłania do drona — tylko po to, by odświeżyć podgląd.
            preview_mode = True
            controller.set_target(static_target)
            controller.compute(
                state.x_m, state.y_m, state.z_m, state.yaw_rad, dt,
                active_axes=state.active_axes, freeze_axes=state.freeze_axes,
                vx=state.vx_m_s, vy=state.vy_m_s, vz=state.vz_m_s,
                heading_rad=state.heading_rad, heading_valid=state.heading_valid,
            )

        telemetry = tello.get_snapshot()
        tgt = controller.last_debug.target
        status_lines = [
            (
                f"phase={director.phase} ({director.reason})"
            ),
            (
                f"track top={state.top_state} side={state.side_state} valid={state.valid} "
                f"axes={'/'.join(state.active_axes) or '-'} ref_tag={reference_tag_id} "
                f"top_ref={'YES' if top_ref_seen else 'NO'}"
            ),
            (
                f"pose x={state.x_m:+.2f}m y={state.y_m:+.2f}m "
                f"z={state.z_m:.2f}m yaw={state.yaw_rad:+.2f}rad age={state.pose_age_s:.2f}s"
            ),
            (
                f"vel vx={state.vx_m_s:+.2f} vy={state.vy_m_s:+.2f} vz={state.vz_m_s:+.2f} m/s (predykcja)"
            ),
            (
                f"target x={tgt.x_m:+.2f}m y={tgt.y_m:+.2f}m "
                f"z={tgt.z_m:.2f}m yaw={tgt.yaw_rad:+.2f}rad"
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
        yaw_on = "yaw" in state.active_axes
        status_lines.append(
            f"yaw src={cfg.stabilization.yaw_source} active={'YES' if yaw_on else 'NO'} "
            f"heading={math.degrees(state.heading_rad):+.0f}deg hv={'YES' if state.heading_valid else 'NO'} "
            f"target={math.degrees(debug.target.yaw_rad):+.0f}deg yaw_sign={cfg.controller.yaw_sign:+.0f}"
        )
        status_lines.append(
            f"side_hdg={math.degrees(state.side_heading_rad):+.0f}deg "
            f"sv={'YES' if state.side_heading_valid else 'NO'} "
            f"side_tag={side_obs.marker_id if side_obs.marker_id is not None else '-'} "
            f"facing={math.degrees(side_obs.yaw_facing_rad):+.0f}deg"
        )
        status_lines.append(
            f"signs roll={cfg.controller.roll_sign:+.0f} pitch={cfg.controller.pitch_sign:+.0f} "
            f"thr={cfg.controller.throttle_sign:+.0f} | body_frame={cfg.stabilization.body_frame_control}"
        )
        if cfg.gui.show_pid_debug:
            tag = "PID PODGLAD(nie wysylane)" if preview_mode else "pid"
            status_lines.append(
                f"{tag} lr={debug.left_right.output:+.1f}(e={debug.left_right.error:+.2f}) "
                f"fb={debug.forward_back.output:+.1f}(e={debug.forward_back.error:+.2f}) "
                f"ud={debug.up_down.output:+.1f}(e={debug.up_down.error:+.2f}) "
                f"yaw={debug.yaw.output:+.1f}(e={debug.yaw.error:+.2f})"
            )
        if preview_mode:
            status_lines.append(
                "KALIBRACJA (dron w rece, w kadrze TOP). RC+ = lr:PRAWO fb:PRZOD ud:GORA yaw:CW."
            )
            status_lines.append(
                "Przesun drona w dana strone -> komenda ma byc PRZECIWNA (wraca do srodka). "
                "Jesli ZGODNA (ucieka) -> odwroc znak osi: roll/pitch/throttle/yaw_sign."
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
