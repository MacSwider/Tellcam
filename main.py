"""Tellcam — nawigacja AprilTag tag36h11 (TOP + SIDE) dla Ryze Tello."""
from __future__ import annotations

import argparse
import dataclasses
import logging
import queue
import sys
import time
import cv2

from config import AppConfig, load_config
from controller import DroneController
from flight_mission import AIRBORNE_PHASES, FlightMission
from flight_path import load_path
from position_filter import PositionFilter, tello_velocity_to_camera
from state_estimator import DroneState
from tello_interface import TelloConfig, TelloInterface
from video_source import create_video_source
from vision_pipeline import FlightUiState, PipelineConfig, build_vision_pipeline

log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tellcam — nawigacja AprilTag TOP/SIDE + (opcjonalnie) Tello")
    p.add_argument("--config", type=str, default=None, help="JSON z nadpisaniami configu")
    p.add_argument("--no-control", action="store_true", help="Wyłącz RC do drona (tylko detekcja / podgląd)")
    p.add_argument("--control", action="store_true", help="Włącz sterowanie Tello (przyciski START/LAND)")
    p.add_argument("--path", type=str, default=None, help="Plik JSON z trasą (domyślnie flight.path_file)")
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
    if args.control:
        cfg.control_enabled = True
    if args.no_control:
        cfg.control_enabled = False
    if args.no_windows:
        cfg.debug_windows = False
    if not cfg.apriltag.calibration_path:
        cfg.apriltag.intrinsics_from_frame_size = True

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

    path_file = args.path or cfg.flight.path_file
    try:
        flight_path = load_path(path_file)
    except Exception as e:
        log.error("Trasa %s: %s", path_file, e)
        return 1
    log.info(
        "Trasa „%s”: %d punktów głównych, %d z pomocniczymi",
        flight_path.name,
        len(flight_path.main_points),
        len(flight_path.points),
    )

    controller = DroneController(cfg.controller)
    tello = TelloInterface(TelloConfig(mock=args.mock_tello))
    mission = FlightMission(cfg.flight, tello, controller, flight_path)
    path_ui_points = tuple((p.xyz, p.is_main) for p in flight_path.points)
    last_depth_m: float | None = None
    pos_filter = PositionFilter(cfg.filter)
    last_valid_state: DroneState | None = None

    def _ensure_tello() -> bool:
        if tello.connected:
            return True
        try:
            tello.connect()
            log.info("Połączono z Tello")
            return True
        except Exception as e:
            log.error("Tello: %s: %s — sprawdź Wi-Fi TELLO-* i czy dron jest włączony", type(e).__name__, e)
            return False

    if cfg.control_enabled and args.connect_tello:
        if not _ensure_tello():
            log.warning("Brak Tello przy starcie — START spróbuje ponownie.")

    def on_control_tick(state: DroneState, top_obs, side_obs, dt: float) -> list[str]:
        nonlocal last_depth_m, last_valid_state

        pipeline = on_control_tick._pipeline  # type: ignore[attr-defined]
        shared = pipeline.shared
        _, _, top_ts, side_ts = shared.read_poses()
        while True:
            try:
                cmd = shared.ui_commands.get_nowait()
            except queue.Empty:
                break
            if not cfg.control_enabled:
                log.info("START/LAND wyłączone (tryb detekcji)")
            elif cmd == "takeoff":
                if _ensure_tello():
                    mission.request_takeoff()
                else:
                    log.error("START: brak połączenia z Tello")
            elif cmd == "land":
                mission.request_land()

        if state.valid:
            last_valid_state = state
            last_depth_m = state.z_m

        raw_vel = tello.velocity() if tello.connected else None
        filter_line = ""
        ctl_state, vel_xy = state, None
        if cfg.filter.enabled:
            vel_meas = None
            if cfg.filter.use_tello_velocity and raw_vel is not None and last_valid_state is not None:
                vel_meas = tello_velocity_to_camera(
                    raw_vel[0], raw_vel[1], last_valid_state.yaw_rad, cfg.filter, cfg.controller.heading_offset_rad
                )
            fs = pos_filter.step(dt, state.valid, (state.x_m, state.y_m), top_ts, vel_meas)
            if fs.valid and last_valid_state is not None:
                # Podczas predykcji (brak świeżej detekcji) yaw i odległość z ostatniej poprawnej pozy.
                ctl_state = dataclasses.replace(last_valid_state, valid=True, x_m=fs.x_m, y_m=fs.y_m)
                vel_xy = (fs.vx_mps, fs.vy_mps)
            else:
                ctl_state = dataclasses.replace(state, valid=False)
            filter_line = (
                f"filtr: x={fs.x_m:+.2f} y={fs.y_m:+.2f} vx={fs.vx_mps:+.2f} vy={fs.vy_mps:+.2f} m/s "
                f"predykcja={fs.predicted_s:.2f}s"
            )
        if raw_vel is not None:
            filter_line += f" | tello vgx={raw_vel[0]} vgy={raw_vel[1]}"

        mission_lines = mission.tick(ctl_state, dt, vel_xy) if cfg.control_enabled and tello.connected else []
        shared.set_flight_ui(
            FlightUiState(
                phase=mission.phase.value,
                airborne=mission.phase in AIRBORNE_PHASES,
                path_points=path_ui_points,
                target_index=mission.follower.index if mission.phase.value in ("FOLLOW", "FINISHED") else -1,
                drone_xy=(ctl_state.x_m, ctl_state.y_m) if ctl_state.valid else None,
                depth_m=last_depth_m,
            )
        )

        status_lines = [
            f"valid={state.valid} top={top_obs.ok} side={side_obs.ok} "
            f"top_ids={list(top_obs.markers_seen)} side_ids={list(side_obs.markers_seen)}",
            f"x={state.x_m:+.2f} y={state.y_m:+.2f} dist_cam={state.z_m:.2f} yaw={state.yaw_rad:+.2f}",
            f"control={cfg.control_enabled} tello={'OK' if tello.connected else '—'} trasa={flight_path.name}",
        ]
        if filter_line:
            status_lines.append(filter_line)
        status_lines += mission_lines
        if not cfg.control_enabled:
            status_lines.append("tryb detekcji (bez sterowania dronem)")
        if top_obs.hint:
            status_lines.append(f"top_hint={top_obs.hint}")
        if side_obs.hint:
            status_lines.append(f"side_hint={side_obs.hint}")

        now = time.perf_counter()
        status_lines.append(
            f"pose_age top={now - top_ts:.2f}s side={now - side_ts:.2f}s "
            f"det top={pipeline.top_detection.detection_hz:.1f}Hz "
            f"side={pipeline.side_detection.detection_hz:.1f}Hz"
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
            top_factory,
            side_factory,
            on_control_tick,
            preview_enabled=cfg.debug_windows,
            top_label=_source_label("top", args),
            side_label=_source_label("side", args),
            preview_max_width=args.preview_max_width,
        )
    except Exception as e:
        log.error("Źródła TOP/SIDE: %s", e)
        return 1

    on_control_tick._pipeline = pipeline  # type: ignore[attr-defined]

    if cfg.control_enabled:
        log.info(
            "Pipeline: control=%.0f Hz det_top=%.0f Hz det_side=%.0f Hz | Q kończy",
            pipeline_cfg.control_hz,
            pipeline_cfg.detection_hz_top,
            pipeline_cfg.detection_hz_side,
        )
    else:
        log.info("Pipeline detekcji — Q kończy | bez Tello")

    try:
        pipeline.run()
    except KeyboardInterrupt:
        log.info("Przerwano przez użytkownika")
    finally:
        if tello.connected:
            mission.shutdown()
            tello.send_rc_zero()
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
