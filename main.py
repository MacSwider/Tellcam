"""Tellcam — nawigacja ArUco (TOP + SIDE) dla Ryze Tello."""
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
from tello_interface import TelloConfig, TelloInterface
from video_source import create_video_source
from vision_pipeline import PipelineConfig, build_vision_pipeline

log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tellcam — nawigacja ArUco TOP/SIDE + (opcjonalnie) Tello")
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


def run_aruco_stack(cfg: AppConfig, args: argparse.Namespace) -> int:
    if args.no_control:
        cfg.control_enabled = False
    if args.no_windows:
        cfg.debug_windows = False
    if not cfg.aruco.calibration_path:
        cfg.aruco.intrinsics_from_frame_size = True

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

    controller = DroneController(cfg.controller, cfg.target)
    tello = TelloInterface(TelloConfig(mock=args.mock_tello))
    lost_frames = 0
    failsafe_active = False
    flight_active = False

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
        nonlocal lost_frames, failsafe_active, flight_active

        status_lines = [
            f"valid={state.valid} top={top_obs.ok} side={side_obs.ok}",
            f"top_ids={list(top_obs.markers_seen)} side_ids={list(side_obs.markers_seen)}",
            f"x={state.x_m:.2f} y={state.y_m:.2f} z={state.z_m:.2f}",
            f"yaw={state.yaw_rad:.2f} pitch={state.pitch_rad:.2f} roll={state.roll_rad:.2f}",
            f"failsafe={failsafe_active} lost={lost_frames}",
            f"control={cfg.control_enabled} tello={'OK' if tello.connected else '—'}",
        ]
        if top_obs.hint:
            status_lines.append(f"top_hint={top_obs.hint}")
        if side_obs.hint:
            status_lines.append(f"side_hint={side_obs.hint}")
        if cfg.control_enabled:
            status_lines.append(f"flight={'ON' if flight_active else 'OFF'} (START/LAND)")
        else:
            status_lines.append("tryb detekcji (bez sterowania dronem)")

        pipeline = on_control_tick._pipeline  # type: ignore[attr-defined]
        shared = pipeline.shared
        while True:
            try:
                cx, cy = shared.mouse_clicks.get_nowait()
            except queue.Empty:
                break
            x0, y0, x1, y1 = shared.get_button_rect()
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                if not cfg.control_enabled:
                    log.info("START/LAND wyłączone (tryb detekcji)")
                elif not _ensure_tello():
                    log.error("START/LAND: brak połączenia z Tello")
                elif not flight_active:
                    tello.takeoff()
                    flight_active = True
                    controller.reset()
                    log.info("START — takeoff")
                else:
                    tello.send_rc_zero()
                    tello.land()
                    flight_active = False
                    controller.reset()
                    log.info("LAND — lądowanie")

        if state.valid:
            lost_frames = 0
            if failsafe_active:
                failsafe_active = False
                controller.reset()
                log.info("Detekcja przywrócona — wznowienie PID")
            if flight_active and cfg.control_enabled and tello.connected:
                cmd = controller.compute(state.x_m, state.y_m, state.z_m, state.yaw_rad, dt)
                tello.send_rc(cmd)
                status_lines.append(f"RC r={cmd.roll} p={cmd.pitch} thr={cmd.throttle} y={cmd.yaw}")
        else:
            lost_frames += 1
            if not failsafe_active:
                log.warning("Utrata pełnej detekcji — RC=0")
                failsafe_active = True
            controller.reset()
            if (
                flight_active
                and cfg.failsafe.on_lost_send_zero_rc
                and cfg.control_enabled
                and tello.connected
            ):
                tello.send_rc_zero()
            if lost_frames >= cfg.failsafe.max_lost_frames and lost_frames % 30 == 0:
                log.warning("Brak detekcji przez %s klatek", lost_frames)

        _, _, top_ts, side_ts = shared.read_poses()
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
            tello.send_rc_zero()
            if flight_active:
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

    return run_aruco_stack(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
