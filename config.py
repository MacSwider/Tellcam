"""
Centralna konfiguracja Tellcam: kamery, pipeline, AprilTag, PID, failsafe.
Nadpisania: plik JSON (tellcam_config.json) lub TELLCAM_CONFIG.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class CameraConfig:
    index_ceiling: int = 0
    index_side: int = 1
    width: int = 1920
    height: int = 1080
    # MJPG trzeba ustawić już przy otwarciu (DSHOW ignoruje późniejszą zmianę); YUY2 przy 720p daje ~10 fps.
    fourcc: str = "MJPG"
    fps: int = 30
    # Ekspozycja DirectShow (2^x s): -5 ≈ 1/32 s. None = automatyczna (u Creative ~1/16 s → max 16 fps).
    exposure: float | None = -5.0
    buffer_size: int = 1
    drain_frames_on_read: int = 2


@dataclass
class AprilTagConfig:
    family: str = "tag36h11"
    marker_ids: Tuple[int, ...] = (0, 1, 2, 3, 4)
    marker_length_m: float = 0.04  # bok czarnego kwadratu tagu [m], bez białej ramki
    layout_half_forward_m: float = 0.07
    layout_half_lateral_m: float = 0.07
    marker_side_height_m: float = 0.03
    calibration_path: str | None = None
    intrinsics_from_frame_size: bool = False
    use_clahe: bool = True
    quad_decimate: float = 1.0
    quad_sigma: float = 0.0
    refine_edges: bool = True
    decode_sharpening: float = 0.5
    min_decision_margin: float = 5.0
    # Tylko dla naklejek bez białej ramki; bez tagu w kadrze kosztuje ~5x więcej czasu.
    use_black_crop_fallback: bool = False
    black_crop_thresholds: Tuple[int, ...] = (50, 60, 70, 80, 90)
    black_crop_pad_frac: float = 0.35
    black_crop_upscale: int = 2
    max_detection_width: int = 0
    max_markers_per_frame: int = 5


@dataclass
class PIDGains:
    """Wejście: błąd w metrach (lub radianach dla yaw), wyjście: wartość RC Tello (-100..100)."""
    kp: float = 60.0
    ki: float = 4.0
    kd: float = 15.0
    output_limit: float = 30.0
    integral_limit: float = 10.0


@dataclass
class ControllerConfig:
    pid_x: PIDGains = field(default_factory=PIDGains)
    pid_y: PIDGains = field(default_factory=PIDGains)
    pid_z: PIDGains = field(default_factory=lambda: PIDGains(kp=80.0, ki=5.0, kd=10.0, output_limit=40.0))
    pid_yaw: PIDGains = field(
        default_factory=lambda: PIDGains(kp=50.0, ki=0.0, kd=5.0, output_limit=30.0)
    )
    deadzone_m: float = 0.03
    deadzone_yaw_rad: float = 0.05
    # Błąd XY z kamery TOP obracany do układu drona (przód/prawo) wg zmierzonego yaw.
    body_frame: bool = True
    # Kąt między osią X układu tagów a faktycznym przodem drona.
    heading_offset_rad: float = 0.0
    sign_roll: int = 1
    sign_pitch: int = 1
    sign_throttle: int = 1
    sign_yaw: int = 1


@dataclass
class TargetConfig:
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 1.0
    yaw_rad: float = 0.0


@dataclass
class FailsafeConfig:
    max_lost_frames: int = 15
    on_lost_send_zero_rc: bool = True


@dataclass
class FilterConfig:
    enabled: bool = True
    position_sigma_m: float = 0.03  # szum pozycji z kamery TOP
    accel_sigma: float = 1.5  # jak gwałtownie może zmieniać się prędkość [m/s²]
    # Ile sekund filtr przewiduje pozycję bez detekcji, zanim uzna tracking za utracony.
    max_predict_s: float = 0.4
    # Pomiar dalej niż gate_m od przewidywanej pozycji = prawdopodobnie fałszywa detekcja → odrzucony.
    # Po gate_reset_count takich pomiarach z rzędu filtr przyjmuje nową pozycję (dron faktycznie tam jest).
    gate_m: float = 0.25
    gate_reset_count: int = 3
    # Ile zgodnych detekcji z rzędu potrzeba po przerwie, zanim pozycja trafi do sterowania.
    confirm_count: int = 3
    # Prędkość z telemetrii Tello — włączyć dopiero po sprawdzeniu znaków osi w locie.
    use_tello_velocity: bool = False
    tello_velocity_scale: float = 0.1  # jednostka vgx/vgy → m/s (dm/s)
    tello_velocity_sigma_mps: float = 0.1
    tello_vel_sign_x: int = 1
    tello_vel_sign_y: int = 1


@dataclass
class FlightConfig:
    path_file: str = "paths/square.json"
    takeoff_altitude_m: float = 1.0
    altitude_tolerance_m: float = 0.08
    climb_timeout_s: float = 12.0
    hover_before_path_s: float = 5.0
    # "tello" = wysokościomierz drona, "vision" = camera_height_m - odległość z kamery TOP
    altitude_source: str = "tello"
    camera_height_m: float = 2.5
    hold_yaw: bool = True
    # "tello" = kurs z IMU drona (pewny znak), "camera" = obrót tagu z kamery TOP
    yaw_source: str = "tello"
    # 0 = wisi bez limitu czasu do odzyskania trackingu
    lost_tracking_land_after_s: float = 0.0
    # Kalibracja w locie: krótki ruch do przodu i w prawo, kamera TOP mierzy, gdzie dron faktycznie poleciał.
    calibrate_heading: bool = True
    calib_rc: int = 25
    calib_push_s: float = 0.7
    calib_settle_s: float = 1.2
    calib_min_move_m: float = 0.06
    calib_timeout_s: float = 25.0


@dataclass
class PipelineConfig:
    control_hz: float = 30.0
    detection_hz_top: float = 25.0
    detection_hz_side: float = 25.0
    preview_hz: float = 30.0
    show_detection_overlay: bool = True


@dataclass
class AppConfig:
    cameras: CameraConfig = field(default_factory=CameraConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    apriltag: AprilTagConfig = field(default_factory=AprilTagConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    failsafe: FailsafeConfig = field(default_factory=FailsafeConfig)
    flight: FlightConfig = field(default_factory=FlightConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    debug_windows: bool = True
    control_enabled: bool = True


def _deep_update(obj, updates: dict) -> None:
    for k, v in updates.items():
        if hasattr(obj, k) and isinstance(v, dict) and not isinstance(getattr(obj, k), (tuple, type(None))):
            _deep_update(getattr(obj, k), v)
        elif hasattr(obj, k):
            setattr(obj, k, v)


def _migrate_legacy_aruco_section(data: dict) -> None:
    """Stary klucz JSON „aruco” → „apriltag” (tag36h11, 40 mm)."""
    if "apriltag" in data or "aruco" not in data:
        return
    legacy = dict(data.pop("aruco"))
    legacy.pop("dictionary_name", None)
    legacy.pop("use_aruco3_detection", None)
    legacy.pop("use_template_fallback", None)
    legacy.pop("template_match_threshold", None)
    legacy.pop("template_match_margin", None)
    legacy.pop("template_match_scales", None)
    legacy.pop("min_marker_perimeter_rate", None)
    if legacy.get("marker_length_m") == 0.02:
        legacy["marker_length_m"] = 0.04
    legacy.setdefault("family", "tag36h11")
    data["apriltag"] = legacy


def load_config(path: str | Path | None = None) -> AppConfig:
    cfg = AppConfig()
    env_path = os.environ.get("TELLCAM_CONFIG")
    p = Path(path or env_path or Path(__file__).resolve().parent / "tellcam_config.json")
    if p.is_file():
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        _migrate_legacy_aruco_section(data)
        _deep_update(cfg, data)
    return cfg
