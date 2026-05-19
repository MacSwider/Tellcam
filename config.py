"""
Centralna konfiguracja Tellcam: kamery, pipeline, ArUco, PID, failsafe.
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
    width: int = 640
    height: int = 480
    buffer_size: int = 1
    drain_frames_on_read: int = 2


@dataclass
class ArucoConfig:
    dictionary_name: str = "DICT_4X4_50"
    marker_ids: Tuple[int, ...] = (0, 1, 2, 3, 4)
    marker_length_m: float = 0.02
    layout_half_forward_m: float = 0.07
    layout_half_lateral_m: float = 0.07
    marker_side_height_m: float = 0.03
    calibration_path: str | None = None
    intrinsics_from_frame_size: bool = False
    use_aruco3_detection: bool = True
    use_clahe: bool = True
    min_marker_perimeter_rate: float = 0.01
    use_template_fallback: bool = True
    template_match_threshold: float = 0.38
    template_match_margin: float = 0.05
    template_match_scales: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    max_detection_width: int = 640
    max_markers_per_frame: int = 5


@dataclass
class PIDGains:
    kp: float = 0.8
    ki: float = 0.02
    kd: float = 0.15
    output_limit: float = 100.0
    integral_limit: float = 25.0


@dataclass
class ControllerConfig:
    pid_x: PIDGains = field(default_factory=PIDGains)
    pid_y: PIDGains = field(default_factory=PIDGains)
    pid_z: PIDGains = field(default_factory=lambda: PIDGains(kp=1.2, ki=0.05, kd=0.2))
    pid_yaw: PIDGains = field(
        default_factory=lambda: PIDGains(kp=0.5, ki=0.01, kd=0.1, output_limit=100.0)
    )
    deadzone_m: float = 0.02
    deadzone_yaw_rad: float = 0.05


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
class PipelineConfig:
    control_hz: float = 30.0
    detection_hz_top: float = 12.0
    detection_hz_side: float = 12.0
    preview_hz: float = 30.0
    show_detection_overlay: bool = True


@dataclass
class AppConfig:
    cameras: CameraConfig = field(default_factory=CameraConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    aruco: ArucoConfig = field(default_factory=ArucoConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    failsafe: FailsafeConfig = field(default_factory=FailsafeConfig)
    debug_windows: bool = True
    control_enabled: bool = False


def _deep_update(obj, updates: dict) -> None:
    for k, v in updates.items():
        if hasattr(obj, k) and isinstance(v, dict) and not isinstance(getattr(obj, k), (tuple, type(None))):
            _deep_update(getattr(obj, k), v)
        elif hasattr(obj, k):
            setattr(obj, k, v)


def load_config(path: str | Path | None = None) -> AppConfig:
    cfg = AppConfig()
    env_path = os.environ.get("TELLCAM_CONFIG")
    p = Path(path or env_path or Path(__file__).resolve().parent / "tellcam_config.json")
    if p.is_file():
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        _deep_update(cfg, data)
    return cfg
