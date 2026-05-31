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
    # MC-Venus: driver często zwraca 1280x960 niezależnie od żądania 1280x720
    width: int = 1280
    height: int = 960
    buffer_size: int = 1
    drain_frames_on_read: int = 2
    fourcc: str = "MJPG"


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
    quad_decimate: float = 2.0
    quad_sigma: float = 0.0
    refine_edges: bool = True
    decode_sharpening: float = 0.5
    min_decision_margin: float = 4.0
    # Ręczny upscaling jest kosztowny i rzadko pomaga — domyślnie tylko 1.0×.
    detection_upscales: Tuple[float, ...] = (1.0,)
    detection_lazy_upscale: bool = True
    # Kosztowny fallback (kontury/morfologia) — tylko do trybu offline/debug.
    use_black_crop_fallback: bool = False
    black_crop_thresholds: Tuple[int, ...] = (50, 60, 70, 80, 90)
    black_crop_pad_frac: float = 0.35
    black_crop_upscale: int = 2
    max_detection_width: int = 0
    max_markers_per_frame: int = 5
    # Śledzenie ROI: po wykryciu szukaj w kolejnej klatce tylko w wycinku wokół taga.
    roi_tracking: bool = True
    roi_pad_frac: float = 0.6
    roi_min_size_px: int = 160


@dataclass
class PIDGains:
    kp: float = 35.0
    ki: float = 0.0
    kd: float = 10.0
    output_limit: float = 20.0
    integral_limit: float = 8.0
    deadzone: float = 0.03
    slew_rate: float = 80.0


@dataclass
class ControllerConfig:
    pid_x: PIDGains = field(default_factory=PIDGains)
    pid_y: PIDGains = field(default_factory=PIDGains)
    pid_z: PIDGains = field(default_factory=lambda: PIDGains(kp=30.0, kd=8.0, deadzone=0.04, output_limit=18.0))
    pid_yaw: PIDGains = field(
        default_factory=lambda: PIDGains(kp=25.0, kd=6.0, deadzone=0.08, output_limit=15.0)
    )
    max_rc_abs: int = 20
    use_yaw: bool = False


@dataclass
class TargetConfig:
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.8
    yaw_rad: float = 0.0


@dataclass
class StabilizationConfig:
    demo_side_only: bool = False
    control_frame: str = "top"
    enabled_axes: Tuple[str, ...] = ("x", "y", "z")
    use_yaw: bool = False
    top_reference_tag_id: int = 0
    target_pose_m: TargetConfig = field(default_factory=TargetConfig)


@dataclass
class TrackingConfig:
    hold_last_pose_s: float = 0.6
    lost_timeout_s: float = 1.5
    pose_lowpass_alpha: float = 0.35


@dataclass
class SafetyConfig:
    max_rc_abs: int = 20
    max_rc_slew_per_s: float = 80.0
    rc_send_hz: float = 15.0
    zero_rc_on_loss: bool = True
    command_timeout_s: float = 1.0


@dataclass
class GuiConfig:
    show_pid_debug: bool = True


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
    apriltag: AprilTagConfig = field(default_factory=AprilTagConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    stabilization: StabilizationConfig = field(default_factory=StabilizationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    gui: GuiConfig = field(default_factory=GuiConfig)
    failsafe: FailsafeConfig = field(default_factory=FailsafeConfig)
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
