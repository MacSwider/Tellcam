"""
Central Tellcam configuration: cameras, pipeline, AprilTag, PID, failsafe.
Overrides: JSON file (tellcam_config.json) or TELLCAM_CONFIG.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class CameraConfig:
    index_ceiling: int = 0
    index_side: int = 1
    # Default USB capture 1080p — less MJPEG blur than 1440p on Creative Live! Cam 4K / USB2.
    width: int = 1920
    height: int = 1080
    top_width: int | None = None
    top_height: int | None = None
    side_width: int | None = None
    side_height: int | None = None

    def resolution_for(self, role: str) -> Tuple[int, int]:
        role = (role or "top").strip().lower()
        if role == "side":
            w = self.side_width if self.side_width is not None else self.width
            h = self.side_height if self.side_height is not None else self.height
        else:
            w = self.top_width if self.top_width is not None else self.width
            h = self.top_height if self.top_height is not None else self.height
        return int(w), int(h)
    buffer_size: int = 1
    drain_frames_on_read: int = 2
    fourcc: str = "MJPG"
    # USB exposure/focus (None = leave driver default). Lower exposure reduces glare blowout.
    auto_exposure: bool | None = None
    exposure: float | None = None
    gain: float | None = None
    auto_focus: bool | None = False
    focus: float | None = None
    # Creative Live! Cam 4K etc.: manual focus ring — software sweep does not move the lens.
    focus_autotune: bool = False
    focus_autotune_cameras: Tuple[str, ...] = ("top",)
    focus_sweep_min: float = 0.0
    focus_sweep_max: float = 255.0
    focus_sweep_steps: int = 13
    focus_settle_frames: int = 2
    focus_roi_norm: Tuple[float, float, float, float] | None = (0.30, 0.30, 0.70, 0.70)
    brightness: float | None = None
    contrast: float | None = None
    sharpness: float | None = None
    # Per-camera capture overrides (None = use shared exposure/gain/brightness above).
    top_exposure: float | None = None
    side_exposure: float | None = None
    top_gain: float | None = None
    side_gain: float | None = None
    top_brightness: float | None = None
    side_brightness: float | None = None
    top_contrast: float | None = None
    side_contrast: float | None = None
    top_sharpness: float | None = None
    side_sharpness: float | None = None

    def capture_prop(self, role: str, prop: str) -> float | None:
        """Role-specific USB property with fallback to shared cameras.* value."""
        role = (role or "top").strip().lower()
        key = f"{role}_{prop}"
        val = getattr(self, key, None)
        if val is not None:
            return val
        return getattr(self, prop, None)


@dataclass
class AprilTagConfig:
    family: str = "tag36h11"
    marker_ids: Tuple[int, ...] = (0, 1, 2, 3, 4)
    marker_length_m: float = 0.04  # black square side [m], excluding white border
    layout_half_forward_m: float = 0.07
    layout_half_lateral_m: float = 0.07
    marker_side_height_m: float = 0.03
    # Map TOP PnP tvec (OpenCV camera frame) to lab axes: x=left (+), y=forward (+).
    # Body tag layout uses X=forward, Y=lateral — indices swap camera components by default.
    lab_x_tvec_index: int = 1
    lab_y_tvec_index: int = 0
    lab_x_sign: float = -1.0
    lab_y_sign: float = 1.0
    calibration_path: str | None = None
    intrinsics_from_frame_size: bool = True
    use_clahe: bool = True
    clahe_clip_limit: float = 3.0
    use_histogram_eq: bool = True
    # Reduce specular glare (bright lab lights on shiny surfaces).
    use_highlight_suppression: bool = True
    highlight_blur_sigma: float = 25.0
    # MJPEG blur / soft focus mitigation before AprilTag passes.
    use_denoise: bool = True
    denoise_d: int = 5
    denoise_sigma_color: float = 40.0
    denoise_sigma_space: float = 40.0
    use_unsharp_mask: bool = True
    unsharp_sigma: float = 1.2
    unsharp_amount: float = 1.5
    unsharp_threshold: int = 2
    use_adaptive_threshold: bool = True
    adaptive_block_size: int = 31
    adaptive_c: int = 7
    # 1.0 = full-resolution quads; use 1.0 for small/distant tags on wide fisheye TOP.
    quad_decimate: float = 1.0
    quad_sigma: float = 0.8
    refine_edges: bool = True
    decode_sharpening: float = 0.35
    min_decision_margin: float = 2.5
    detection_upscales: Tuple[float, ...] = (1.0, 2.0)
    detection_lazy_upscale: bool = True
    # Upscale fixed search ROI so tiny landing-pad tags get more pixels.
    roi_upscale: float = 2.0
    use_black_crop_fallback: bool = True
    black_crop_thresholds: Tuple[int, ...] = (50, 60, 70, 80, 90)
    black_crop_pad_frac: float = 0.35
    black_crop_upscale: int = 2
    # Max width for full-frame detection (0 = always use full capture resolution).
    max_detection_width: int = 0
    max_markers_per_frame: int = 5
    # ROI tracking: after detection, search next frame only in a crop around the tag.
    roi_tracking: bool = True
    roi_pad_frac: float = 0.6
    roi_min_size_px: int = 180
    # Optional normalized ROI (x0, y0, x1, y1) searched first at full resolution.
    # Useful for fisheye TOP when the landing pad is always in the same image region.
    search_roi_norm: Tuple[float, float, float, float] | None = None


@dataclass
class AprilTagDetectionProfile:
    """Per-camera detection pipeline overrides (merged onto shared apriltag)."""

    marker_ids: Tuple[int, ...] = (0, 1, 2, 3, 4)
    use_clahe: bool = True
    clahe_clip_limit: float = 3.0
    use_histogram_eq: bool = True
    use_highlight_suppression: bool = True
    highlight_blur_sigma: float = 25.0
    use_denoise: bool = True
    denoise_d: int = 5
    denoise_sigma_color: float = 40.0
    denoise_sigma_space: float = 40.0
    use_unsharp_mask: bool = True
    unsharp_sigma: float = 1.2
    unsharp_amount: float = 1.5
    unsharp_threshold: int = 2
    use_adaptive_threshold: bool = True
    adaptive_block_size: int = 31
    adaptive_c: int = 7
    quad_decimate: float = 1.0
    quad_sigma: float = 0.8
    refine_edges: bool = True
    decode_sharpening: float = 0.35
    min_decision_margin: float = 2.5
    detection_upscales: Tuple[float, ...] = (1.0, 2.0)
    detection_lazy_upscale: bool = True
    roi_upscale: float = 2.0
    use_black_crop_fallback: bool = True
    black_crop_thresholds: Tuple[int, ...] = (50, 60, 70, 80, 90)
    black_crop_pad_frac: float = 0.35
    black_crop_upscale: int = 2
    max_detection_width: int = 0
    max_markers_per_frame: int = 5
    roi_tracking: bool = True
    roi_pad_frac: float = 0.6
    roi_min_size_px: int = 180
    search_roi_norm: Tuple[float, float, float, float] | None = None


def default_top_detection_profile() -> AprilTagDetectionProfile:
    """TOP fisheye: small tag 0 on dark mat — keep preprocessing light, upscale more."""
    return AprilTagDetectionProfile(
        marker_ids=(0,),
        use_clahe=False,
        use_histogram_eq=False,
        use_highlight_suppression=False,
        use_adaptive_threshold=False,
        use_black_crop_fallback=False,
        use_denoise=True,
        denoise_d=5,
        denoise_sigma_color=35.0,
        denoise_sigma_space=35.0,
        use_unsharp_mask=True,
        unsharp_sigma=1.0,
        unsharp_amount=1.2,
        unsharp_threshold=3,
        quad_decimate=1.0,
        min_decision_margin=2.0,
        detection_upscales=(1.0, 2.0, 3.0),
        detection_lazy_upscale=True,
        roi_upscale=2.0,
        roi_tracking=True,
        roi_pad_frac=0.75,
        search_roi_norm=None,
        max_markers_per_frame=1,
    )


def default_side_detection_profile() -> AprilTagDetectionProfile:
    """SIDE face view: larger wall tags — mild CLAHE + black-crop for sticker borders."""
    return AprilTagDetectionProfile(
        marker_ids=(1, 2, 3, 4),
        use_clahe=True,
        clahe_clip_limit=2.5,
        use_histogram_eq=False,
        use_highlight_suppression=False,
        use_adaptive_threshold=False,
        use_black_crop_fallback=True,
        black_crop_thresholds=(50, 60, 70, 80, 90),
        use_denoise=True,
        use_unsharp_mask=True,
        unsharp_sigma=1.0,
        unsharp_amount=1.3,
        quad_decimate=1.0,
        min_decision_margin=2.5,
        detection_upscales=(1.0, 2.0),
        detection_lazy_upscale=True,
        roi_tracking=True,
        search_roi_norm=None,
        max_markers_per_frame=4,
    )


def apriltag_for_role(cfg: "AppConfig", role: str) -> AprilTagConfig:
    """Shared marker geometry + per-camera detection profile."""
    from dataclasses import fields, replace

    role = (role or "top").strip().lower()
    profile = cfg.apriltag_top if role == "top" else cfg.apriltag_side
    overrides = {f.name: getattr(profile, f.name) for f in fields(AprilTagDetectionProfile)}
    return replace(cfg.apriltag, **overrides)


@dataclass
class PIDGains:
    kp: float = 35.0
    # Small integral term compensates steady drift on cheap Tello (tune on hardware).
    ki: float = 4.0
    kd: float = 10.0
    output_limit: float = 20.0
    integral_limit: float = 10.0
    deadzone: float = 0.02
    slew_rate: float = 80.0
    # Minimum effective RC when error > deadzone (Tello ignores very small RC).
    min_command: float = 6.0


@dataclass
class ControllerConfig:
    pid_x: PIDGains = field(
        default_factory=lambda: PIDGains(kp=42.0, ki=5.5, kd=12.0, deadzone=0.015, min_command=7.0)
    )
    pid_y: PIDGains = field(
        default_factory=lambda: PIDGains(kp=42.0, ki=5.5, kd=12.0, deadzone=0.015, min_command=7.0)
    )
    pid_z: PIDGains = field(
        default_factory=lambda: PIDGains(kp=30.0, kd=8.0, deadzone=0.04, output_limit=18.0)
    )
    pid_yaw: PIDGains = field(
        default_factory=lambda: PIDGains(
            kp=18.0,
            ki=0.0,
            kd=4.0,
            output_limit=12.0,
            integral_limit=6.0,
            deadzone=0.06,
            slew_rate=60.0,
            min_command=0.0,
        )
    )
    # SIDE camera corrections (hold point/distance in SIDE image).
    pid_side_u: PIDGains = field(
        default_factory=lambda: PIDGains(
            kp=0.18,
            ki=0.0,
            kd=0.06,
            output_limit=12.0,
            integral_limit=6.0,
            deadzone=10.0,
            slew_rate=60.0,
            min_command=0.0,
        )
    )
    pid_side_depth: PIDGains = field(
        default_factory=lambda: PIDGains(
            kp=22.0,
            ki=0.5,
            kd=5.0,
            output_limit=12.0,
            integral_limit=6.0,
            deadzone=0.03,
            slew_rate=60.0,
            min_command=0.0,
        )
    )
    pid_side_x_m: PIDGains = field(
        default_factory=lambda: PIDGains(
            kp=28.0,
            ki=2.0,
            kd=6.0,
            output_limit=14.0,
            integral_limit=6.0,
            deadzone=0.02,
            slew_rate=60.0,
            min_command=0.0,
        )
    )
    pid_top_u: PIDGains = field(
        default_factory=lambda: PIDGains(
            kp=0.12,
            ki=0.0,
            kd=0.04,
            output_limit=18.0,
            integral_limit=6.0,
            deadzone=4.0,
            slew_rate=80.0,
            min_command=5.0,
        )
    )
    pid_top_v: PIDGains = field(
        default_factory=lambda: PIDGains(
            kp=0.12,
            ki=0.0,
            kd=0.04,
            output_limit=18.0,
            integral_limit=6.0,
            deadzone=4.0,
            slew_rate=80.0,
            min_command=5.0,
        )
    )
    side_u_sign: float = 1.0
    side_depth_sign: float = 1.0
    top_u_sign: float = 1.0
    top_v_sign: float = 1.0
    max_rc_abs: int = 20
    use_yaw: bool = False
    # Per-axis command signs — flip (-1.0) if the drone moves the wrong way.
    # roll=left_right, pitch=forward_back, throttle=up_down, yaw.
    roll_sign: float = 1.0
    pitch_sign: float = 1.0
    throttle_sign: float = 1.0
    yaw_sign: float = 1.0


@dataclass
class TargetConfig:
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.8
    yaw_rad: float = 0.0
    # SIDE targets (None = do not hold that quantity in the side image).
    side_u_px: float | None = None
    side_v_px: float | None = None
    side_area_px: float | None = None
    side_z_m: float | None = None
    # SIDE PnP targets [m] for side-guided TRAVEL (lateral x, depth z).
    side_x_m: float | None = None
    # TOP image hold: reference tag centroid [px] at HOLD capture.
    top_u_px: float | None = None
    top_v_px: float | None = None


@dataclass
class StabilizationConfig:
    demo_side_only: bool = False
    control_frame: str = "top"
    enabled_axes: Tuple[str, ...] = ("x", "y", "z")
    # Yaw stabilization to prevent spinning. Yaw source:
    #   "top"  – heading from SHAPE of tag 0 seen from above (edge angle in image;
    #            unambiguous for top-down view),
    #   "side" – heading from side tags 1–4 in LAYOUT (yaw_facing − tag azimuth;
    #            continuous when the visible tag changes),
    #   "auto" – TOP when available, otherwise SIDE (parallel/fallback),
    #   "off"  – no yaw control.
    use_yaw: bool = True
    yaw_source: str = "auto"
    # Min TOP markers for a trustworthy heading. For a TOP-DOWN view, yaw is rotation
    # IN THE PLANE — observable with a single marker (tag 0). With ≥2 markers,
    # heading is additionally supported by layout (front/back).
    yaw_top_min_markers: int = 1
    # Outward azimuth of each side tag in body frame (deg), top-down view
    # (front=0°, CCW positive). Layout:
    #        FRONT
    #     [4]     [1]
    #         [0]
    #     [3]     [2]
    #         BACK
    # Azimuth keeps SIDE heading CONTINUOUS when switching tags
    # (heading = side_yaw_sign * (yaw_facing − azimuth[id])). If heading JUMPS ~90°
    # at tag boundaries, reverse the order (negate signs).
    side_tag_azimuths_deg: dict = field(
        default_factory=lambda: {1: -45.0, 2: -135.0, 3: 135.0, 4: 45.0}
    )
    # Global sign for SIDE heading (if drone rotation gives opposite direction).
    side_yaw_sign: float = 1.0
    # Constant SIDE heading offset [deg] — cosmetic only (panel zeroing).
    # Does not affect control (yaw target captured relatively); helps calibration:
    # point drone at SIDE camera, read side_hdg=X, enter −X here.
    side_yaw_offset_deg: float = 0.0
    # Rotate position error into drone body frame using heading (TOP) before roll/pitch.
    # Without this, yaw drift sends commands the wrong way -> arc/escape.
    body_frame_control: bool = True
    # Above this heading error, strongly limit horizontal motion (align yaw first).
    yaw_align_deg_for_translation: float = 35.0
    top_reference_tag_id: int = 0
    # SIDE camera layout relative to drone at takeoff:
    #   "face_to_face" — SIDE looks at the drone nose (start heading toward SIDE).
    #     Approach/retreat toward SIDE ≈ lab forward (pitch). Marker area/depth → pitch.
    #   "profile" — SIDE views the drone from the side (90°). Approach ≈ lab lateral (roll).
    side_camera_layout: str = "face_to_face"
    # Altitude source (up/down axis): "side" = vertical position from side camera,
    # "top" = distance from ceiling camera (when no side camera).
    altitude_source: str = "side"
    # TOP x/y is primary for horizontal hold when side_primary_horiz is false.
    top_primary_horiz: bool = True
    # SIDE marker pixels primary for horizontal hold (faster detection than TOP tag 0).
    side_primary_horiz: bool = False
    # Scale TOP roll/pitch PID output during HOLD (1.0 = no boost).
    top_hold_gain: float = 1.1
    # Hold reference tag at fixed TOP image point [px] — robust vs PnP meter drift.
    top_image_hold: bool = True
    # 1.0 = horizontal RC from TOP pixels only; 0 = PnP x/y only.
    top_image_hold_blend: float = 1.0
    # Map TOP image u/v error to roll/pitch RC ("roll" | "pitch").
    top_u_rc_axis: str = "roll"
    top_v_rc_axis: str = "pitch"
    # Parallel SIDE marker hold in image (along with TOP):
    #   - centroid_u: same horizontal image point -> roll correction,
    #   - distance: marker size (area) or z_m from PnP -> pitch correction.
    # Disabled during TRAVEL when travel_guidance is "side" (SIDE PnP drives roll/pitch).
    side_image_hold: bool = True
    # 1.0 = horizontal RC from SIDE marker centroid [px]; 0 = ignore SIDE u.
    side_image_hold_blend: float = 1.0
    # Hold SIDE marker area for approach/retreat (face_to_face → pitch).
    side_image_area_hold: bool = True
    side_image_area_blend: float = 1.0
    side_u_rc_axis: str = "roll"
    side_area_rc_axis: str = "pitch"
    side_hold_u: bool = True
    side_hold_v: bool = False
    # "area" = error from marker area px² (~1/d²),
    # "z_m"  = error from PnP distance along side camera axis.
    side_depth_source: str = "area"
    # RC axis for SIDE depth/area hold: "pitch" when face_to_face (toward/away from SIDE),
    # "roll" when profile (SIDE views from the side of the lab).
    side_depth_rc_axis: str = "pitch"
    # Use SIDE marker area / PnP depth for horizontal RC (usually off: altitude drift
    # changes apparent marker size on SIDE).
    side_depth_hold: bool = False
    side_rc_blend: float = 0.4
    # SIDE PnP (x_m, z_m) backup when TOP x/y is weak or lost.
    side_pnp_hold: bool = True
    side_pnp_hold_blend: float = 0.65
    # Multiply side_rc_blend when TOP horizontal axes are inactive.
    side_rc_blend_boost_when_top_lost: float = 1.25
    # Blend for SIDE PnP guidance during TRAVEL (1.0 = ignore TOP x/y feedforward).
    side_travel_blend: float = 0.85
    # Constant pitch trim — counter steady creep toward SIDE (face_to_face: forward drift).
    side_forward_trim_rc: int = 0
    # Constant roll trim — counter steady lateral creep (profile SIDE layout).
    side_lateral_trim_rc: int = -5
    # Open-loop velocity feedforward during HOLD [m/s] in lab frame (x=left+, y=fwd+, z=up+).
    # face_to_face: drone starts nose toward SIDE → drift toward camera ≈ +y; use vy < 0 to compensate.
    hold_creep_enabled: bool = False
    hold_creep_vx_m_s: float = 0.0
    hold_creep_vy_m_s: float = 0.0
    hold_creep_vz_m_s: float = 0.0
    # RC units per 1 m/s lab velocity (tune on hardware; ~80–120 for Tello indoors).
    hold_creep_rc_per_m_s: float = 90.0
    # Hold in place: target = current pose at HOLD entry.
    hold_capture_on_enter: bool = True
    target_pose_m: TargetConfig = field(default_factory=TargetConfig)
    # --- Anti-spin: prevent uncontrolled rotation (spinning) ---
    # Heading is stabilized from the first visible frame (including CLIMB),
    # and yaw target is locked relative to TAG 0 orientation measured at takeoff.
    # No need to align tag perfectly to the camera — we measure actual start
    # heading and hold it for the whole flight.
    yaw_lock_on_takeoff: bool = True
    # Yaw target offset relative to locked start heading [deg].
    # 0 = hold exact orientation at takeoff.
    yaw_reference_offset_deg: float = 0.0
    # Correct yaw during climb (CLIMB) before the drone can spin up.
    yaw_control_during_climb: bool = True
    # Spin watchdog: above this angular rate [deg/s] enter anti-spin mode —
    # yaw correction has priority, horizontal motion (roll/pitch) is damped
    # so the drone does not arc away while spinning. 0 = watchdog disabled.
    spin_rate_limit_deg_s: float = 120.0


@dataclass
class NavigationConfig:
    """
    Navigation mode and (optional) flight path.
    - "hold"  – drone holds one point (TOP+SIDE fusion).
    - "path"  – waypoint file with absolute 3D points or legacy relative steps.

    Absolute path JSON (preferred):
        {"coordinate_unit": "cm", "points": [{"X": 0, "Y": 0, "Z": 100}, ...]}
    X = lateral left (+), Y = forward (+), Z = up (+) in TOP/lab frame.

    Legacy relative steps:
        {"steps": [{"forward_m": 0.5, "left_m": -0.5}, ...]}
    """

    mode: str = "hold"  # "hold" | "path"
    flight_path_path: str | None = None
    # Scale for legacy steps (1 unit = 1 m when 1.0). Absolute paths use coordinate_unit in file.
    units_to_meter: float = 0.01
    # Waypoint reached threshold (position [m] and heading [deg]).
    waypoint_radius_m: float = 0.15
    waypoint_yaw_tol_deg: float = 8.0
    # Speed gate: waypoint counts as reached only when horizontal speed
    # drops below this [m/s] (0 = no gate). Reduces overshoot and oscillation.
    waypoint_speed_gate_m_s: float = 0.12
    # Dwell at each waypoint before advancing [s].
    waypoint_settle_s: float = 0.5
    # "top" = TRAVEL uses TOP waypoints; "side" = follow path using SIDE PnP x/z.
    travel_guidance: str = "top"
    # Default square leg length [m] when loading flight_path.square_cw.json.
    square_side_m: float = 0.5
    # Map body-frame leg integrals to SIDE PnP axes (tune signs on hardware).
    side_travel_fwd_sign: float = 1.0
    side_travel_lat_sign: float = 1.0


@dataclass
class ClimbConfig:
    """Open-loop climb after takeoff until cameras acquire markers."""

    target_height_m: float = 1.0
    rc_up: int = 25
    timeout_s: float = 6.0
    settle_s: float = 1.0
    require_top_and_side: bool = True


@dataclass
class OdometryFusionConfig:
    """
    Complementary fusion: Tello body velocities (high rate) + AprilTag pose (low rate).
    By default only feeds PID D-term — does not shift the alpha-beta position estimate
    (avoids cumulative drift from noisy Tello vgx/vgy).
    """

    enabled: bool = True
    # Mutate alpha-beta filter velocity (risky — can drift position estimate).
    affect_filter: bool = False
    # Blend Tello velocity into exported vx/vy/vz for PID derivative damping.
    use_for_derivative: bool = True
    derivative_blend_max: float = 0.3
    # Used only when affect_filter=true.
    velocity_blend: float = 0.2
    velocity_blend_max: float = 0.45
    stale_ramp_s: float = 0.12
    max_speed_m_s: float = 1.5
    # Ignore odometry below this speed [m/s] (hover noise).
    deadband_m_s: float = 0.04
    # Tello SDK reports cm/s in body frame: x=forward, y=lateral, z=up.
    speed_x_sign: float = 1.0
    speed_y_sign: float = -1.0
    speed_z_sign: float = 1.0
    require_flying: bool = True
    require_heading: bool = True
    fuse_altitude: bool = False


@dataclass
class TrackingConfig:
    # Hold last pose briefly on short marker loss (TOP hold memory).
    hold_last_pose_s: float = 3.0
    lost_timeout_s: float = 4.0
    pose_lowpass_alpha: float = 0.35  # used for yaw (alpha-beta for x/y/z)
    # Alpha-beta filter (position + velocity from detections) and prediction.
    predict_enabled: bool = True
    # Extrapolate pose into control loop (can shrink PID error during gaps — keep false for hold).
    predict_for_control: bool = False
    filter_alpha: float = 0.65  # position correction weight (0..1)
    filter_beta: float = 0.25  # velocity correction weight (0..1)
    # Stronger alpha-beta on TOP horizontal axes (x, y).
    filter_alpha_xy: float = 0.75
    filter_beta_xy: float = 0.28
    max_predict_s: float = 0.6  # max extrapolation horizon in detection gaps
    max_speed_m_s: float = 1.5  # estimated speed limit (noise rejection)
    odometry: OdometryFusionConfig = field(default_factory=OdometryFusionConfig)


@dataclass
class GeofenceConfig:
    """
    Flight space fence in TOP camera frame [m].
    Zone is anchored at the reference point captured on HOLD entry or TRAVEL start
    (center = current drone pose). On breach, drone sends zero RC (hover) until
    return; path with waypoint outside zone does not start / is aborted.
    """

    enabled: bool = True
    half_x_m: float = 0.75
    half_y_m: float = 0.75
    z_below_m: float = 0.4
    z_above_m: float = 0.5
    block_travel_outside: bool = True


@dataclass
class SafetyConfig:
    max_rc_abs: int = 20
    max_rc_slew_per_s: float = 80.0
    rc_send_hz: float = 50.0
    zero_rc_on_loss: bool = True
    command_timeout_s: float = 1.0
    geofence: GeofenceConfig = field(default_factory=GeofenceConfig)


@dataclass
class GuiConfig:
    show_pid_debug: bool = True


@dataclass
class FailsafeConfig:
    max_lost_frames: int = 15
    on_lost_send_zero_rc: bool = True


@dataclass
class PipelineConfig:
    control_hz: float = 50.0
    detection_hz_top: float = 20.0
    detection_hz_side: float = 20.0
    preview_hz: float = 30.0
    show_detection_overlay: bool = True
    # OpenCV window max width (0 = native; capture can still be 2K while display is scaled).
    preview_max_width: int = 1920


@dataclass
class AppConfig:
    cameras: CameraConfig = field(default_factory=CameraConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    apriltag: AprilTagConfig = field(default_factory=AprilTagConfig)
    apriltag_top: AprilTagDetectionProfile = field(default_factory=default_top_detection_profile)
    apriltag_side: AprilTagDetectionProfile = field(default_factory=default_side_detection_profile)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    stabilization: StabilizationConfig = field(default_factory=StabilizationConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    climb: ClimbConfig = field(default_factory=ClimbConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    gui: GuiConfig = field(default_factory=GuiConfig)
    failsafe: FailsafeConfig = field(default_factory=FailsafeConfig)
    debug_windows: bool = True
    control_enabled: bool = True


def _deep_update(obj, updates: dict) -> None:
    for k, v in updates.items():
        # Recurse only into nested dataclasses; plain dicts (e.g. maps) replace wholesale.
        if hasattr(obj, k) and isinstance(v, dict) and is_dataclass(getattr(obj, k)):
            _deep_update(getattr(obj, k), v)
        elif hasattr(obj, k):
            setattr(obj, k, v)


def _migrate_legacy_aruco_section(data: dict) -> None:
    """Legacy JSON key "aruco" -> "apriltag" (tag36h11, 40 mm)."""
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
