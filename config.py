"""
Centralna konfiguracja Tellcam: kamery, pipeline, AprilTag, PID, failsafe.
Nadpisania: plik JSON (tellcam_config.json) lub TELLCAM_CONFIG.
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
    # Mały człon całkujący kompensuje stały dryf taniego Tello (do dostrojenia na sprzęcie).
    ki: float = 4.0
    kd: float = 10.0
    output_limit: float = 20.0
    integral_limit: float = 10.0
    deadzone: float = 0.02
    slew_rate: float = 80.0
    # Minimalna skuteczna komenda RC, gdy błąd > deadzone (Tello ignoruje bardzo małe RC).
    min_command: float = 6.0


@dataclass
class ControllerConfig:
    pid_x: PIDGains = field(default_factory=PIDGains)
    pid_y: PIDGains = field(default_factory=PIDGains)
    pid_z: PIDGains = field(default_factory=lambda: PIDGains(kp=30.0, kd=8.0, deadzone=0.04, output_limit=18.0))
    pid_yaw: PIDGains = field(
        default_factory=lambda: PIDGains(
            kp=18.0, ki=0.0, kd=4.0, output_limit=12.0, integral_limit=6.0,
            deadzone=0.06, slew_rate=60.0, min_command=0.0,
        )
    )
    max_rc_abs: int = 20
    use_yaw: bool = False
    # Znaki komend per oś — odwróć (-1.0), jeśli dron reaguje w przeciwną stronę.
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


@dataclass
class StabilizationConfig:
    demo_side_only: bool = False
    control_frame: str = "top"
    enabled_axes: Tuple[str, ...] = ("x", "y", "z")
    # Stabilizacja obrotu, by dron nie wirował. Źródło yaw:
    #   "top"  – kurs z KSZTAŁTU tagu 0 widzianego z góry (kąt krawędzi w obrazie;
    #            jednoznaczny dla widoku z góry),
    #   "side" – kurs z tagów bocznych 1–4 w UKŁADZIE (yaw_facing − azymut tagu;
    #            ciągły przy zmianie widocznego tagu),
    #   "auto" – TOP gdy dostępny, w przeciwnym razie SIDE (równolegle/awaryjnie),
    #   "off"  – bez sterowania yaw.
    use_yaw: bool = True
    yaw_source: str = "auto"
    # Min. liczba markerów TOP do uznania kursu za wiarygodny. Dla widoku z GÓRY
    # obrót (yaw) to rotacja W PŁASZCZYŹNIE — dobrze obserwowalna już z 1 markera
    # (tag 0). Przy ≥2 markerach kurs jest dodatkowo wspierany układem (front/tył).
    yaw_top_min_markers: int = 1
    # Azymut „na zewnątrz” każdego bocznego tagu w układzie ciała (deg), patrząc z
    # GÓRY (przód=0°, CCW dodatnie). Layout:
    #        PRZÓD
    #     [4]     [1]
    #         [0]
    #     [3]     [2]
    #         TYŁ
    # Dzięki azymutowi kurs z kamery bocznej jest CIĄGŁY przy przełączaniu tagów
    # (heading = side_yaw_sign * (yaw_facing − azymut[id])). Jeśli przy obrocie
    # przez granicę tagów kurs SKACZE o ~90°, odwróć kolejność (zaneguj znaki).
    side_tag_azimuths_deg: dict = field(
        default_factory=lambda: {1: -45.0, 2: -135.0, 3: 135.0, 4: 45.0}
    )
    # Globalny znak kursu z kamery bocznej (gdy obrót drona daje przeciwny kierunek).
    side_yaw_sign: float = 1.0
    # Stałe przesunięcie kursu bocznego [deg] — czysto kosmetyczne (zerowanie panelu).
    # Nie wpływa na sterowanie (cel yaw przechwytywany względnie), ułatwia kalibrację:
    # ustaw drona „przodem do kamery bocznej”, odczytaj side_hdg=X, wpisz tu −X.
    side_yaw_offset_deg: float = 0.0
    # Obróć błąd pozycji do układu ciała drona wg kursu (TOP) zanim trafi na roll/pitch.
    # Bez tego, gdy dron zdryfuje w yaw, komendy idą w złą stronę -> łuk/ucieczka.
    body_frame_control: bool = True
    # Powyżej tego błędu kursu mocno ogranicz ruch poziomy (najpierw wyrównaj yaw).
    yaw_align_deg_for_translation: float = 35.0
    top_reference_tag_id: int = 0
    # Źródło wysokości (oś up/down): "side" = pionowa pozycja z kamery bocznej,
    # "top" = odległość od kamery górnej (gdy brak kamery bocznej).
    altitude_source: str = "side"
    # Hold „w miejscu”: cel = bieżąca poza w chwili wejścia w fazę HOLD.
    hold_capture_on_enter: bool = True
    target_pose_m: TargetConfig = field(default_factory=TargetConfig)


@dataclass
class ClimbConfig:
    """Otwarta pętla wznoszenia po starcie, zanim kamery złapią znaczniki."""
    target_height_m: float = 1.5
    rc_up: int = 25
    timeout_s: float = 6.0
    settle_s: float = 1.0
    require_top_and_side: bool = True


@dataclass
class TrackingConfig:
    # Okno podtrzymania ostatniej pozy przy krótkiej utracie znacznika (1–2 s).
    hold_last_pose_s: float = 0.8
    lost_timeout_s: float = 2.0
    pose_lowpass_alpha: float = 0.35  # używane dla yaw (alfa-beta dla x/y/z)
    # Filtr alfa-beta (estymacja pozycji + prędkości z detekcji) i predykcja.
    predict_enabled: bool = True
    filter_alpha: float = 0.6   # waga korekty pozycji (0..1)
    filter_beta: float = 0.2    # waga korekty prędkości (0..1)
    max_predict_s: float = 0.4  # max horyzont ekstrapolacji w dziurze detekcji
    max_speed_m_s: float = 1.5  # limit estymowanej prędkości (odrzut szumu)


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
    climb: ClimbConfig = field(default_factory=ClimbConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    gui: GuiConfig = field(default_factory=GuiConfig)
    failsafe: FailsafeConfig = field(default_factory=FailsafeConfig)
    debug_windows: bool = True
    control_enabled: bool = True


def _deep_update(obj, updates: dict) -> None:
    for k, v in updates.items():
        # Rekuruj tylko w zagnieżdżone dataclass; zwykłe dict (np. mapy) podmień w całości.
        if hasattr(obj, k) and isinstance(v, dict) and is_dataclass(getattr(obj, k)):
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
