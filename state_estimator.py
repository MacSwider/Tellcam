"""
Fuzja pozy dla demo TOP+SIDE z filtrem alfa-beta (pozycja + prędkość) i predykcją.

Założenia fizyczne (dwie zewnętrzne kamery USB obserwujące drona z tagami):
- Kamera GÓRNA (TOP) najlepiej mierzy pozycję poziomą: x (lewo/prawo), y (przód/tył) i yaw.
- Kamera BOCZNA (SIDE) najlepiej mierzy WYSOKOŚĆ drona (pionowa pozycja = side.y_m).

Każda oś (x, y z TOP; altitude z SIDE) ma filtr alfa-beta, który:
- estymuje prędkość z kolejnych detekcji,
- przewiduje pozycję między detekcjami i w krótkich dziurach (do max_predict_s),
co skraca opóźnienie sprzężenia i pozwala wcześniej tłumić dryf drona.

Stany źródła: TRACKING -> HOLDING_LAST -> LOST -> TIMED_OUT (per kamera).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from apriltag_detector import SideCorrectionObservation, TopPoseObservation
from config import StabilizationConfig, TrackingConfig

TRACKING = "TRACKING"
HOLDING_LAST = "HOLDING_LAST"
LOST = "LOST"
TIMED_OUT = "TIMED_OUT"


@dataclass
class DroneState:
    valid: bool
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0
    yaw_rad: float = 0.0
    pitch_rad: float = 0.0
    roll_rad: float = 0.0
    vx_m_s: float = 0.0
    vy_m_s: float = 0.0
    vz_m_s: float = 0.0
    # Kurs drona (do transformacji do układu ciała): TOP, awaryjnie wyrównany SIDE.
    heading_rad: float = 0.0
    heading_valid: bool = False
    # Surowy kurs z kamery bocznej (layout tagów 1–4) — do diagnostyki/podglądu.
    side_heading_rad: float = 0.0
    side_heading_valid: bool = False
    top_ok: bool = False
    side_ok: bool = False
    top_state: str = LOST
    side_state: str = LOST
    tracking_state: str = LOST
    active_axes: tuple[str, ...] = ()
    freeze_axes: tuple[str, ...] = ()
    pose_age_s: float = 0.0
    measurement_age_s: float = 0.0
    marker_id: int | None = None
    hint: str | None = None


class AlphaBetaAxis:
    """Filtr alfa-beta: estymuje pozycję i prędkość, przewiduje w czasie."""

    def __init__(self, alpha: float, beta: float, max_speed: float) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_speed = float(max_speed)
        self.x: float | None = None
        self.v: float = 0.0
        self.t: float = 0.0

    def reset(self) -> None:
        self.x = None
        self.v = 0.0
        self.t = 0.0

    def update(self, meas: float, t: float) -> None:
        if self.x is None:
            self.x = float(meas)
            self.v = 0.0
            self.t = t
            return
        dt = t - self.t
        if dt <= 1e-4:
            dt = 1e-3
        x_pred = self.x + self.v * dt
        residual = float(meas) - x_pred
        self.x = x_pred + self.alpha * residual
        self.v = self.v + (self.beta / dt) * residual
        self.v = float(np.clip(self.v, -self.max_speed, self.max_speed))
        self.t = t

    def predict(self, t: float, max_horizon: float) -> float:
        """Pozycja przewidziana na czas t (ekstrapolacja prędkością, z limitem horyzontu)."""
        if self.x is None:
            return 0.0
        horizon = min(max(0.0, t - self.t), max_horizon)
        return self.x + self.v * horizon


class PoseEstimator:
    def __init__(self, cfg: TrackingConfig, stab_cfg: StabilizationConfig | None = None) -> None:
        self._cfg = cfg
        self._stab = stab_cfg or StabilizationConfig()
        a, b, vmax = cfg.filter_alpha, cfg.filter_beta, cfg.max_speed_m_s
        self._fx = AlphaBetaAxis(a, b, vmax)
        self._fy = AlphaBetaAxis(a, b, vmax)
        self._falt = AlphaBetaAxis(a, b, vmax)
        self._ftop_z = AlphaBetaAxis(a, b, vmax)
        self._top_ts = 0.0
        self._side_ts = 0.0
        self._top_meas_ts = 0.0
        self._side_meas_ts = 0.0
        self._yaw = 0.0
        self._pitch = 0.0
        self._roll = 0.0
        self._side_facing = 0.0
        self._side_heading = 0.0  # ciągły kurs z layoutu tagów bocznych
        self._has_side_heading = False
        self._side_to_top_offset = 0.0  # wyrównanie kursu SIDE do ramki TOP
        self._has_offset = False
        self._top_n = 0
        self._yaw_source = (self._stab.yaw_source or "off").strip().lower()
        self._use_yaw = bool(self._stab.use_yaw)
        self._side_yaw_sign = float(getattr(self._stab, "side_yaw_sign", 1.0))
        self._side_yaw_offset = np.radians(float(getattr(self._stab, "side_yaw_offset_deg", 0.0)))
        # Azymuty „na zewnątrz” tagów bocznych [rad], klucze jako int.
        az = getattr(self._stab, "side_tag_azimuths_deg", {}) or {}
        self._side_azimuth = {int(k): np.radians(float(v)) for k, v in az.items()}
        self._has_top = False
        self._has_side = False

    def reset(self) -> None:
        for f in (self._fx, self._fy, self._falt, self._ftop_z):
            f.reset()
        self._top_ts = self._side_ts = 0.0
        self._top_meas_ts = self._side_meas_ts = 0.0
        self._has_top = self._has_side = False

    def _state_for_age(self, age: float) -> str:
        if age <= self._cfg.hold_last_pose_s:
            return HOLDING_LAST
        if age <= self._cfg.lost_timeout_s:
            return LOST
        return TIMED_OUT

    def update(
        self,
        top: TopPoseObservation,
        side: SideCorrectionObservation,
        *,
        top_ts: float = 0.0,
        side_ts: float = 0.0,
        now: float = 0.0,
    ) -> DroneState:
        predict = bool(self._cfg.predict_enabled)
        max_h = float(self._cfg.max_predict_s)

        # --- TOP: nowy pomiar tylko gdy zmienił się znacznik czasu publikacji ---
        if top.ok and top_ts > self._top_ts:
            self._fx.update(float(top.x_m), now)
            self._fy.update(float(top.y_m), now)
            self._ftop_z.update(float(top.z_m), now)
            self._yaw = self._unwrap_yaw(float(top.yaw_rad))
            self._pitch = float(top.pitch_rad)
            self._roll = float(top.roll_rad)
            self._top_n = len(top.markers_seen)
            self._top_ts = top_ts
            self._top_meas_ts = now
            self._has_top = True
        top_state = TRACKING if (top.ok and now - self._top_meas_ts <= 1e-6) else (
            self._state_for_age(now - self._top_meas_ts) if self._has_top else LOST
        )

        # --- SIDE: wysokość (pionowa pozycja = side.y_m) + kurs z layoutu tagów ---
        if side.ok and side_ts > self._side_ts:
            self._falt.update(float(side.y_m), now)
            self._side_facing = self._lowpass_angle(self._side_facing, float(side.yaw_facing_rad))
            # Kurs ciągły: yaw_facing tagu skorygowany jego azymutem w układzie.
            # Dzięki temu przy przełączeniu widocznego tagu nie ma skoku ~90°.
            mid = int(side.marker_id) if side.marker_id is not None else -1
            az = self._side_azimuth.get(mid)
            if az is not None:
                raw_heading = self._unwrap_yaw(
                    self._side_yaw_sign * (float(side.yaw_facing_rad) - az) + self._side_yaw_offset
                )
                if self._has_side_heading:
                    self._side_heading = self._lowpass_angle(self._side_heading, raw_heading)
                else:
                    self._side_heading = raw_heading
                    self._has_side_heading = True
            self._side_ts = side_ts
            self._side_meas_ts = now
            self._has_side = True
        side_state = TRACKING if (side.ok and now - self._side_meas_ts <= 1e-6) else (
            self._state_for_age(now - self._side_meas_ts) if self._has_side else LOST
        )

        top_usable = top_state in (TRACKING, HOLDING_LAST)
        side_usable = side_state in (TRACKING, HOLDING_LAST)
        alt_from_side = self._stab.altitude_source.strip().lower() == "side"

        # Predykcja pozycji na bieżącą chwilę (skraca opóźnienie, łapie dryf wcześniej).
        if self._has_top:
            x_m = self._fx.predict(now, max_h) if predict else (self._fx.x or 0.0)
            y_m = self._fy.predict(now, max_h) if predict else (self._fy.x or 0.0)
            vx, vy = self._fx.v, self._fy.v
            top_z = self._ftop_z.predict(now, max_h) if predict else (self._ftop_z.x or 0.0)
        else:
            x_m = y_m = vx = vy = top_z = 0.0

        if alt_from_side and self._has_side:
            z_m = self._falt.predict(now, max_h) if predict else (self._falt.x or 0.0)
            vz = self._falt.v
            z_state, z_usable = side_state, side_usable
        elif not alt_from_side and self._has_top:
            z_m, vz = top_z, self._ftop_z.v
            z_state, z_usable = top_state, top_usable
        else:
            z_m, vz = 0.0, 0.0
            z_state, z_usable = side_state if alt_from_side else top_state, False

        # Predykcja ufana tylko do max_predict_s; dalej w oknie hold -> trzymaj, zamroź integrator.
        top_predicting = (now - self._top_meas_ts) <= max_h
        z_predicting = (now - (self._side_meas_ts if alt_from_side else self._top_meas_ts)) <= max_h

        active: list[str] = []
        freeze: list[str] = []
        if top_usable:
            active.extend(["x", "y"])
            if not top_predicting:
                freeze.extend(["x", "y"])
                vx = vy = 0.0
        if z_usable:
            active.append("z")
            if not z_predicting:
                freeze.append("z")
                vz = 0.0

        # --- yaw: kurs z TOP (kształt tagu 0) i/lub SIDE (layout tagów 1–4) ---
        top_yaw_usable = top_usable and self._top_n >= int(self._stab.yaw_top_min_markers)
        side_yaw_usable = side_usable and self._has_side_heading

        # Naucz przesunięcia SIDE->TOP, gdy oba kursy są dostępne (bezbolesny handoff).
        if top_yaw_usable and side_yaw_usable:
            target_off = self._unwrap_yaw(self._yaw - self._side_heading)
            if self._has_offset:
                self._side_to_top_offset = self._lowpass_angle(self._side_to_top_offset, target_off)
            else:
                self._side_to_top_offset = target_off
                self._has_offset = True
        aligned_side = self._unwrap_yaw(self._side_heading + self._side_to_top_offset)

        # Kurs dla transformacji do układu ciała: zawsze preferuj TOP (ramka x/y),
        # awaryjnie wyrównany kurs SIDE (tylko gdy poznano przesunięcie).
        if top_yaw_usable:
            heading_for_body = self._yaw
            heading_valid = True
        elif side_yaw_usable and self._has_offset:
            heading_for_body = aligned_side
            heading_valid = True
        else:
            heading_for_body = self._yaw
            heading_valid = False

        # Wejście sterujące yaw wg wybranego źródła.
        yaw_meas = self._yaw
        if self._use_yaw and self._yaw_source != "off":
            if self._yaw_source == "side":
                yaw_meas = self._side_heading
                yaw_usable = side_yaw_usable
                yaw_predicting = (now - self._side_meas_ts) <= max_h
            elif self._yaw_source == "auto":
                if top_yaw_usable:
                    yaw_meas = self._yaw
                    yaw_usable = True
                    yaw_predicting = top_predicting
                elif side_yaw_usable:
                    yaw_meas = aligned_side if self._has_offset else self._side_heading
                    yaw_usable = True
                    yaw_predicting = (now - self._side_meas_ts) <= max_h
                else:
                    yaw_usable = False
                    yaw_predicting = False
            else:  # "top"
                yaw_meas = self._yaw
                yaw_usable = top_yaw_usable
                yaw_predicting = top_predicting
            if yaw_usable:
                active.append("yaw")
                if not yaw_predicting:
                    freeze.append("yaw")

        valid = bool(active)
        tracking_state = self._overall_state(top_state, side_state if alt_from_side else top_state)
        age = now - self._top_meas_ts if self._has_top else 0.0

        hints = [h for h in (top.hint, side.hint) if h]
        return DroneState(
            valid=valid,
            x_m=x_m,
            y_m=y_m,
            z_m=z_m,
            yaw_rad=yaw_meas,
            pitch_rad=self._pitch,
            roll_rad=self._roll,
            vx_m_s=vx,
            vy_m_s=vy,
            vz_m_s=vz,
            heading_rad=heading_for_body,
            heading_valid=bool(heading_valid),
            side_heading_rad=self._side_heading,
            side_heading_valid=bool(side_yaw_usable),
            top_ok=bool(top.ok),
            side_ok=bool(side.ok),
            top_state=top_state,
            side_state=side_state,
            tracking_state=tracking_state,
            active_axes=tuple(dict.fromkeys(active)),
            freeze_axes=tuple(dict.fromkeys(freeze)),
            pose_age_s=age,
            measurement_age_s=age,
            marker_id=top.reference_marker_id,
            hint=" | ".join(hints) or None,
        )

    @staticmethod
    def _overall_state(a: str, b: str) -> str:
        order = {TRACKING: 0, HOLDING_LAST: 1, LOST: 2, TIMED_OUT: 3}
        return min(a, b, key=lambda s: order.get(s, 3))

    def _lowpass_angle(self, prev: float, meas: float) -> float:
        alpha = float(np.clip(self._cfg.pose_lowpass_alpha, 0.0, 1.0))
        if alpha <= 0.0:
            return self._unwrap_yaw(meas)
        delta = np.arctan2(np.sin(meas - prev), np.cos(meas - prev))
        return self._unwrap_yaw(prev + (1.0 - alpha) * delta)

    @staticmethod
    def _unwrap_yaw(yaw: float) -> float:
        return float(np.arctan2(np.sin(yaw), np.cos(yaw)))


StateEstimator = PoseEstimator
