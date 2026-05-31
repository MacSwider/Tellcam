"""Estymacja pozy dla demo TOP+SIDE z podtrzymaniem śledzenia."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from apriltag_detector import SideCorrectionObservation, TopPoseObservation
from config import TrackingConfig


@dataclass
class DroneState:
    valid: bool
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0
    yaw_rad: float = 0.0
    pitch_rad: float = 0.0
    roll_rad: float = 0.0
    top_ok: bool = False
    side_ok: bool = False
    tracking_state: str = "LOST"
    pose_age_s: float = 0.0
    measurement_age_s: float = 0.0
    marker_id: int | None = None
    hint: str | None = None


class PoseEstimator:
    def __init__(self, cfg: TrackingConfig) -> None:
        self._cfg = cfg
        self._last_valid: DroneState | None = None
        self._last_valid_ts = 0.0

    def reset(self) -> None:
        self._last_valid = None
        self._last_valid_ts = 0.0

    def update(
        self,
        top: TopPoseObservation,
        side: SideCorrectionObservation,
        *,
        top_ts: float = 0.0,
        side_ts: float = 0.0,
        now: float = 0.0,
    ) -> DroneState:
        top_ok = bool(top.ok)
        side_ok = bool(side.ok)

        if top_ok:
            hint_parts: list[str] = []
            if top.hint:
                hint_parts.append(top.hint)
            if side_ok:
                if side.hint:
                    hint_parts.append(side.hint)
            measured = DroneState(
                valid=True,
                x_m=top.x_m,
                y_m=top.y_m,
                z_m=top.z_m,
                yaw_rad=self._unwrap_yaw(top.yaw_rad),
                pitch_rad=top.pitch_rad,
                roll_rad=top.roll_rad,
                top_ok=True,
                side_ok=side_ok,
                tracking_state="TRACKING",
                pose_age_s=0.0,
                measurement_age_s=max(0.0, now - top_ts) if top_ts else 0.0,
                marker_id=top.reference_marker_id,
                hint=" | ".join(x for x in hint_parts if x) or None,
            )
            measured = self._lowpass(measured)
            self._last_valid = measured
            self._last_valid_ts = now
            return measured

        if self._last_valid is None:
            return DroneState(
                valid=False,
                top_ok=top_ok,
                side_ok=side_ok,
                tracking_state="LOST",
                hint=top.hint or side.hint,
            )

        age = max(0.0, now - self._last_valid_ts)
        stale = replace(
            self._last_valid,
            top_ok=top_ok,
            side_ok=False,
            pose_age_s=age,
            measurement_age_s=age,
            hint=top.hint or side.hint or self._last_valid.hint,
        )
        if age <= self._cfg.hold_last_pose_s:
            return replace(stale, valid=True, tracking_state="HOLDING_LAST")
        if age <= self._cfg.lost_timeout_s:
            return replace(stale, valid=False, tracking_state="LOST")
        return replace(stale, valid=False, tracking_state="TIMED_OUT")

    def _lowpass(self, measured: DroneState) -> DroneState:
        alpha = float(np.clip(self._cfg.pose_lowpass_alpha, 0.0, 1.0))
        if self._last_valid is None or alpha <= 0.0:
            return measured
        if alpha >= 1.0:
            return self._last_valid
        prev = self._last_valid
        return DroneState(
            valid=measured.valid,
            x_m=(1.0 - alpha) * measured.x_m + alpha * prev.x_m,
            y_m=(1.0 - alpha) * measured.y_m + alpha * prev.y_m,
            z_m=(1.0 - alpha) * measured.z_m + alpha * prev.z_m,
            yaw_rad=self._blend_angles(measured.yaw_rad, prev.yaw_rad, alpha),
            pitch_rad=measured.pitch_rad,
            roll_rad=measured.roll_rad,
            top_ok=measured.top_ok,
            side_ok=measured.side_ok,
            tracking_state=measured.tracking_state,
            pose_age_s=measured.pose_age_s,
            measurement_age_s=measured.measurement_age_s,
            marker_id=measured.marker_id,
            hint=measured.hint,
        )

    @staticmethod
    def _unwrap_yaw(yaw: float) -> float:
        return float(np.arctan2(np.sin(yaw), np.cos(yaw)))

    @staticmethod
    def _blend_angles(current: float, previous: float, alpha: float) -> float:
        delta = np.arctan2(np.sin(previous - current), np.cos(previous - current))
        return float(current + alpha * delta)


StateEstimator = PoseEstimator
