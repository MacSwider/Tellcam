"""
Maszyna stanów lotu demo: start -> wznoszenie do ~1.5 m -> hold „w miejscu” -> lądowanie.

- CLIMB: otwarta pętla (delikatne up), zanim kamery złapią znaczniki; sterowana
  wysokością z czujnika Tello, z limitem czasu (failsafe).
- HOLD: zamknięta pętla na fuzji TOP+SIDE; cel = poza zarejestrowana przy wejściu
  w fazę (utrzymanie aktualnej pozycji). Krótka utrata znacznika obsłużona przez
  active_axes/freeze_axes z estymatora.
"""
from __future__ import annotations

import logging

from config import ClimbConfig, SafetyConfig, StabilizationConfig, TargetConfig
from controller import RCCommand, StabilizationController
from state_estimator import DroneState

log = logging.getLogger(__name__)

IDLE = "IDLE"
CLIMB = "CLIMB"
HOLD = "HOLD"
LANDING = "LANDING"


class FlightDirector:
    def __init__(
        self,
        controller: StabilizationController,
        stab_cfg: StabilizationConfig,
        climb_cfg: ClimbConfig,
        safety_cfg: SafetyConfig,
    ) -> None:
        self._controller = controller
        self._stab = stab_cfg
        self._climb = climb_cfg
        self._safety = safety_cfg
        self.phase = IDLE
        self.reason = ""
        self._climb_t0 = 0.0
        self._hold_t0 = 0.0
        self._target_captured = False

    def start_flight(self, now: float) -> None:
        """Wywołać po fizycznym takeoff()."""
        self.phase = CLIMB
        self.reason = "start: wznoszenie"
        self._climb_t0 = now
        self._target_captured = False
        self._controller.reset()
        log.info("FlightDirector: CLIMB do %.2f m", self._climb.target_height_m)

    def request_land(self) -> None:
        self.phase = LANDING
        self.reason = "lądowanie"

    def reset(self) -> None:
        self.phase = IDLE
        self.reason = ""
        self._target_captured = False
        self._controller.reset()

    @property
    def in_flight_sequence(self) -> bool:
        return self.phase in (CLIMB, HOLD)

    def _acquired(self, state: DroneState) -> bool:
        need = {"x", "y", "z"}
        return need.issubset(set(state.active_axes))

    def _capture_hold_target(self, state: DroneState) -> None:
        if not self._stab.hold_capture_on_enter:
            return
        # Dla yaw "side" celem jest 0 (tag zwrócony wprost do kamery bocznej);
        # dla "top" przechwytujemy bieżący kurs.
        yaw_source = (self._stab.yaw_source or "off").strip().lower()
        yaw_target = 0.0 if yaw_source != "top" else state.yaw_rad
        self._controller.set_target(
            TargetConfig(
                x_m=state.x_m,
                y_m=state.y_m,
                z_m=state.z_m,
                yaw_rad=yaw_target,
            )
        )
        self._target_captured = True
        log.info(
            "FlightDirector: HOLD — cel x=%.2f y=%.2f z=%.2f yaw=%.2f",
            state.x_m, state.y_m, state.z_m, yaw_target,
        )

    def update(self, state: DroneState, height_m: float, dt: float, now: float) -> RCCommand:
        if self.phase == IDLE or self.phase == LANDING:
            return RCCommand.zero()

        if self.phase == CLIMB:
            height_ok = height_m >= float(self._climb.target_height_m)
            acquired = self._acquired(state)
            timed_out = (now - self._climb_t0) >= float(self._climb.timeout_s)
            ready = (height_ok and (acquired or not self._climb.require_top_and_side))
            if ready or timed_out:
                self.phase = HOLD
                self._hold_t0 = now
                self.reason = "hold (kamery)" if not timed_out else "hold (timeout wznoszenia)"
                self._controller.reset()
                if acquired:
                    self._capture_hold_target(state)
                return RCCommand.zero()
            # Delikatne wznoszenie; przy celu wysokości zwolnij, by nie przestrzelić.
            up = int(self._climb.rc_up if not height_ok else max(0, self._climb.rc_up // 3))
            self.reason = f"wznoszenie h={height_m:.2f}/{self._climb.target_height_m:.2f}m"
            return RCCommand(0, 0, up, 0)

        # HOLD
        if not self._target_captured:
            if self._acquired(state):
                self._capture_hold_target(state)
            else:
                self.reason = "hold: czekam na pełną pozę (TOP+SIDE)"
                return RCCommand.zero()
        if not state.valid:
            self.reason = f"hold: brak pozy ({state.tracking_state})"
            return RCCommand.zero()
        self.reason = f"hold: {state.tracking_state}"
        return self._controller.compute(
            state.x_m,
            state.y_m,
            state.z_m,
            state.yaw_rad,
            dt,
            active_axes=state.active_axes,
            freeze_axes=state.freeze_axes,
            vx=state.vx_m_s,
            vy=state.vy_m_s,
            vz=state.vz_m_s,
            heading_rad=state.heading_rad,
            heading_valid=state.heading_valid,
        )
