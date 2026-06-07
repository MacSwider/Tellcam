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
import math

from config import ClimbConfig, SafetyConfig, StabilizationConfig, TargetConfig
from controller import RCCommand, StabilizationController
from state_estimator import DroneState

log = logging.getLogger(__name__)


def _angle_diff(a: float, b: float) -> float:
    """Najkrótsza różnica kątów a-b w zakresie (-pi, pi]."""
    return math.atan2(math.sin(a - b), math.cos(a - b))

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
        # --- Anti-spin: zatrzaśnięty kurs odniesienia + watchdog wirowania ---
        self._yaw_lock_rad: float | None = None
        self._yaw_offset = math.radians(float(getattr(stab_cfg, "yaw_reference_offset_deg", 0.0)))
        self._prev_yaw: float | None = None
        self._prev_yaw_t = 0.0
        self.spin_rate_deg_s = 0.0
        self.spin_active = False
        # Kamera jest wysoko — TAG 0 staje się widoczny dopiero po wzniesieniu.
        # Kurs odniesienia zatrzaskujemy więc dopiero po osiągnięciu wysokości CLIMB.
        self._climb_height_reached = False

    def start_flight(self, now: float) -> None:
        """Wywołać po fizycznym takeoff()."""
        self.phase = CLIMB
        self.reason = "start: wznoszenie"
        self._climb_t0 = now
        self._target_captured = False
        self._yaw_lock_rad = None
        self._prev_yaw = None
        self.spin_active = False
        self._climb_height_reached = False
        self._controller.reset()
        log.info("FlightDirector: CLIMB do %.2f m", self._climb.target_height_m)

    def request_land(self) -> None:
        self.phase = LANDING
        self.reason = "lądowanie"

    def reset(self) -> None:
        self.phase = IDLE
        self.reason = ""
        self._target_captured = False
        self._yaw_lock_rad = None
        self._prev_yaw = None
        self.spin_active = False
        self._climb_height_reached = False
        self._controller.reset()

    @property
    def in_flight_sequence(self) -> bool:
        return self.phase in (CLIMB, HOLD)

    def _acquired(self, state: DroneState) -> bool:
        need = {"x", "y", "z"}
        return need.issubset(set(state.active_axes))

    def _maybe_lock_yaw(self, state: DroneState) -> None:
        """Zatrzaśnij kurs odniesienia z TAG 0 — dopiero po wzniesieniu na wysokość CLIMB.

        Kamera jest na tyle wysoko, że tuż nad podłogą TAG 0 nie jest widoczny,
        więc blokadę łapiemy po osiągnięciu wysokości docelowej (~1.5 m), gdy tag
        jest w kadrze i kurs jest wiarygodny.
        """
        if not getattr(self._stab, "yaw_lock_on_takeoff", True):
            return
        if not self._climb_height_reached:
            return
        if self._yaw_lock_rad is not None or not state.heading_valid:
            return
        self._yaw_lock_rad = _angle_diff(state.yaw_rad + self._yaw_offset, 0.0)
        log.info(
            "FlightDirector: zatrzaśnięto kurs odniesienia yaw=%.1f deg (anti-spin)",
            math.degrees(self._yaw_lock_rad),
        )

    def _desired_yaw(self, state: DroneState) -> float:
        """Docelowy kurs: zatrzaśnięta orientacja startowa, gdy dostępna."""
        yaw_source = (self._stab.yaw_source or "off").strip().lower()
        if self._yaw_lock_rad is not None and yaw_source in ("top", "auto"):
            return self._yaw_lock_rad
        # Dla yaw "side" celem jest 0 (tag zwrócony wprost do kamery bocznej).
        return 0.0 if yaw_source != "top" else state.yaw_rad

    def _update_spin_watchdog(self, state: DroneState, now: float) -> None:
        """Oszacuj prędkość kątową kursu i ustaw flagę trybu anti-spin."""
        limit = float(getattr(self._stab, "spin_rate_limit_deg_s", 0.0))
        if not state.heading_valid or limit <= 0.0:
            self.spin_active = False
            self._prev_yaw = state.yaw_rad if state.heading_valid else None
            self._prev_yaw_t = now
            return
        if self._prev_yaw is not None:
            dt = max(1e-3, now - self._prev_yaw_t)
            rate = math.degrees(_angle_diff(state.yaw_rad, self._prev_yaw)) / dt
            # Lekkie wygładzenie, by pojedyncza skokowa detekcja nie wyzwalała trybu.
            self.spin_rate_deg_s = 0.6 * self.spin_rate_deg_s + 0.4 * rate
            spinning = abs(self.spin_rate_deg_s) > limit
            if spinning and not self.spin_active:
                log.warning(
                    "FlightDirector: ANTI-SPIN — wykryto obrót %.0f deg/s (>%.0f). "
                    "Tłumię ruch poziomy. Jeśli obrót się nasila, sprawdź yaw_sign!",
                    self.spin_rate_deg_s, limit,
                )
            self.spin_active = spinning
        self._prev_yaw = state.yaw_rad
        self._prev_yaw_t = now

    def _capture_hold_target(self, state: DroneState) -> None:
        if not self._stab.hold_capture_on_enter:
            return
        self._maybe_lock_yaw(state)
        yaw_target = self._desired_yaw(state)
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

        # Watchdog wirowania liczony w każdej fazie lotu (anti-spin).
        self._update_spin_watchdog(state, now)

        if self.phase == CLIMB:
            height_ok = height_m >= float(self._climb.target_height_m)
            # Po osiągnięciu wysokości docelowej TAG 0 jest w kadrze -> można
            # zatrzasnąć kurs odniesienia i włączyć korekcję yaw (anti-spin).
            if height_ok:
                self._climb_height_reached = True
            self._maybe_lock_yaw(state)
            acquired = self._acquired(state)
            timed_out = (now - self._climb_t0) >= float(self._climb.timeout_s)
            ready = (height_ok and (acquired or not self._climb.require_top_and_side))
            if ready or timed_out:
                self.phase = HOLD
                self._hold_t0 = now
                # W HOLD jesteśmy już w zawisie — pozwól zatrzasnąć kurs nawet po timeoucie.
                self._climb_height_reached = True
                self.reason = "hold (kamery)" if not timed_out else "hold (timeout wznoszenia)"
                self._controller.reset()
                if acquired:
                    self._capture_hold_target(state)
                return RCCommand.zero()
            # Delikatne wznoszenie; przy celu wysokości zwolnij, by nie przestrzelić.
            up = int(self._climb.rc_up if not height_ok else max(0, self._climb.rc_up // 3))
            # Anti-spin: koryguj yaw dopiero gdy mamy zatrzaśnięty kurs (czyli po
            # wzniesieniu i wykryciu TAG 0). Na niskim pułapie dron tylko się wznosi.
            yaw_cmd = 0
            if (
                getattr(self._stab, "yaw_control_during_climb", True)
                and self._yaw_lock_rad is not None
                and state.heading_valid
            ):
                yaw_cmd = self._controller.compute_yaw_command(state.yaw_rad, self._yaw_lock_rad, dt)
            spin = " [ANTI-SPIN]" if self.spin_active else ""
            self.reason = f"wznoszenie h={height_m:.2f}/{self._climb.target_height_m:.2f}m{spin}"
            return RCCommand(0, 0, up, yaw_cmd)

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
        cmd = self._controller.compute(
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
        # Anti-spin: gdy dron wiruje, najpierw wyrównaj kurs — tłum ruch poziomy,
        # by nie „uciekł” łukiem. Wysokość (up_down) i korekcję yaw zostawiamy.
        if self.spin_active:
            self.reason = f"hold: {state.tracking_state} [ANTI-SPIN {self.spin_rate_deg_s:+.0f}deg/s]"
            return RCCommand(0, 0, cmd.up_down, cmd.yaw)
        return cmd
