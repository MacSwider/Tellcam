"""
Misja demo: START → wznoszenie do takeoff_altitude_m → kalibracja kierunku → zawis hover_before_path_s
→ lot po trasie → LAND.
Przy utracie trackingu z kamery TOP dron wisi w miejscu (tylko utrzymanie wysokości) do czasu odzyskania detekcji.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from enum import Enum
from typing import List, Optional, Tuple

from config import FlightConfig
from controller import ZERO_RC, DroneController, RCCommand
from flight_path import FlightPath, PathFollower
from state_estimator import DroneState
from tello_interface import TelloInterface

log = logging.getLogger(__name__)


class Phase(str, Enum):
    IDLE = "IDLE"
    TAKEOFF = "TAKEOFF"
    CLIMB = "CLIMB"
    CALIBRATE = "CALIBRATE"
    HOVER_WAIT = "HOVER_WAIT"
    FOLLOW = "FOLLOW"
    FINISHED = "FINISHED"
    LANDING = "LANDING"


AIRBORNE_PHASES = (Phase.CLIMB, Phase.CALIBRATE, Phase.HOVER_WAIT, Phase.FOLLOW, Phase.FINISHED)

_CALIB_STEPS = ("stabilizacja", "ruch do przodu", "stabilizacja", "ruch w prawo", "stabilizacja")


class FlightMission:
    def __init__(
        self,
        cfg: FlightConfig,
        tello: TelloInterface,
        controller: DroneController,
        path: FlightPath,
    ) -> None:
        self._cfg = cfg
        self._tello = tello
        self._ctl = controller
        self.path = path
        self.follower = PathFollower(path)
        self.phase = Phase.IDLE
        self._phase_t0 = time.perf_counter()
        self._busy: Optional[threading.Thread] = None
        self._land_requested = False
        self._hold_xyz: Optional[Tuple[float, float, float]] = None
        self._hold_yaw: Optional[float] = None
        self._lost_since: Optional[float] = None
        self._was_tracking = False
        self._ground_since: Optional[float] = None
        self._vel_xy: Optional[Tuple[float, float]] = None
        self.last_rc: RCCommand = ZERO_RC
        self.last_alt_m: Optional[float] = None
        self._calib_step = 0
        self._calib_step_t0 = 0.0
        self._calib_pts: List[Tuple[float, float]] = []
        self._calib_yaw_ref: Optional[float] = None
        # (kąt przodu drona w układzie kamery [rad], 1 lub -1 = obraz lustrzany, kurs IMU przy kalibracji [rad])
        self.heading_cal: Optional[Tuple[float, int, Optional[float]]] = None

    # --- polecenia z GUI ---

    def request_takeoff(self) -> None:
        if self.phase != Phase.IDLE:
            log.info("START zignorowany — faza %s", self.phase.value)
            return
        self._land_requested = False
        self._set_phase(Phase.TAKEOFF)
        self._run_async(self._do_takeoff)

    def request_land(self) -> None:
        if self.phase == Phase.IDLE:
            return
        if self.phase == Phase.TAKEOFF:
            self._land_requested = True
            log.info("LAND zapamiętany — wykonam po zakończeniu startu")
            return
        if self.phase == Phase.LANDING:
            return
        self._begin_landing()

    def shutdown(self) -> None:
        if self.phase in AIRBORNE_PHASES:
            self._tello.send_rc_zero()
            try:
                self._tello.land()
            except Exception as e:
                log.warning("land() przy zamykaniu: %s", e)
        if self._busy and self._busy.is_alive():
            self._busy.join(timeout=10.0)

    # --- pętla sterowania ---

    def tick(
        self, state: DroneState, dt: float, vel_xy: Optional[Tuple[float, float]] = None
    ) -> List[str]:
        """state: pozycja XY (po filtrze), vel_xy: prędkość XY w układzie kamery TOP lub None."""
        self._vel_xy = vel_xy
        now = time.perf_counter()
        alt = self._altitude(state)
        self.last_alt_m = alt

        if self.phase in AIRBORNE_PHASES and self._detect_self_landing(now):
            return self._status_lines(state, alt, now)

        if self.phase == Phase.CLIMB:
            self._tick_climb(alt, now, dt)
        elif self.phase == Phase.CALIBRATE:
            self._tick_calibrate(state, alt, now, dt)
        elif self.phase == Phase.HOVER_WAIT:
            self._tick_hover_wait(state, alt, now, dt)
        elif self.phase in (Phase.FOLLOW, Phase.FINISHED):
            self._tick_follow(state, alt, now, dt)

        return self._status_lines(state, alt, now)

    # --- fazy ---

    def _tick_climb(self, alt: Optional[float], now: float, dt: float) -> None:
        target = self._cfg.takeoff_altitude_m
        if alt is None:
            self._send(ZERO_RC)
            return
        reached = abs(alt - target) <= self._cfg.altitude_tolerance_m
        if reached or now - self._phase_t0 > self._cfg.climb_timeout_s:
            if reached:
                log.info("Osiągnięto %.2f m", alt)
            else:
                log.warning("Wznoszenie: timeout (wysokość %.2f m)", alt)
            if self._cfg.calibrate_heading:
                self._enter_calibrate()
            else:
                self._enter_hover_wait()
            return
        self._send(self._ctl.altitude_only(target, alt, dt))

    def _enter_calibrate(self) -> None:
        self._ctl.reset()
        self.heading_cal = None
        self._calib_restart()
        self._set_phase(Phase.CALIBRATE)

    def _calib_restart(self) -> None:
        self._calib_step = 0
        self._calib_step_t0 = time.perf_counter()
        self._calib_pts = []

    def _tick_calibrate(self, state: DroneState, alt: Optional[float], now: float, dt: float) -> None:
        cfg = self._cfg
        if now - self._phase_t0 > cfg.calib_timeout_s:
            log.error("Kalibracja kierunku nieudana w %.0f s — lądowanie", cfg.calib_timeout_s)
            self._begin_landing()
            return
        hold = self._ctl.altitude_only(cfg.takeoff_altitude_m, alt, dt) if alt is not None else ZERO_RC
        if not self._tracking(state, alt):
            if self._calib_step > 0 or self._calib_pts:
                log.warning("Kalibracja: utrata trackingu — zaczynam od nowa")
            self._calib_restart()
            self._send(hold)
            return

        elapsed = now - self._calib_step_t0
        step = self._calib_step
        if step in (1, 3):
            push = cfg.calib_rc
            self._send(
                RCCommand(push if step == 3 else 0, push if step == 1 else 0, hold.throttle, 0)
            )
            if elapsed >= cfg.calib_push_s:
                self._calib_next(now)
            return

        self._send(hold)
        if elapsed < cfg.calib_settle_s:
            return
        self._calib_pts.append((state.x_m, state.y_m))
        if step == 0:
            yaw = self._tello.yaw_deg()
            self._calib_yaw_ref = math.radians(yaw) if yaw is not None else None
        if step < 4:
            self._calib_next(now)
            return
        self._calib_finish()

    def _calib_next(self, now: float) -> None:
        self._calib_step += 1
        self._calib_step_t0 = now

    def _calib_finish(self) -> None:
        (x0, y0), (x1, y1), (x2, y2) = self._calib_pts
        fx, fy = x1 - x0, y1 - y0
        rx, ry = x2 - x1, y2 - y1
        nf, nr = math.hypot(fx, fy), math.hypot(rx, ry)
        if min(nf, nr) < self._cfg.calib_min_move_m:
            log.warning(
                "Kalibracja: za mały ruch (przód %.0f cm, prawo %.0f cm) — powtarzam", nf * 100, nr * 100
            )
            self._calib_restart()
            return
        hand = 1 if fx * ry - fy * rx >= 0 else -1
        phi_f = math.atan2(fy, fx)
        phi_r = math.atan2(ry, rx) - hand * math.pi / 2
        skew = math.atan2(math.sin(phi_r - phi_f), math.cos(phi_r - phi_f))
        phi = phi_f + skew / 2
        self.heading_cal = (phi, hand, self._calib_yaw_ref)
        log.info(
            "Kalibracja OK: przód drona = %.0f° w obrazie TOP, obraz lustrzany=%s, ruch %.0f/%.0f cm, "
            "odchyłka od prostopadłości %.0f°",
            math.degrees(phi), "tak" if hand < 0 else "nie", nf * 100, nr * 100, math.degrees(skew),
        )
        if abs(skew) > math.radians(45):
            log.warning("Kalibracja: ruchy mało prostopadłe — dron mógł dryfować, wynik niepewny")
        self._enter_hover_wait()

    def _basis(self) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        if self.heading_cal is None:
            return None
        phi, hand, yaw_ref = self.heading_cal
        yaw = self._tello.yaw_deg()
        delta = 0.0
        if yaw_ref is not None and yaw is not None:
            d = math.radians(yaw) - yaw_ref
            delta = math.atan2(math.sin(d), math.cos(d))
        th = phi + hand * delta
        tr = th + hand * math.pi / 2
        return (math.cos(th), math.sin(th)), (math.cos(tr), math.sin(tr))

    def _yaw(self, state: DroneState) -> float:
        if self.heading_cal is not None and self._cfg.yaw_source == "tello":
            yaw = self._tello.yaw_deg()
            if yaw is not None:
                return math.radians(yaw)
        return state.yaw_rad

    def _hold_alt(self, target_alt: float, alt: Optional[float], dt: float) -> RCCommand:
        return self._ctl.altitude_only(target_alt, alt, dt) if alt is not None else ZERO_RC

    def _enter_hover_wait(self) -> None:
        self._ctl.reset()
        self._hold_xyz = None
        self._hold_yaw = None
        self._set_phase(Phase.HOVER_WAIT)

    def _tick_hover_wait(self, state: DroneState, alt: Optional[float], now: float, dt: float) -> None:
        if not self._tracking(state, alt):
            self._send(self._hold_alt(self._cfg.takeoff_altitude_m, alt, dt))
            return
        yaw = self._yaw(state)
        if self._hold_xyz is None:
            self._hold_xyz = (state.x_m, state.y_m, self._cfg.takeoff_altitude_m)
        if self._hold_yaw is None and self._cfg.hold_yaw:
            self._hold_yaw = yaw
        self._send(
            self._ctl.compute(
                self._hold_xyz, self._hold_yaw, state.x_m, state.y_m, alt, yaw, dt, self._vel_xy,
                basis=self._basis(),
            )
        )
        if now - self._phase_t0 >= self._cfg.hover_before_path_s:
            log.info("Start trasy „%s” (%d punktów, w tym pomocnicze)", self.path.name, len(self.path.points))
            self.follower.reset()
            self._ctl.reset()
            self._set_phase(Phase.FOLLOW)

    def _tick_follow(self, state: DroneState, alt: Optional[float], now: float, dt: float) -> None:
        if not self._tracking(state, alt):
            self._send(self._hold_alt(self.follower.target.xyz[2], alt, dt))
            limit = self._cfg.lost_tracking_land_after_s
            if limit > 0 and self._lost_since is not None and now - self._lost_since > limit:
                log.warning("Brak trackingu > %.1f s — lądowanie", limit)
                self._begin_landing()
            return
        yaw = self._yaw(state)
        if self._hold_yaw is None and self._cfg.hold_yaw:
            self._hold_yaw = yaw
        tgt = self.follower.update(state.x_m, state.y_m, alt, dt)
        if self.follower.finished and self.phase != Phase.FINISHED:
            log.info("Trasa zakończona — zawis w ostatnim punkcie (LAND aby wylądować)")
            self._set_phase(Phase.FINISHED)
        self._send(
            self._ctl.compute(
                tgt.xyz, self._hold_yaw, state.x_m, state.y_m, alt, yaw, dt, self._vel_xy, basis=self._basis()
            )
        )

    def _tracking(self, state: DroneState, alt: Optional[float]) -> bool:
        ok = state.valid and alt is not None
        if ok and not self._was_tracking:
            if self._lost_since is not None:
                log.info("Tracking odzyskany po %.1f s — wznawiam", time.perf_counter() - self._lost_since)
            self._ctl.reset()
            self._lost_since = None
        elif not ok and self._was_tracking:
            log.warning("Utrata trackingu TOP — zawis w miejscu")
            self._lost_since = time.perf_counter()
        elif not ok and self._lost_since is None:
            self._lost_since = time.perf_counter()
        self._was_tracking = ok
        return ok

    def _detect_self_landing(self, now: float) -> bool:
        """Tello ląduje sam m.in. przy przegrzaniu lub niskiej baterii — wtedy nie wysyłamy już RC."""
        if self._tello.get_height_cm() > 5:
            self._ground_since = None
            return False
        if self._ground_since is None:
            self._ground_since = now
            return False
        if now - self._ground_since < 2.0:
            return False
        log.warning("Dron wylądował samoczynnie (wysokość 0 przez 2 s). Telemetria: %s", self._tello.telemetry())
        self._tello.send_rc_zero()
        self._tello.mark_landed()
        self.last_rc = ZERO_RC
        self._ctl.reset()
        self._ground_since = None
        self._set_phase(Phase.IDLE)
        return True

    # --- start / lądowanie w tle ---

    def _run_async(self, fn) -> None:
        self._busy = threading.Thread(target=fn, name="TelloCommand", daemon=True)
        self._busy.start()

    def _do_takeoff(self) -> None:
        log.info("START — telemetria przed startem: %s", self._tello.telemetry())
        try:
            self._tello.takeoff()
        except Exception as e:
            log.error("takeoff nieudany: %s", e)
            self._set_phase(Phase.IDLE)
            return
        log.info("Start zakończony — wznoszenie do %.2f m", self._cfg.takeoff_altitude_m)
        self._ctl.reset()
        self._was_tracking = False
        self._lost_since = None
        self._ground_since = None
        self._set_phase(Phase.CLIMB)
        if self._land_requested:
            self._begin_landing()

    def _begin_landing(self) -> None:
        self._tello.send_rc_zero()
        self.last_rc = ZERO_RC
        self._set_phase(Phase.LANDING)
        self._run_async(self._do_land)

    def _do_land(self) -> None:
        try:
            self._tello.land()
        except Exception as e:
            log.error("land nieudany: %s", e)
        self._ctl.reset()
        self._set_phase(Phase.IDLE)

    # --- pomocnicze ---

    def _altitude(self, state: DroneState) -> Optional[float]:
        if self._cfg.altitude_source == "vision":
            return self._cfg.camera_height_m - state.z_m if state.valid else None
        if not self._tello.connected:
            return None
        return self._tello.get_height_m()

    def _send(self, cmd: RCCommand) -> None:
        self.last_rc = cmd
        self._tello.send_rc(cmd)

    def _set_phase(self, phase: Phase) -> None:
        if phase != self.phase:
            log.info("Faza: %s → %s", self.phase.value, phase.value)
        self.phase = phase
        self._phase_t0 = time.perf_counter()

    def _status_lines(self, state: DroneState, alt: Optional[float], now: float) -> List[str]:
        alt_s = f"{alt:.2f}m" if alt is not None else "—"
        line = f"FAZA={self.phase.value} alt={alt_s}"
        if self.phase == Phase.CALIBRATE:
            line += f" kalibracja {self._calib_step + 1}/5: {_CALIB_STEPS[self._calib_step]}"
        elif self.phase == Phase.HOVER_WAIT:
            left = max(0.0, self._cfg.hover_before_path_s - (now - self._phase_t0))
            line += f" start trasy za {left:.1f}s" if left > 0 else " czekam na tracking"
        elif self.phase in (Phase.FOLLOW, Phase.FINISHED):
            tgt = self.follower.target
            kind = "główny" if tgt.is_main else "pomocn."
            line += (
                f" cel #{self.follower.index + 1}/{len(self.path.points)} ({kind}) "
                f"[{tgt.xyz[0]:+.2f},{tgt.xyz[1]:+.2f},{tgt.xyz[2]:.2f}]"
            )
        if self.phase in AIRBORNE_PHASES and self._lost_since is not None:
            line += f" | BRAK TRACKINGU {now - self._lost_since:.1f}s — zawis"
        if self.heading_cal is not None and self.phase in AIRBORNE_PHASES:
            phi, hand, _ = self.heading_cal
            line += f" | kal: przod={math.degrees(phi):.0f}st{' lustro' if hand < 0 else ''}"
        rc = self.last_rc
        tel = self._tello.telemetry() or {}
        return [
            line,
            f"RC r={rc.roll} p={rc.pitch} thr={rc.throttle} y={rc.yaw} | "
            f"bateria={tel.get('bat', '?')}% temp={tel.get('temp', '?')}C",
        ]
