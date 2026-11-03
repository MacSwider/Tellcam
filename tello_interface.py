"""Komunikacja z Ryze Tello (djitellopy) + tryb symulacji bez sprzętu."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from controller import RCCommand

log = logging.getLogger(__name__)


@dataclass
class TelloConfig:
    mock: bool = False
    auto_connect: bool = True


class TelloInterface:
    """
    left_right = roll, forward_back = pitch, up_down = throttle, yaw = yaw
    (zgodnie z djitellopy.send_rc_control).
    """

    def __init__(self, cfg: TelloConfig) -> None:
        self._cfg = cfg
        self._tello = None
        self._connected = False
        self._mock_height_cm = 0.0
        self._mock_yaw_deg = 0.0
        self._mock_last_rc_t = 0.0

    def connect(self) -> bool:
        if self._cfg.mock:
            log.info("TelloInterface: tryb MOCK — brak połączenia z dronem.")
            self._connected = True
            self._mock_height_cm = 0
            return True
        try:
            from djitellopy import Tello
            from djitellopy import tello as tello_module
        except ImportError as e:
            log.error("Brak pakietu djitellopy: pip install djitellopy")
            raise e

        # Nowa instancja przy każdej próbie jest szkodliwa: __del__ starej usuwa z globalnego
        # rejestru djitellopy wpis dla tego samego adresu, a kolejne connect() kończy się KeyError.
        if self._tello is None:
            self._tello = Tello()
        tello_module.drones.setdefault(self._tello.address[0], {"responses": [], "state": {}})
        self._tello.connect()
        self._connected = True
        log.info("Połączono z Tello, bateria: %s", self._tello.get_battery())
        return True

    def disconnect(self) -> None:
        if self._tello is not None:
            try:
                self._tello.end()
            except Exception as e:
                log.warning("end(): %s", e)
            # __del__ woła end() ponownie; przy is_flying=True próbowałby land() na usuniętym wpisie → KeyError.
            self._tello.is_flying = False
        self._tello = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def send_rc(self, cmd: RCCommand) -> None:
        if not self._connected:
            return
        if self._cfg.mock:
            now = time.perf_counter()
            dt = min(now - self._mock_last_rc_t, 0.2) if self._mock_last_rc_t else 0.0
            self._mock_last_rc_t = now
            if self._mock_height_cm > 0:
                # ~1 cm/s na jednostkę throttle
                self._mock_height_cm = max(10.0, self._mock_height_cm + cmd.throttle * dt)
                self._mock_yaw_deg += cmd.yaw * dt
            return
        if self._tello is None:
            return
        self._tello.send_rc_control(cmd.roll, cmd.pitch, cmd.throttle, cmd.yaw)

    def send_rc_zero(self) -> None:
        self.send_rc(RCCommand(0, 0, 0, 0))

    def takeoff(self) -> None:
        if self._cfg.mock:
            self._mock_height_cm = 85
            log.info("MOCK takeoff — wysokość ~%s cm", self._mock_height_cm)
            return
        if self._tello:
            self._tello.takeoff()

    def land(self) -> None:
        if self._cfg.mock:
            self._mock_height_cm = 0
            log.info("MOCK land")
            return
        if self._tello:
            self._tello.land()

    def mark_landed(self) -> None:
        """Dron wylądował bez naszej komendy land()."""
        if self._cfg.mock:
            self._mock_height_cm = 0
        if self._tello is not None:
            self._tello.is_flying = False

    def emergency_stop(self) -> None:
        """Natychmiastowe wyłączenie silników — używać ostrożnie."""
        if self._cfg.mock:
            self._mock_height_cm = 0
            log.warning("MOCK emergency — silniki zatrzymane (symulacja)")
            return
        if self._tello:
            try:
                self._tello.emergency()
            except Exception as e:
                log.warning("emergency(): %s", e)

    def get_height_cm(self) -> int:
        """Wysokość wg Tello (pole `h` w stanie, zwykle cm nad startem). MOCK: symulowane."""
        if self._cfg.mock or self._tello is None:
            return int(self._mock_height_cm)
        try:
            return int(self._tello.get_height())
        except Exception as e:
            log.warning("get_height: %s", e)
            return 0

    def telemetry(self) -> Optional[dict]:
        """Bateria [%] i temperatura [°C] ze strumienia stanu Tello."""
        if self._cfg.mock:
            return {"bat": 100, "temp": 50}
        if self._tello is None:
            return None
        try:
            st = self._tello.get_current_state()
            return {"bat": int(st.get("bat", -1)), "temp": int(st.get("temph", -1))}
        except Exception:
            return None

    def velocity(self) -> Optional[tuple[int, int]]:
        """Surowe vgx, vgy ze strumienia stanu Tello (jednostki wg SDK, zwykle dm/s)."""
        if self._cfg.mock or self._tello is None:
            return None
        try:
            st = self._tello.get_current_state()
            return int(st.get("vgx", 0)), int(st.get("vgy", 0))
        except Exception:
            return None

    def yaw_deg(self) -> Optional[float]:
        """Kurs z IMU Tello [°], rośnie przy obrocie w prawo (zgodnie z RC yaw +)."""
        if self._cfg.mock:
            return self._mock_yaw_deg
        if self._tello is None:
            return None
        try:
            return float(self._tello.get_current_state()["yaw"])
        except Exception:
            return None

    def get_height_m(self) -> float:
        return self.get_height_cm() / 100.0
