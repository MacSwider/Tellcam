"""Komunikacja z Ryze Tello (djitellopy) + tryb symulacji bez sprzętu."""
from __future__ import annotations

import logging
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
        self._mock_height_cm = 0

    def connect(self) -> bool:
        if self._cfg.mock:
            log.info("TelloInterface: tryb MOCK — brak połączenia z dronem.")
            self._connected = True
            self._mock_height_cm = 0
            return True
        try:
            from djitellopy import Tello
        except ImportError as e:
            log.error("Brak pakietu djitellopy: pip install djitellopy")
            raise e

        self._tello = Tello()
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
        self._tello = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def send_rc(self, cmd: RCCommand) -> None:
        if not self._connected:
            return
        if self._cfg.mock or self._tello is None:
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

    def get_height_m(self) -> float:
        return self.get_height_cm() / 100.0
