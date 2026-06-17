"""Ryze Tello communication (djitellopy) + hardware-free simulation mode."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from controller import RCCommand

log = logging.getLogger(__name__)


@dataclass
class TelloConfig:
    mock: bool = False
    auto_connect: bool = True
    rc_send_hz: float = 50.0
    telemetry_refresh_s: float = 1.0


@dataclass
class BodyVelocity:
    """Tello-reported speed in body frame [m/s]: x=forward, y=lateral, z=up."""

    vx_m_s: float = 0.0
    vy_m_s: float = 0.0
    vz_m_s: float = 0.0
    valid: bool = False


@dataclass
class TelemetrySnapshot:
    connected: bool = False
    flying: bool = False
    battery: int | None = None
    current_rc: RCCommand = field(default_factory=RCCommand.zero)
    body_velocity: BodyVelocity = field(default_factory=BodyVelocity)
    last_hover_reason: str = ""
    last_rc_sent_s: float = 0.0


class TelloController:
    """
    left_right = roll, forward_back = pitch, up_down = throttle, yaw = yaw
    (per djitellopy.send_rc_control).
    """

    def __init__(self, cfg: TelloConfig) -> None:
        self._cfg = cfg
        self._tello = None
        self._connected = False
        self._mock_height_cm = 0
        self._flying = False
        self._last_rc_sent_s = 0.0
        self._last_battery_refresh_s = 0.0
        self._body_velocity = BodyVelocity()
        self._telemetry = TelemetrySnapshot()

    def connect(self) -> bool:
        if self._cfg.mock:
            log.info("TelloInterface: MOCK mode — no drone connection.")
            self._connected = True
            self._mock_height_cm = 0
            self._flying = False
            self._telemetry.connected = True
            self._telemetry.flying = False
            self._telemetry.battery = 100
            return True
        try:
            from djitellopy import Tello
        except ImportError as e:
            log.error("Missing djitellopy package: pip install djitellopy")
            raise e

        self._tello = Tello()
        self._tello.connect()
        self._connected = True
        self._refresh_telemetry(force=True)
        log.info("Connected to Tello, battery: %s", self._telemetry.battery)
        return True

    def disconnect(self) -> None:
        if self._tello is not None:
            try:
                self._tello.end()
            except Exception as e:
                log.warning("end(): %s", e)
        self._tello = None
        self._connected = False
        self._flying = False
        self._telemetry.connected = False
        self._telemetry.flying = False
        self._telemetry.current_rc = RCCommand.zero()

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def flying(self) -> bool:
        return self._flying

    def get_snapshot(self) -> TelemetrySnapshot:
        self._refresh_telemetry()
        return TelemetrySnapshot(
            connected=self._telemetry.connected,
            flying=self._telemetry.flying,
            battery=self._telemetry.battery,
            current_rc=self._telemetry.current_rc,
            body_velocity=self._body_velocity,
            last_hover_reason=self._telemetry.last_hover_reason,
            last_rc_sent_s=self._telemetry.last_rc_sent_s,
        )

    def send_rc(self, cmd: RCCommand, *, force: bool = False, hover_reason: str = "") -> bool:
        self._telemetry.current_rc = cmd
        if hover_reason:
            self._telemetry.last_hover_reason = hover_reason
        if not self._connected:
            return False
        now = time.perf_counter()
        min_period = 1.0 / max(self._cfg.rc_send_hz, 1.0)
        if not force and (now - self._last_rc_sent_s) < min_period:
            return False
        if self._cfg.mock or self._tello is None:
            self._last_rc_sent_s = now
            self._telemetry.last_rc_sent_s = now
            return True
        self._tello.send_rc_control(cmd.roll, cmd.pitch, cmd.throttle, cmd.yaw)
        self._last_rc_sent_s = now
        self._telemetry.last_rc_sent_s = now
        return True

    def send_rc_zero(self, *, force: bool = False, hover_reason: str = "hover") -> bool:
        return self.send_rc(RCCommand.zero(), force=force, hover_reason=hover_reason)

    def takeoff(self) -> None:
        if self._cfg.mock:
            self._mock_height_cm = 85
            self._flying = True
            self._telemetry.flying = True
            log.info("MOCK takeoff — height ~%s cm", self._mock_height_cm)
            return
        if self._tello:
            self._tello.takeoff()
            self._flying = True
            self._telemetry.flying = True

    def land(self) -> None:
        if self._cfg.mock:
            self._mock_height_cm = 0
            self._flying = False
            self._telemetry.flying = False
            log.info("MOCK land")
            return
        if self._tello:
            self._tello.land()
            self._flying = False
            self._telemetry.flying = False

    def emergency_stop(self) -> None:
        """Immediate motor shutdown — use with caution."""
        if self._cfg.mock:
            self._mock_height_cm = 0
            self._flying = False
            self._telemetry.flying = False
            self._telemetry.current_rc = RCCommand.zero()
            self._telemetry.last_hover_reason = "emergency"
            log.warning("MOCK emergency — motors stopped (simulation)")
            return
        if self._tello:
            try:
                self._tello.emergency()
                self._flying = False
                self._telemetry.flying = False
                self._telemetry.current_rc = RCCommand.zero()
                self._telemetry.last_hover_reason = "emergency"
            except Exception as e:
                log.warning("emergency(): %s", e)

    def get_height_cm(self) -> int:
        """Height from Tello (`h` field in state, usually cm above takeoff). MOCK: simulated."""
        if self._cfg.mock or self._tello is None:
            return int(self._mock_height_cm)
        try:
            return int(self._tello.get_height())
        except Exception as e:
            log.warning("get_height: %s", e)
            return 0

    def get_height_m(self) -> float:
        return self.get_height_cm() / 100.0

    def get_body_velocity_m_s(self) -> BodyVelocity:
        """Body-frame velocity from Tello SDK (cm/s -> m/s). MOCK: RC-derived estimate."""
        self._refresh_body_velocity()
        return self._body_velocity

    def _refresh_body_velocity(self) -> None:
        if not self._connected:
            self._body_velocity = BodyVelocity()
            return
        if self._cfg.mock or self._tello is None:
            rc = self._telemetry.current_rc
            # Rough open-loop estimate for simulation (not for long dead reckoning).
            scale = 0.012
            self._body_velocity = BodyVelocity(
                vx_m_s=float(rc.forward_back) * scale,
                vy_m_s=float(rc.left_right) * scale,
                vz_m_s=float(rc.throttle) * scale,
                valid=bool(self._flying),
            )
            return
        try:
            vx = float(self._tello.get_speed_x()) / 100.0
            vy = float(self._tello.get_speed_y()) / 100.0
            vz = float(self._tello.get_speed_z()) / 100.0
            self._body_velocity = BodyVelocity(vx_m_s=vx, vy_m_s=vy, vz_m_s=vz, valid=True)
        except Exception as e:
            log.debug("get_speed: %s", e)
            self._body_velocity = BodyVelocity()

    def _refresh_telemetry(self, *, force: bool = False) -> None:
        self._telemetry.connected = self._connected
        self._telemetry.flying = self._flying
        now = time.perf_counter()
        if not self._connected:
            return
        if not force and (now - self._last_battery_refresh_s) < self._cfg.telemetry_refresh_s:
            return
        self._last_battery_refresh_s = now
        if self._cfg.mock or self._tello is None:
            self._telemetry.battery = 100
            return
        try:
            self._telemetry.battery = int(self._tello.get_battery())
        except Exception as e:
            log.warning("get_battery(): %s", e)


TelloInterface = TelloController
