"""
Minimal Ryze Tello communication test (no auto-start).
Run after connecting to drone hotspot:  python test_tello_connect.py

Checks: SDK connection, battery, temperature, height. Does not start motors.
Add --takeoff to test takeoff+land (WARNING: drone will fly!).
"""
from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--takeoff", action="store_true", help="Take off and land after 3 s (drone will fly!)")
    args = ap.parse_args()

    try:
        from djitellopy import Tello
    except ImportError:
        print("Missing djitellopy: pip install djitellopy")
        return 1

    t = Tello()
    print("Connecting to Tello (192.168.10.1)... ensure PC is on TELLO-XXXX hotspot")
    try:
        t.connect()
    except Exception as e:
        print(f"ERROR connect(): {e}")
        print("Check: WiFi = drone hotspot, no other connection, Windows firewall allows UDP 8889/8890.")
        return 2

    try:
        bat = t.get_battery()
        print(f"Connected. Battery: {bat}%")
        print(f"Temperature: {t.get_temperature()} C")
        print(f"Height: {t.get_height()} cm   ToF: {t.get_distance_tof()} cm")
        if bat is not None and bat < 15:
            print("WARNING: battery < 15% — Tello usually REFUSES takeoff. Charge first.")
    except Exception as e:
        print(f"Incomplete telemetry: {e}")

    if args.takeoff:
        try:
            print("TAKEOFF...")
            t.takeoff()
            time.sleep(3.0)
            print("LAND...")
            t.land()
        except Exception as e:
            print(f"ERROR takeoff/land: {e}")
            try:
                t.land()
            except Exception:
                pass
            return 3

    t.end()
    print("OK — communication works.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
