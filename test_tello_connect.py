"""
Minimalny test komunikacji z Ryze Tello (bez startu).
Uruchom po połączeniu z hotspotem drona:  python test_tello_connect.py

Sprawdza: połączenie SDK, baterię, temperaturę, wysokość. Nie startuje silników.
Dodaj --takeoff, aby przetestować start+lądowanie (UWAGA: dron wzbije się!).
"""
from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--takeoff", action="store_true", help="Wykonaj start i po 3 s lądowanie (dron poleci!)")
    args = ap.parse_args()

    try:
        from djitellopy import Tello
    except ImportError:
        print("Brak djitellopy: pip install djitellopy")
        return 1

    t = Tello()
    print("Łączenie z Tello (192.168.10.1)... upewnij się, że PC jest na hotspocie TELLO-XXXX")
    try:
        t.connect()
    except Exception as e:
        print(f"BŁĄD connect(): {e}")
        print("Sprawdź: WiFi = hotspot drona, brak innego połączenia, zapora Windows nie blokuje UDP 8889/8890.")
        return 2

    try:
        bat = t.get_battery()
        print(f"Połączono. Bateria: {bat}%")
        print(f"Temperatura: {t.get_temperature()} C")
        print(f"Wysokość: {t.get_height()} cm   ToF: {t.get_distance_tof()} cm")
        if bat is not None and bat < 15:
            print("UWAGA: bateria < 15% — Tello zwykle ODMAWIA startu. Naładuj.")
    except Exception as e:
        print(f"Telemetria niepełna: {e}")

    if args.takeoff:
        try:
            print("TAKEOFF...")
            t.takeoff()
            time.sleep(3.0)
            print("LAND...")
            t.land()
        except Exception as e:
            print(f"BŁĄD startu/lądowania: {e}")
            try:
                t.land()
            except Exception:
                pass
            return 3

    t.end()
    print("OK — komunikacja działa.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
