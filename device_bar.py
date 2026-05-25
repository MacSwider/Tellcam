"""Launcher wyboru źródeł TOP/SIDE (styl OBS)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

# Kolory zbliżone do ciemnego motywu (OBS)
BG = "#1e1e1e"
BG_PANEL = "#2d2d2d"
FG = "#e8e8e8"
ACCENT = "#3c5a8a"


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _apply_dark_style(root: tk.Tk) -> ttk.Style:
    root.configure(bg=BG)
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("TFrame", background=BG_PANEL)
    style.configure("Bar.TFrame", background=BG)
    style.configure("TLabel", background=BG_PANEL, foreground=FG, font=("Segoe UI", 10))
    style.configure("Bar.TLabel", background=BG, foreground=FG, font=("Segoe UI", 10))
    style.configure("TLabelframe", background=BG_PANEL, foreground=FG)
    style.configure("TLabelframe.Label", background=BG_PANEL, foreground=FG)
    style.configure(
        "TCombobox",
        fieldbackground="#3a3a3a",
        background="#3a3a3a",
        foreground=FG,
        arrowcolor=FG,
    )
    style.map("TCombobox", fieldbackground=[("readonly", "#3a3a3a")])
    style.configure("TSpinbox", fieldbackground="#3a3a3a", foreground=FG)
    style.configure("TButton", background=ACCENT, foreground=FG, padding=6, font=("Segoe UI", 10, "bold"))
    style.map("TButton", background=[("active", "#4a6aaa")])
    return style


class DeviceBarApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Tellcam — źródła TOP/SIDE")
        self.root.minsize(900, 280)
        _apply_dark_style(self.root)

        self._proc: subprocess.Popen | None = None
        self._main_script = _project_root() / "main.py"

        outer = ttk.Frame(self.root, style="Bar.TFrame", padding=8)
        outer.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(outer, text="Tellcam · wybór źródeł TOP i SIDE", style="Bar.TLabel")
        title.pack(anchor=tk.W, pady=(0, 6))
        self.top_ui = self._build_source_row(outer, "TOP")
        self.side_ui = self._build_source_row(outer, "SIDE")

        row2 = ttk.Frame(outer, style="Bar.TFrame")
        row2.pack(fill=tk.X, pady=8)

        ttk.Label(row2, text="Maks. szerokość podglądu:", style="Bar.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        self.preview_w_var = tk.IntVar(value=1400)
        ttk.Spinbox(row2, from_=0, to=3840, textvariable=self.preview_w_var, width=8).pack(side=tk.LEFT)

        ttk.Label(row2, text="  Tryb: AprilTag TOP+SIDE", style="Bar.TLabel").pack(side=tk.LEFT, padx=(16, 4))

        row3 = ttk.Frame(outer, style="Bar.TFrame")
        row3.pack(fill=tk.X, pady=8)

        self.btn_start = ttk.Button(row3, text="▶ Start", command=self._start)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 8))
        self.btn_stop = ttk.Button(row3, text="■ Stop", command=self._stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=4)

        self.status = ttk.Label(outer, text="Gotowe. Wybierz źródło i Start.", style="Bar.TLabel")
        self.status.pack(anchor=tk.W, pady=(8, 0))

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._on_source_change(self.top_ui)
        self._on_source_change(self.side_ui)
        self._avoid_same_usb_defaults()

    def _build_source_row(self, parent: ttk.Frame, label: str) -> dict:
        bar = ttk.Frame(parent, style="Bar.TFrame")
        bar.pack(fill=tk.X, pady=4)
        ttk.Label(bar, text=f"{label}:", style="Bar.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        source_combo = ttk.Combobox(
            bar,
            state="readonly",
            width=22,
            values=("Kamera USB", "Ekran (monitor)", "Okno aplikacji"),
        )
        source_combo.current(0)
        source_combo.pack(side=tk.LEFT, padx=4)
        detail_label = ttk.Label(bar, text="Urządzenie:", style="Bar.TLabel")
        detail_label.pack(side=tk.LEFT, padx=(8, 4))
        detail_combo = ttk.Combobox(bar, state="readonly", width=54)
        detail_combo.pack(side=tk.LEFT, padx=4)
        filter_var = tk.StringVar()
        filter_entry = ttk.Entry(bar, textvariable=filter_var, width=20)
        filter_entry.pack(side=tk.LEFT, padx=6)
        ttk.Button(bar, text="Odśwież", command=lambda: self._refresh_detail(ui)).pack(side=tk.LEFT, padx=8)
        ui = {
            "name": label.lower(),
            "source_combo": source_combo,
            "detail_label": detail_label,
            "detail_combo": detail_combo,
            "filter_var": filter_var,
            "filter_entry": filter_entry,
            "detail_values": [],
        }
        source_combo.bind("<<ComboboxSelected>>", lambda e: self._on_source_change(ui))
        filter_entry.bind("<Return>", lambda e: self._refresh_detail(ui))
        return ui

    def _capture_mode(self, ui: dict) -> str:
        labels = ("usb", "screen", "window")
        idx = ui["source_combo"].current()
        return labels[idx] if 0 <= idx < len(labels) else "usb"

    def _on_source_change(self, ui: dict) -> None:
        mode = self._capture_mode(ui)
        if mode == "usb":
            ui["detail_label"].configure(text="Kamera:")
        elif mode == "screen":
            ui["detail_label"].configure(text="Monitor:")
        else:
            ui["detail_label"].configure(text="Okno:")
        self._refresh_detail(ui)

    def _refresh_detail(self, ui: dict) -> None:
        mode = self._capture_mode(ui)
        try:
            from video_source import list_mss_monitors_for_ui, list_window_titles, probe_usb_camera_indices
        except Exception as e:
            messagebox.showerror("Tellcam", str(e))
            return

        if mode == "usb":
            ids = probe_usb_camera_indices()
            if ids:
                ui["detail_values"] = ids
                ui["detail_combo"]["values"] = [f"Kamera {i}" for i in ids]
                if ui["name"] == "side" and len(ids) > 1:
                    ui["detail_combo"].current(1)
                else:
                    ui["detail_combo"].current(0)
            else:
                ui["detail_values"] = []
                ui["detail_combo"]["values"] = ["(brak wykrytych kamer USB)"]
                ui["detail_combo"].set(ui["detail_combo"]["values"][0])
        elif mode == "screen":
            try:
                mons = list_mss_monitors_for_ui()
            except Exception as e:
                messagebox.showerror("mss", str(e))
                ui["detail_combo"]["values"] = []
                return
            ui["detail_values"] = [m[0] for m in mons]
            ui["detail_combo"]["values"] = [m[1] for m in mons]
            ui["detail_combo"].current(min(1, len(mons) - 1) if len(mons) > 1 else 0)
        else:
            try:
                titles = list_window_titles(300)
            except RuntimeError as e:
                messagebox.showerror("Okna", str(e))
                ui["detail_combo"]["values"] = []
                return
            filt = ui["filter_var"].get().strip().lower()
            if filt:
                titles = [t for t in titles if filt in t.lower()]
            if titles:
                ui["detail_values"] = titles
                display = [t[:80] + "…" if len(t) > 80 else t for t in titles]
                ui["detail_combo"]["values"] = display
                ui["detail_combo"].current(0)
            else:
                ui["detail_values"] = []
                ui["detail_combo"]["values"] = ["(brak okien — odśwież lub zmień filtr)"]
                ui["detail_combo"].set(ui["detail_combo"]["values"][0])
        self._avoid_same_usb_defaults()

    def _avoid_same_usb_defaults(self) -> None:
        top_mode = self._capture_mode(self.top_ui)
        side_mode = self._capture_mode(self.side_ui)
        if top_mode != "usb" or side_mode != "usb":
            return
        top_values = self.top_ui["detail_values"]
        side_values = self.side_ui["detail_values"]
        if not top_values or not side_values:
            return
        top_cur = self.top_ui["detail_combo"].current()
        side_cur = self.side_ui["detail_combo"].current()
        if top_cur < 0:
            top_cur = 0
        if side_cur < 0:
            side_cur = 0
        if top_values[top_cur] == side_values[side_cur] and len(side_values) > 1:
            for i, cam_id in enumerate(side_values):
                if cam_id != top_values[top_cur]:
                    self.side_ui["detail_combo"].current(i)
                    break

    def _append_source_args(self, argv: list[str], ui: dict) -> None:
        mode = self._capture_mode(ui)
        prefix = ui["name"]
        argv += [f"--{prefix}-capture", mode]
        values = ui["detail_values"]
        combo = ui["detail_combo"]

        if mode == "usb":
            if not values:
                raise ValueError("Brak kamer USB — podłącz urządzenie i kliknij Odśwież.")
            cur = combo.current()
            if cur < 0 or cur >= len(values):
                cur = 0
            argv += [f"--{prefix}-camera-index", str(values[cur])]
        elif mode == "screen":
            if not values:
                raise ValueError("Brak monitorów w konfiguracji mss.")
            cur = combo.current()
            if cur < 0 or cur >= len(values):
                cur = min(1, len(values) - 1) if len(values) > 1 else 0
            argv += [f"--{prefix}-screen-monitor", str(values[cur])]
        else:
            title = values[combo.current()] if values else ""
            if not title or title.startswith("("):
                raise ValueError("Wybierz okno z listy.")
            argv += [f"--{prefix}-window-title", title]

    def _build_argv(self) -> list[str]:
        exe = sys.executable
        argv = [exe, str(self._main_script)]

        self._append_source_args(argv, self.top_ui)
        self._append_source_args(argv, self.side_ui)
        pw = int(self.preview_w_var.get())
        if pw < 0:
            raise ValueError("Maks. szerokość podglądu nie może być ujemna.")
        argv += ["--preview-max-width", str(pw)]

        return argv

    def _start(self) -> None:
        if self._proc and self._proc.poll() is None:
            messagebox.showinfo("Tellcam", "Proces już działa — najpierw Stop.")
            return
        try:
            argv = self._build_argv()
        except ValueError as e:
            messagebox.showwarning("Tellcam", str(e))
            return
        except (tk.TclError, TypeError) as e:
            messagebox.showwarning("Tellcam", f"Niepoprawna wartość pola: {e}")
            return

        self.status.configure(text="Uruchamianie: " + " ".join(argv[2:]))
        try:
            self._proc = subprocess.Popen(
                argv,
                cwd=str(_project_root()),
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
            )
        except Exception as e:
            messagebox.showerror("Tellcam", f"Nie udało się uruchomić:\n{e}")
            self._proc = None
            return

        self.btn_start.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        self._poll_proc()

    def _poll_proc(self) -> None:
        if self._proc is None:
            return
        code = self._proc.poll()
        if code is None:
            self.root.after(500, self._poll_proc)
            return
        self._proc = None
        self.btn_start.configure(state=tk.NORMAL)
        self.btn_stop.configure(state=tk.DISABLED)
        self.status.configure(text=f"Proces zakończony (kod {code}).")

    def _stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self.status.configure(text="Wysłano zatrzymanie procesu.")
        self.btn_start.configure(state=tk.NORMAL)
        self.btn_stop.configure(state=tk.DISABLED)

    def _on_close(self) -> None:
        if self._proc and self._proc.poll() is None:
            if not messagebox.askokcancel("Tellcam", "Trwa przetwarzanie. Zamknąć launcher i zatrzymać podgląd?"):
                return
            self._proc.terminate()
        self.root.destroy()

    def run(self) -> int:
        self.root.mainloop()
        return 0


def run_device_bar() -> int:
    app = DeviceBarApp()
    return app.run()
