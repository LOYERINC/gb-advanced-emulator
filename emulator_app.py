"""Small desktop front end for the GBA emulator learning project."""

from __future__ import annotations

import argparse
import tkinter as tk
import time
import os
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
try:
    from PIL import Image, ImageTk
except ImportError:
    Image = ImageTk = None

from gba.audio import AudioOutput
from gba.arm7tdmi import Arm7Tdmi, UnsupportedInstruction
try:
    from gba._native_cpu import run_thumb_batch as _run_native_thumb_batch
    from gba._native_cpu import run_arm_batch as _run_native_arm_batch
except ImportError:
    _run_native_thumb_batch = None
    _run_native_arm_batch = None
from gba.cartridge import Cartridge, CartridgeError
from gba.memory import MemoryBus
from gba.video import GbaVideo


class EmulatorWindow:
    SCREEN_WIDTH = 240
    SCREEN_HEIGHT = 160
    SCALE = 3
    RUN_BATCH_SIZE = 24000
    UI_FRAME_INTERVAL = 1 / 30
    CYCLE_UPDATE_QUANTUM = 4096
    SAVE_TYPE_OPTIONS = {
        "Not identified": None,
        "SRAM · 32 KiB": "SRAM",
        "Flash · 64 KiB": "FLASH",
        "Flash 1M · 128 KiB": "FLASH1M",
        "EEPROM · auto size": "EEPROM",
    }

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        settings_root = Path(
            os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        )
        self.settings_dir = settings_root / "GbaEmulatorLearningBuild"
        self.last_rom_file = self.settings_dir / "last_rom.txt"
        self.cartridge: Cartridge | None = None
        self.bios_data: bytes | None = None
        self.bios_path: Path | None = None
        self.cpu: Arm7Tdmi | None = None
        self.video: GbaVideo | None = None
        self.screen_photo = None
        self._screen_image_item: int | None = None
        self._screen_placeholder_items: tuple[int, int, int] | None = None
        self._last_drawn_video_generation: int | None = None
        self.running = False
        self.run_after_id: str | None = None
        self.instructions_executed = 0
        self._last_ui_refresh = 0.0
        self.audio_output = AudioOutput()
        self.key_map = {
            "z": "a",
            "x": "b",
            "backspace": "select",
            "return": "start",
            "right": "right",
            "left": "left",
            "up": "up",
            "down": "down",
            "a": "l",
            "s": "r",
        }
        self.root.title("GBA Emulator — learning build")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(920, max(640, screen_width - 80))
        window_height = min(720, max(480, screen_height - 100))
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.minsize(min(760, screen_width), min(480, screen_height))
        self.root.configure(bg="#111820")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#111820")
        style.configure("Card.TFrame", background="#1b2732")
        style.configure("TLabel", background="#111820", foreground="#e8edf2", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 19, "bold"), foreground="#f3f7fa")
        style.configure("Sub.TLabel", foreground="#a9bac8", font=("Segoe UI", 10))
        style.configure(
            "Notice.TLabel",
            background="#15212a",
            foreground="#ffd479",
            font=("Segoe UI", 10),
            padding=10,
        )
        style.configure("CardTitle.TLabel", background="#1b2732", foreground="#8fc7ff", font=("Segoe UI", 10, "bold"))
        style.configure("Value.TLabel", background="#1b2732", foreground="#f3f7fa", font=("Consolas", 10))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(16, 10))
        style.configure("TButton", padding=(12, 9))

        outer = ttk.Frame(root, padding=24)
        outer.pack(fill="both", expand=True)

        heading = ttk.Frame(outer)
        heading.pack(fill="x", pady=(0, 18))
        ttk.Label(heading, text="GBA Emulator", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            heading,
            text="Learning build · ROM loader, memory bus, and early ARM CPU stepper",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        display_card = ttk.Frame(body, style="Card.TFrame", padding=16)
        display_card.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        ttk.Label(display_card, text="GAME SCREEN", style="CardTitle.TLabel").pack(anchor="w", pady=(0, 12))
        self.screen = tk.Canvas(
            display_card,
            width=self.SCREEN_WIDTH * self.SCALE,
            height=self.SCREEN_HEIGHT * self.SCALE,
            background="#070d12",
            highlightthickness=1,
            highlightbackground="#34495a",
        )
        self.screen.pack(expand=True)
        self._screen_image_item = self.screen.create_image(
            0, 0, anchor="nw", state="hidden"
        )
        self._draw_placeholder("EMULATOR READY", "Mode 0 tiles and bitmap graphics are supported")

        info_panel = ttk.Frame(body)
        self.info_panel = info_panel
        info_panel.grid(row=0, column=1, sticky="nsew")
        self.info_canvas = tk.Canvas(
            info_panel,
            background="#1b2732",
            highlightthickness=0,
            borderwidth=0,
        )
        info_scrollbar = ttk.Scrollbar(info_panel, orient="vertical", command=self.info_canvas.yview)
        self.info_canvas.configure(yscrollcommand=info_scrollbar.set)
        info_scrollbar.pack(side="right", fill="y")
        self.info_canvas.pack(side="left", fill="both", expand=True)
        info_card = ttk.Frame(self.info_canvas, style="Card.TFrame", padding=18)
        info_card_window = self.info_canvas.create_window((0, 0), window=info_card, anchor="nw")
        info_card.bind(
            "<Configure>",
            lambda _event: self.info_canvas.configure(scrollregion=self.info_canvas.bbox("all")),
        )
        self.info_canvas.bind(
            "<Configure>",
            lambda event: self.info_canvas.itemconfigure(info_card_window, width=event.width),
        )
        # Keep the two actions needed to start a game outside the scrollable
        # cartridge details. This makes them reachable even on short screens.
        quick_actions = ttk.Frame(info_panel, style="Card.TFrame", padding=(10, 8))
        quick_actions.pack(side="bottom", fill="x")
        self.open_rom_button = ttk.Button(
            quick_actions, text="Open ROM…", style="Accent.TButton", command=self.open_rom
        )
        self.open_rom_button.pack(fill="x")
        self.run_button = ttk.Button(
            quick_actions, text="Run", command=self.run_cpu, state="disabled"
        )
        self.run_button.pack(fill="x", pady=(6, 0))
        ttk.Button(
            quick_actions,
            text="Load BIOS… (optional)",
            command=self.load_bios,
        ).pack(fill="x", pady=(6, 0))
        # Keep emulator errors visible even when the cartridge details are
        # scrolled. In particular, a white screen must not hide the CPU's
        # address and opcode when an instruction is not implemented yet.
        self.warning_var = tk.StringVar(
            value="Open a ROM you have the right to use to inspect its cartridge header."
        )
        self.warning_label = ttk.Label(
            info_panel,
            textvariable=self.warning_var,
            style="Notice.TLabel",
            wraplength=220,
            justify="left",
        )
        self.warning_label.pack(side="bottom", fill="x", padx=4, pady=(0, 4))
        self.warning_label.bind("<Configure>", self._fit_warning_text)
        self.root.bind_all("<MouseWheel>", self._scroll_info_panel, add="+")
        self.root.bind_all("<Button-4>", self._scroll_info_panel, add="+")
        self.root.bind_all("<Button-5>", self._scroll_info_panel, add="+")
        self.root.bind_all("<Prior>", self._scroll_info_panel, add="+")
        self.root.bind_all("<Next>", self._scroll_info_panel, add="+")
        ttk.Label(info_card, text="CARTRIDGE", style="CardTitle.TLabel").pack(anchor="w", pady=(0, 14))
        self.info_vars: dict[str, tk.StringVar] = {}
        self.save_type_var = tk.StringVar(value="Not identified")
        self.save_type_combo: ttk.Combobox | None = None
        for label, key in (
            ("Status", "status"),
            ("Title", "title"),
            ("Game code", "game_code"),
            ("Maker", "maker"),
            ("Version", "version"),
            ("ROM size", "size"),
            ("Save memory", "save"),
            ("Header", "header"),
        ):
            row = ttk.Frame(info_card, style="Card.TFrame")
            row.pack(fill="x", pady=5)
            ttk.Label(row, text=label, style="Sub.TLabel", width=11).pack(side="left", anchor="nw")
            if key == "save":
                self.save_type_combo = ttk.Combobox(
                    row,
                    textvariable=self.save_type_var,
                    values=tuple(self.SAVE_TYPE_OPTIONS),
                    state="disabled",
                    width=20,
                )
                self.save_type_combo.pack(side="left", fill="x", expand=True, anchor="w")
                self.save_type_combo.bind("<<ComboboxSelected>>", self._save_type_selected)
            else:
                value = tk.StringVar(value="—")
                self.info_vars[key] = value
                ttk.Label(row, textvariable=value, style="Value.TLabel", wraplength=220, justify="left").pack(
                    side="left", fill="x", expand=True, anchor="w"
                )

        controls = ttk.Frame(info_card, style="Card.TFrame")
        controls.pack(fill="x", side="bottom", pady=(16, 0))
        bios_controls = ttk.Frame(controls, style="Card.TFrame")
        bios_controls.pack(fill="x", pady=(8, 0))
        ttk.Label(
            bios_controls,
            text="A BIOS is optional. You can load a ROM without one.",
            style="Sub.TLabel",
            wraplength=230,
            justify="left",
        ).pack(anchor="w")
        self.unload_bios_button = ttk.Button(
            bios_controls, text="Unload", command=self.unload_bios, state="disabled"
        )
        self.unload_bios_button.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.reset_button = ttk.Button(controls, text="Reset game", command=self.reset_game, state="disabled")
        self.reset_button.pack(fill="x", pady=(8, 0))
        self.step_button = ttk.Button(controls, text="Step CPU once", command=self.step_cpu, state="disabled")
        self.step_button.pack(fill="x", pady=(8, 0))
        self.pause_button = ttk.Button(controls, text="Pause", command=self.pause_cpu, state="disabled")
        self.pause_button.pack(fill="x", pady=(8, 0))
        ttk.Button(controls, text="Eject", command=self.eject_rom).pack(fill="x", pady=(8, 0))

        footer = ttk.Label(
            outer,
            text="Controls: arrows move · Z=A · X=B · Enter=Start · Backspace=Select · A/S=L/R",
            style="Sub.TLabel",
        )
        footer.pack(anchor="w", pady=(14, 0))
        # Bind at the application level so game controls keep working after
        # the user clicks Run, a BIOS button, or any other widget.
        self.root.bind_all("<KeyPress>", self._key_down, add="+")
        self.root.bind_all("<KeyRelease>", self._key_up, add="+")
        self.root.after_idle(self._restore_last_rom)

    def _scroll_info_panel(self, event: tk.Event) -> str | None:
        """Scroll the cartridge/control panel when the pointer is over it."""
        pointer_x, pointer_y = self.root.winfo_pointerxy()
        # Check the whole panel, including buttons, because Tk sends wheel
        # events to whichever child is under the pointer.
        left = self.info_panel.winfo_rootx()
        top = self.info_panel.winfo_rooty()
        if not (
            left <= pointer_x < left + self.info_panel.winfo_width()
            and top <= pointer_y < top + self.info_panel.winfo_height()
        ):
            return None
        if getattr(event, "keysym", "") == "Prior":
            self.info_canvas.yview_scroll(-8, "units")
        elif getattr(event, "keysym", "") == "Next":
            self.info_canvas.yview_scroll(8, "units")
        elif getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            self.info_canvas.yview_scroll(-3, "units")
        else:
            self.info_canvas.yview_scroll(3, "units")
        return "break"

    def _fit_warning_text(self, event: tk.Event) -> None:
        """Keep status and CPU error text within the panel's actual width."""
        wraplength = max(120, event.width - 16)
        if self.warning_label.cget("wraplength") != wraplength:
            self.warning_label.configure(wraplength=wraplength)

    def _draw_placeholder(self, title: str, subtitle: str) -> None:
        width = self.SCREEN_WIDTH * self.SCALE
        height = self.SCREEN_HEIGHT * self.SCALE
        if self._screen_image_item is not None:
            self.screen.itemconfigure(self._screen_image_item, image="", state="hidden")
        self.screen_photo = None
        self._last_drawn_video_generation = None
        if self._screen_placeholder_items is None:
            border = self.screen.create_rectangle(
                20, 20, width - 20, height - 20, outline="#244056", width=2
            )
            heading = self.screen.create_text(
                width // 2,
                height // 2 - 16,
                fill="#8fc7ff",
                font=("Consolas", 18, "bold"),
            )
            detail = self.screen.create_text(
                width // 2,
                height // 2 + 22,
                fill="#90a4b3",
                font=("Segoe UI", 11),
                width=width - 64,
                justify="center",
            )
            self._screen_placeholder_items = (border, heading, detail)
        border, heading, detail = self._screen_placeholder_items
        self.screen.itemconfigure(border, state="normal")
        self.screen.itemconfigure(heading, text=title, state="normal")
        self.screen.itemconfigure(detail, text=subtitle, state="normal")

    def open_rom(self, selected_path: str | Path | None = None) -> None:
        selected = str(selected_path) if selected_path is not None else filedialog.askopenfilename(
            title="Open a GBA ROM",
            filetypes=(("GBA ROMs and ZIP archives", "*.gba *.agb *.bin *.zip"), ("All files", "*.*")),
        )
        if not selected:
            return

        try:
            cartridge = Cartridge.load(selected)
        except (OSError, CartridgeError) as exc:
            self.warning_var.set(f"Could not load ROM: {exc}")
            return

        try:
            if self.cpu is not None:
                self.cpu.bus.flush_save()
            bus = MemoryBus(cartridge, bios_data=self.bios_data)
        except OSError as exc:
            self.warning_var.set(f"Could not open this ROM or its save file: {exc}")
            return

        self._stop_execution()
        self.audio_output.reset()
        self.cartridge = cartridge
        self.cpu = Arm7Tdmi(bus, start_at_bios=self.bios_data is not None)
        self.video = GbaVideo(bus)
        self.reset_button.configure(state="normal")
        self.step_button.configure(state="normal")
        self.run_button.configure(state="normal")
        self.pause_button.configure(state="disabled")
        self.instructions_executed = 0
        self.info_vars["status"].set("ROM loaded")
        self.info_vars["title"].set(cartridge.title or "(no title)")
        self.info_vars["game_code"].set(cartridge.game_code or "(unknown)")
        self.info_vars["maker"].set(cartridge.maker_code or "(unknown)")
        self.info_vars["version"].set(str(cartridge.version))
        self.info_vars["size"].set(f"{len(cartridge.data):,} bytes")
        save_label = next(
            label for label, save_type in self.SAVE_TYPE_OPTIONS.items()
            if save_type == cartridge.save_type
        )
        if cartridge.save_type == "EEPROM":
            save_label = "EEPROM · auto size"
        self.save_type_var.set(save_label)
        self.save_type_combo.configure(state="disabled" if cartridge.save_type else "readonly")
        self.info_vars["header"].set("Valid" if not cartridge.header_warnings else "Warnings")
        self.root.title(f"{cartridge.title or Path(selected).name} — GBA Emulator")
        first_word = bus.read32(0x08000000)
        if self.bios_data is None:
            warning_text = "ROM is loaded. Step CPU once to execute a supported ARM or Thumb instruction."
        else:
            warning_text = "ROM is loaded. The user BIOS reset sequence will run when you press Run."
        if cartridge.save_type in ("SRAM", "FLASH", "FLASH1M", "EEPROM"):
            warning_text += f" Save data will be kept in {cartridge.save_path.name}."
        else:
            warning_text += " No save-hardware marker was found in the ROM."
        if cartridge.header_warnings:
            warning_text += " Header: " + "; ".join(cartridge.header_warnings)
        if self.bios_path is not None:
            warning_text += f" Using BIOS {self.bios_path.name}."
        self.warning_var.set(f"{warning_text}\nFirst ROM word: 0x{first_word:08X}")
        try:
            self.settings_dir.mkdir(parents=True, exist_ok=True)
            self.last_rom_file.write_text(str(Path(selected).resolve()), encoding="utf-8")
        except OSError:
            # Remembering the path is optional; it must not prevent loading a game.
            pass
        self._draw_frame()

    def _restore_last_rom(self) -> None:
        """Reopen the last selected ROM after an app restart, without starting it."""
        try:
            selected = self.last_rom_file.read_text(encoding="utf-8").strip()
        except OSError:
            return
        if not selected:
            return
        if not Path(selected).is_file():
            self.warning_var.set(
                "The last ROM file is no longer at its saved location. Use Open ROM… to choose it again."
            )
            return
        self.open_rom(selected)

    def load_bios(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose your GBA BIOS image",
            filetypes=(("BIOS image", "*.bin *.gba *.bios"), ("All files", "*.*")),
        )
        if not selected:
            return
        try:
            data = Path(selected).read_bytes()
        except OSError as exc:
            messagebox.showerror("Could not read BIOS", str(exc), parent=self.root)
            return
        if len(data) != 16 * 1024:
            messagebox.showerror(
                "Invalid BIOS size",
                f"A GBA BIOS image must be exactly 16 KiB; this file is {len(data):,} bytes.",
                parent=self.root,
            )
            return
        self._set_bios(data, Path(selected))

    def unload_bios(self) -> None:
        self._set_bios(None, None)

    def _set_bios(self, data: bytes | None, path: Path | None) -> None:
        if self.cartridge is not None:
            try:
                if self.cpu is not None:
                    self.cpu.bus.flush_save()
                bus = MemoryBus(self.cartridge, bios_data=data)
            except OSError as exc:
                messagebox.showerror("Could not change BIOS", str(exc), parent=self.root)
                return
            self._stop_execution()
            self.audio_output.reset()
            self.cpu = Arm7Tdmi(bus, start_at_bios=data is not None)
            self.video = GbaVideo(bus)
            self.instructions_executed = 0
            self.info_vars["status"].set("Game restarted with BIOS setting")
            self.step_button.configure(state="normal")
            self.run_button.configure(state="normal")
            self.pause_button.configure(state="disabled")
            self._draw_frame()
        self.bios_data = data
        self.bios_path = path
        self.unload_bios_button.configure(state="normal" if data is not None else "disabled")
        if data is None:
            self.warning_var.set("User BIOS unloaded. The built-in high-level BIOS call replacements are active.")
        elif self.cartridge is None:
            self.warning_var.set(f"BIOS loaded from {path.name}. Open a ROM to use it.")
        else:
            self.warning_var.set(
                f"BIOS loaded from {path.name}. The game restarted at BIOS reset; its startup and SWI code will run."
            )

    def reset_game(self) -> None:
        """Restart the loaded ROM while keeping its selected save hardware."""
        if self.cartridge is None:
            return
        try:
            if self.cpu is not None:
                self.cpu.bus.flush_save()
            bus = MemoryBus(self.cartridge, bios_data=self.bios_data)
        except OSError as exc:
            messagebox.showerror("Could not reset game", str(exc), parent=self.root)
            return

        self._stop_execution()
        self.audio_output.reset()
        self.cpu = Arm7Tdmi(bus, start_at_bios=self.bios_data is not None)
        self.video = GbaVideo(bus)
        self.instructions_executed = 0
        self.info_vars["status"].set("Game reset")
        self.step_button.configure(state="normal")
        self.run_button.configure(state="normal")
        self.pause_button.configure(state="disabled")
        if self.bios_data is None:
            restart_detail = "the ROM entry point"
        else:
            restart_detail = "the BIOS reset vector"
        self.warning_var.set(
            f"Restarted {self.cartridge.title or self.cartridge.path.name} from {restart_detail}."
        )
        self._draw_frame()

    def _save_type_selected(self, _event: tk.Event | None = None) -> None:
        """Apply a manual backup-memory choice when the ROM lacks a marker."""
        if self.cartridge is None or self.cpu is None or self.save_type_combo is None:
            return
        label = self.save_type_var.get()
        if label not in self.SAVE_TYPE_OPTIONS:
            return
        new_save_type = self.SAVE_TYPE_OPTIONS[label]
        if new_save_type == self.cartridge.save_type:
            return

        previous_label = next(
            option for option, kind in self.SAVE_TYPE_OPTIONS.items()
            if kind == self.cartridge.save_type
        )

        save_path = self.cartridge.save_path
        if save_path is not None and save_path.exists():
            accepted = messagebox.askyesno(
                "Change save memory",
                "This ROM already has a .sav file. Changing save memory resets the game and may reinterpret or resize that file. Continue?",
                parent=self.root,
            )
            if not accepted:
                self.save_type_var.set(previous_label)
                return

        try:
            self.cpu.bus.flush_save()
            cartridge = replace(self.cartridge, save_type=new_save_type)
            bus = MemoryBus(cartridge, bios_data=self.bios_data)
        except OSError as exc:
            messagebox.showerror("Save memory error", str(exc), parent=self.root)
            self.save_type_var.set(previous_label)
            return

        self._stop_execution()
        self.audio_output.reset()
        self.cartridge = cartridge
        self.cpu = Arm7Tdmi(bus, start_at_bios=self.bios_data is not None)
        self.video = GbaVideo(bus)
        self.instructions_executed = 0
        self.info_vars["status"].set("ROM reset for save type")
        self.save_type_combo.configure(state="readonly")
        self.warning_var.set(
            f"Save memory set to {label}. The game restarted from its ROM entry point."
        )
        self.step_button.configure(state="normal")
        self.run_button.configure(state="normal")
        self.pause_button.configure(state="disabled")
        self._draw_frame()

    def step_cpu(self) -> None:
        if self.cpu is None or self.running:
            return
        try:
            result = self.cpu.step()
            if not self.cpu.stopped:
                cycles = result.cycles
                if cycles is None:
                    cycles = self._estimated_cycles(result.operation, result.instruction)
                self.cpu.bus.advance_cycles(cycles + result.fetch_cycles + result.data_cycles)
        except UnsupportedInstruction as exc:
            self._submit_audio()
            self.info_vars["status"].set("CPU paused")
            self.warning_var.set(f"CPU reached an instruction not implemented yet: {exc}")
            return

        self.info_vars["status"].set("CPU stepped")
        self._submit_audio()
        self.warning_var.set(
            f"Executed {result.operation} at 0x{result.address:08X}. "
            f"Next PC: 0x{self.cpu.registers[15]:08X}. "
            f"Display mode: {self.video.mode if self.video else '—'}."
        )
        self._draw_frame()

    def run_cpu(self) -> None:
        if self.cpu is None or self.running:
            return
        self.running = True
        self.run_button.configure(state="disabled")
        self.step_button.configure(state="disabled")
        self.pause_button.configure(state="normal")
        self.info_vars["status"].set("Running")
        self._last_ui_refresh = 0.0
        self.run_after_id = self.root.after(1, self._run_batch)

    def _run_batch(self) -> None:
        self.run_after_id = None
        if not self.running or self.cpu is None:
            return
        executed = 0
        pending_cycles = 0
        stopped_idle = False
        idle_wait_cycles = 0
        try:
            while executed < self.RUN_BATCH_SIZE:
                if not self.cpu.thumb_state and _run_native_arm_batch is not None:
                    native_count, native_cycles, _native_stopped = _run_native_arm_batch(
                        self.cpu,
                        min(8192, self.RUN_BATCH_SIZE - executed),
                        self.CYCLE_UPDATE_QUANTUM,
                        pending_cycles,
                    )
                    if native_count:
                        executed += native_count
                        pending_cycles = native_cycles
                        continue
                    if pending_cycles:
                        self.cpu.bus.advance_cycles(pending_cycles)
                        pending_cycles = 0
                if _run_native_thumb_batch is not None:
                    native_count, native_cycles, _native_stopped = _run_native_thumb_batch(
                        self.cpu,
                        min(8192, self.RUN_BATCH_SIZE - executed),
                        self.CYCLE_UPDATE_QUANTUM,
                        pending_cycles,
                    )
                    if native_count:
                        executed += native_count
                        pending_cycles = native_cycles
                        continue
                    if pending_cycles:
                        self.cpu.bus.advance_cycles(pending_cycles)
                        pending_cycles = 0
                result = self.cpu.step()
                if result.operation == "STOP idle":
                    stopped_idle = True
                    break
                if result.operation in ("HALT idle", "IntrWait idle"):
                    if pending_cycles:
                        self.cpu.bus.advance_cycles(pending_cycles)
                        pending_cycles = 0
                    wait_mask = (
                        self.cpu._intr_wait_mask
                        if result.operation == "IntrWait idle"
                        else self.cpu.bus.read16(0x04000200)
                    )
                    idle_wait_cycles = self.cpu.bus.cycles_until_next_interrupt(wait_mask)
                    self.cpu.bus.advance_cycles(idle_wait_cycles)
                    break
                if not self.cpu.stopped:
                    cycles = result.cycles
                    if cycles is None:
                        cycles = self._estimated_cycles(result.operation, result.instruction)
                    pending_cycles += cycles + result.fetch_cycles + result.data_cycles
                    if pending_cycles >= self.CYCLE_UPDATE_QUANTUM:
                        self.cpu.bus.advance_cycles(pending_cycles)
                        pending_cycles = 0
                executed += 1
        except UnsupportedInstruction as exc:
            if pending_cycles:
                self.cpu.bus.advance_cycles(pending_cycles)
            self.instructions_executed += executed
            self._submit_audio()
            self._stop_execution()
            self.info_vars["status"].set("Paused: unsupported instruction")
            cpu_mode = "Thumb" if self.cpu.thumb_state else "ARM"
            self.warning_var.set(
                f"The CPU stopped in {cpu_mode} mode at 0x{exc.address:08X} on instruction "
                f"0x{exc.instruction:08X}. This instruction is not implemented yet."
            )
            self._draw_frame()
            return

        if pending_cycles:
            self.cpu.bus.advance_cycles(pending_cycles)

        self.instructions_executed += executed
        if not stopped_idle:
            now = time.monotonic()
            if now - self._last_ui_refresh >= self.UI_FRAME_INTERVAL:
                self._submit_audio()
                self._draw_frame()
                if idle_wait_cycles:
                    scanlines = max(1, round(idle_wait_cycles / 1232))
                    self.warning_var.set(
                        f"CPU is waiting for an interrupt; advanced about {scanlines} scanlines. "
                        f"PC = 0x{self.cpu.registers[15]:08X}."
                    )
                else:
                    self.warning_var.set(
                        f"Ran {self.instructions_executed:,} instructions. "
                        f"PC = 0x{self.cpu.registers[15]:08X}. "
                        "Display and audio timing use rough instruction-cycle estimates."
                    )
                self._last_ui_refresh = now
        else:
            self.warning_var.set(
                "The game is in STOP sleep. Press a button selected in its keypad-interrupt settings to wake it."
            )
        if stopped_idle:
            delay_ms = 50
        elif idle_wait_cycles:
            # Convert skipped emulated time back to wall time to keep idle game
            # loops from burning a full CPU core while preserving pacing.
            delay_ms = max(1, min(50, round(idle_wait_cycles * 1000 / 16_777_216)))
        else:
            delay_ms = 1
        self.run_after_id = self.root.after(delay_ms, self._run_batch)

    @staticmethod
    def _estimated_cycles(operation: str, instruction: int) -> int:
        """Small interim timing estimate for timer-driven game logic."""
        # These common ALU results all use the ordinary two-cycle estimate.
        # Keep their hot path out of the slower load/store/branch classifiers.
        if operation in ("CMP", "ADD", "MOV", "ORR", "SUB", "LSR", "LSL", "AND"):
            return 2
        if operation in ("HALT idle", "IntrWait idle"):
            # The CPU is asleep while LCD/timer hardware keeps advancing.
            # Advance roughly one scanline at a time, then recheck interrupts;
            # stepping this idle loop every 32 cycles wastes CPU during boot.
            return 1232
        if operation == "condition skipped":
            return 1
        if operation == "BL prefix" or operation.endswith("not taken"):
            return 1
        if operation.endswith("taken"):
            return 3
        if operation == "IRQ ENTRY" or operation in ("B", "BL") or operation.startswith(("B ", "BL", "BX", "SWI")):
            return 3
        if operation.startswith("LDR"):
            return 3
        if operation.startswith("STR"):
            return 2
        if operation.startswith(("MRS", "MSR")):
            return 1
        if operation in ("SWP", "SWPB"):
            return 4
        if operation in ("PUSH", "POP"):
            # Thumb's list occupies bits 0-7; bit 8 adds LR/PC.
            count = (instruction & 0x01FF).bit_count()
            return max(2, count + (2 if operation == "POP" else 1))
        if operation.startswith(("LDM", "STM")):
            # ARM uses a 16-bit list; Thumb LDMIA/STMIA use only bits 0-7.
            mask = instruction & (0x00FF if operation.endswith("(Thumb)") else 0xFFFF)
            count = mask.bit_count() or 1  # Empty ARMv4T lists transfer the PC.
            return count + (2 if operation.startswith("LDM") else 1)
        # A normal ARM/Thumb instruction takes a sequential fetch and an
        # internal execute cycle; memory and branch cases above have their own
        # estimates.
        return 2

    def _stop_execution(self) -> None:
        self.running = False
        if self.run_after_id is not None:
            try:
                self.root.after_cancel(self.run_after_id)
            except tk.TclError:
                pass
            self.run_after_id = None
        self.run_button.configure(state="normal" if self.cpu else "disabled")
        self.step_button.configure(state="normal" if self.cpu else "disabled")
        self.pause_button.configure(state="disabled")

    def _submit_audio(self) -> None:
        if self.cpu is not None:
            self.audio_output.submit(self.cpu.bus.drain_audio_samples())

    def pause_cpu(self) -> None:
        if not self.running:
            return
        self._stop_execution()
        self._submit_audio()
        self.info_vars["status"].set("Paused")

    def _key_down(self, event: tk.Event) -> None:
        if self.cpu is not None:
            button = self.key_map.get(event.keysym.lower())
            if button:
                self.cpu.bus.set_key(button, True)

    def _key_up(self, event: tk.Event) -> None:
        if self.cpu is not None:
            button = self.key_map.get(event.keysym.lower())
            if button:
                self.cpu.bus.set_key(button, False)

    def _draw_frame(self) -> None:
        if self.video is None:
            self._draw_placeholder("EMULATOR READY", "Open a ROM to begin")
            return
        if not self.video.display_output_enabled:
            mode = self.video.mode
            subtitle = "The game has not enabled a display layer" if mode in (0, 1, 2) else "The game has not enabled BG2 or objects"
            self._draw_placeholder(f"DISPLAY MODE {mode}", subtitle)
            return
        generation = self.cpu.bus.video_generation if self.cpu is not None else None
        if generation is not None and generation == self._last_drawn_video_generation:
            return
        rgb = self.video.frame_rgb()
        if Image is not None and ImageTk is not None:
            image = Image.frombytes("RGB", (self.SCREEN_WIDTH, self.SCREEN_HEIGHT), rgb)
            image = image.resize(
                (self.SCREEN_WIDTH * self.SCALE, self.SCREEN_HEIGHT * self.SCALE),
                Image.Resampling.NEAREST,
            )
            if isinstance(self.screen_photo, ImageTk.PhotoImage):
                self.screen_photo.paste(image)
            else:
                self.screen_photo = ImageTk.PhotoImage(image, master=self.root)
        else:
            image = tk.PhotoImage(width=self.SCREEN_WIDTH, height=self.SCREEN_HEIGHT)
            bytes_per_row = self.SCREEN_WIDTH * 3
            for y in range(self.SCREEN_HEIGHT):
                start = y * bytes_per_row
                colors = (
                    f"#{rgb[index]:02x}{rgb[index + 1]:02x}{rgb[index + 2]:02x}"
                    for index in range(start, start + bytes_per_row, 3)
                )
                image.put("{" + " ".join(colors) + "}", to=(0, y))
            self.screen_photo = image.zoom(self.SCALE, self.SCALE)
        if self._screen_image_item is not None:
            self.screen.itemconfigure(self._screen_image_item, image=self.screen_photo, state="normal")
        if self._screen_placeholder_items is not None:
            for item in self._screen_placeholder_items:
                self.screen.itemconfigure(item, state="hidden")
        self._last_drawn_video_generation = generation

    def eject_rom(self) -> None:
        if self.cpu is not None:
            try:
                self.cpu.bus.flush_save()
            except OSError as exc:
                self.warning_var.set(f"Could not write the save file: {exc}")
                return
        self._stop_execution()
        self.audio_output.reset()
        try:
            self.last_rom_file.unlink(missing_ok=True)
        except OSError:
            pass
        self.cartridge = None
        self.cpu = None
        self.video = None
        self.screen_photo = None
        self.step_button.configure(state="disabled")
        self.run_button.configure(state="disabled")
        self.reset_button.configure(state="disabled")
        for value in self.info_vars.values():
            value.set("—")
        self.save_type_var.set("Not identified")
        if self.save_type_combo is not None:
            self.save_type_combo.configure(state="disabled")
        self.root.title("GBA Emulator — learning build")
        if self.bios_path is None:
            self.warning_var.set("Open a ROM you have the right to use to inspect its cartridge header.")
        else:
            self.warning_var.set(
                f"BIOS {self.bios_path.name} remains loaded. Open a ROM to use it."
            )
        self._draw_placeholder("EMULATOR READY", "Mode 0 tiles and bitmap graphics are supported")

    def close(self) -> None:
        if self.cpu is not None:
            try:
                self.cpu.bus.flush_save()
            except OSError as exc:
                messagebox.showerror("Save failed", f"Could not write the save file:\n{exc}")
                return
        self._submit_audio()
        self._stop_execution()
        self.audio_output.close()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Desktop GBA emulator")
    parser.add_argument("rom", nargs="?", help="optional .gba/.agb ROM to open at startup")
    parser.add_argument("--run", action="store_true", help="start the ROM after opening it")
    args = parser.parse_args()

    root = tk.Tk()
    app = EmulatorWindow(root)
    if args.rom:
        def open_startup_rom() -> None:
            app.open_rom(args.rom)
            if args.run and app.cpu is not None:
                app.run_cpu()

        root.after_idle(open_startup_rom)
    root.mainloop()


if __name__ == "__main__":
    main()
