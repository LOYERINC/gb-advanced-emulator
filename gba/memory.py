"""A first-pass GBA memory bus.

This maps main memory and cartridge ROM. Timer, display-status, interrupt-flag,
keypad, and immediate-DMA registers have initial device behavior; most other
hardware registers are still simplified storage.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Callable, TypeVar

from gba.cartridge import Cartridge
from gba.psg import GbaPsg
try:
    from gba._native_cpu import run_psg_samples as _run_native_psg_samples
except ImportError:
    _run_native_psg_samples = None


KEY_BITS = {
    "a": 0,
    "b": 1,
    "select": 2,
    "start": 3,
    "right": 4,
    "left": 5,
    "up": 6,
    "down": 7,
    "r": 8,
    "l": 9,
}

# Small, locally implemented ARM IRQ trampoline. Real hardware runs a BIOS
# dispatcher at these addresses; this equivalent forwards to the callback
# pointer in mirrored IWRAM without embedding any Nintendo BIOS bytes.
BIOS_IRQ_STUB = {
    0x0018: 0xEA000042,  # B 0x128
    0x0128: 0xE92D500F,  # STMFD sp!, {r0-r3,r12,lr}
    0x012C: 0xE3A00301,  # MOV r0, #0x04000000
    0x0130: 0xE5100004,  # LDR r0, [r0, #-4] (mirrors 0x03007FFC)
    0x0134: 0xE1A0E00F,  # MOV lr, pc
    0x0138: 0xE12FFF10,  # BX r0
    0x013C: 0xE8BD500F,  # LDMFD sp!, {r0-r3,r12,lr}
    0x0140: 0xE25EF004,  # SUBS pc, lr, #4 (restore SPSR_irq)
}

_AccessMethod = TypeVar("_AccessMethod", bound=Callable[..., object])


def _track_bus_access(width: int, *, write: bool) -> Callable[[_AccessMethod], _AccessMethod]:
    """Track only outermost bus transactions, not read32/read8 internals."""
    def decorate(method: _AccessMethod) -> _AccessMethod:
        if write:
            @wraps(method)
            def wrapped(self: "MemoryBus", address: int, value: int) -> object:
                if not self._instruction_access_tracking:
                    return method(self, address, value)
                outermost = self._access_depth == 0
                self._access_depth += 1
                try:
                    return method(self, address, value)
                finally:
                    self._access_depth -= 1
                    if outermost and self._instruction_access_tracking:
                        self._record_instruction_data_access(address, width, write=True)
        else:
            @wraps(method)
            def wrapped(self: "MemoryBus", address: int) -> object:
                if not self._instruction_access_tracking:
                    return method(self, address)
                outermost = self._access_depth == 0
                self._access_depth += 1
                try:
                    return method(self, address)
                finally:
                    self._access_depth -= 1
                    if outermost and self._instruction_access_tracking:
                        self._record_instruction_data_access(address, width, write=False)

        return wrapped  # type: ignore[return-value]

    return decorate


@dataclass
class MemoryBus:
    cartridge: Cartridge
    bios_data: bytes | None = None
    ewram: bytearray = field(default_factory=lambda: bytearray(256 * 1024))
    iwram: bytearray = field(default_factory=lambda: bytearray(32 * 1024))
    io: bytearray = field(default_factory=lambda: bytearray(1024))
    palette_ram: bytearray = field(default_factory=lambda: bytearray(1024))
    vram: bytearray = field(default_factory=lambda: bytearray(96 * 1024))
    # The direct-boot setup clears OAM by marking every object disabled.
    oam: bytearray = field(default_factory=lambda: bytearray([0, 2, 0, 0, 0, 0, 0, 0]) * 128)
    key_input: int = 0x03FF
    _dma_active: bool = field(default=False, init=False, repr=False)
    _timer_enabled_mask: int = field(default=0, init=False, repr=False)
    _video_generation: int = field(default=0, init=False, repr=False)
    _timer_prescale_remainder: list[int] = field(default_factory=lambda: [0] * 4, init=False, repr=False)
    _timer_reload: list[int] = field(default_factory=lambda: [0] * 4, init=False, repr=False)
    _scanline_cycle: int = field(default=0, init=False, repr=False)
    _scanline: int = field(default=0, init=False, repr=False)
    _save_ram: bytearray | None = field(default=None, init=False, repr=False)
    _save_dirty: bool = field(default=False, init=False, repr=False)
    _cpu_pc: int | None = field(default=None, init=False, repr=False)
    _bios_open_bus: int = field(default=0, init=False, repr=False)
    _low_power_request: str | None = field(default=None, init=False, repr=False)
    _access_depth: int = field(default=0, init=False, repr=False)
    _instruction_access_tracking: bool = field(default=False, init=False, repr=False)
    _instruction_data_extra_cycles: int = field(default=0, init=False, repr=False)
    _last_instruction_data_access: tuple[int, int] | None = field(default=None, init=False, repr=False)
    _flash_state: str = field(default="idle", init=False, repr=False)
    _flash_id_mode: bool = field(default=False, init=False, repr=False)
    _flash_bank: int = field(default=0, init=False, repr=False)
    _eeprom_input_bits: list[int] = field(default_factory=list, init=False, repr=False)
    _eeprom_output_bits: deque[int] = field(default_factory=deque, init=False, repr=False)
    _eeprom_address_bits: int = field(default=14, init=False, repr=False)
    _sound_fifo_a: deque[int] = field(default_factory=lambda: deque(maxlen=32), init=False, repr=False)
    _sound_fifo_b: deque[int] = field(default_factory=lambda: deque(maxlen=32), init=False, repr=False)
    _sound_last_a: int = field(default=0, init=False, repr=False)
    _sound_last_b: int = field(default=0, init=False, repr=False)
    _audio_cycle_remainder: int = field(default=0, init=False, repr=False)
    _gpio_data: int = field(default=0, init=False, repr=False)
    _gpio_direction: int = field(default=0, init=False, repr=False)
    _gpio_control: int = field(default=0, init=False, repr=False)
    _rtc_phase: int = field(default=0, init=False, repr=False)
    _rtc_last_clock: int = field(default=0, init=False, repr=False)
    _rtc_byte: int = field(default=0, init=False, repr=False)
    _rtc_bit_count: int = field(default=0, init=False, repr=False)
    _rtc_command: int = field(default=0, init=False, repr=False)
    _rtc_bytes_left: int = field(default=0, init=False, repr=False)
    _rtc_reading: bool = field(default=False, init=False, repr=False)
    _rtc_tx_bits: deque[int] = field(default_factory=deque, init=False, repr=False)
    _rtc_data_output: int = field(default=0, init=False, repr=False)
    _rtc_snapshot: bytes = field(default=bytes(7), init=False, repr=False)
    _rtc_control: int = field(default=0x40, init=False, repr=False)
    _psg: GbaPsg = field(init=False, repr=False)
    _audio_samples: deque[tuple[tuple[int, int], int]] = field(
        default_factory=lambda: deque(maxlen=32768), init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.bios_data is not None and len(self.bios_data) != 16 * 1024:
            raise ValueError("a GBA BIOS image must be exactly 16 KiB")
        self._gpio_control = 0
        self._psg = GbaPsg(self.io)
        self._update_vcount_match(raise_irq=False)
        save_sizes = {
            "SRAM": 32 * 1024,
            "FLASH": 64 * 1024,
            "FLASH1M": 128 * 1024,
            "EEPROM": 8 * 1024,
        }
        save_size = save_sizes.get(self.cartridge.save_type or "")
        save_path = self.cartridge.save_path
        saved = save_path.read_bytes() if save_path is not None and save_path.exists() else None
        if self.cartridge.save_type == "EEPROM" and saved is not None and len(saved) in (512, 8192):
            save_size = len(saved)
        if save_size is not None:
            self._save_ram = bytearray([0xFF]) * save_size
            if saved is not None:
                self._save_ram[: min(len(saved), len(self._save_ram))] = saved[: len(self._save_ram)]
            if self.cartridge.save_type == "EEPROM":
                self._eeprom_address_bits = 6 if save_size == 512 else 14

    @_track_bus_access(1, write=False)
    def read8(self, address: int) -> int:
        """Read one byte. Unmapped and not-yet-implemented regions return 0."""
        address &= 0xFFFFFFFF

        if address < 0x00004000:
            if self.bios_data is not None:
                if self._cpu_pc is not None and self._cpu_pc >= 0x00004000:
                    # GBA BIOS is protected after execution leaves it; the
                    # CPU sees the recently fetched BIOS opcode on this bus.
                    return (self._bios_open_bus >> ((address & 3) * 8)) & 0xFF
                return self.bios_data[address]
            word = BIOS_IRQ_STUB.get(address & ~3, 0)
            return (word >> (8 * (address & 3))) & 0xFF

        if 0x02000000 <= address <= 0x02FFFFFF:
            return self.ewram[(address - 0x02000000) & 0x3FFFF]
        if 0x03000000 <= address <= 0x03FFFFFF:
            return self.iwram[(address - 0x03000000) & 0x7FFF]
        if 0x04000000 <= address <= 0x040003FF:
            offset = address - 0x04000000
            if address == 0x04000130:
                return self.key_input & 0xFF
            if address == 0x04000131:
                return (self.key_input >> 8) & 0x03
            if address == 0x04000205:
                return self.io[offset] & 0x7F  # WAITCNT bit 15 is read-only zero.
            # Timer counters are live registers, while the adjacent reload
            # values stay in the ordinary IO byte array.
            if 0x04000100 <= address <= 0x0400010F and (address & 3) < 2:
                timer = (address - 0x04000100) // 4
                count = self._io16(0x100 + timer * 4)
                return (count >> (8 * (address & 1))) & 0xFF
            if 0x090 <= offset <= 0x09F:
                return self._psg.read_wave_ram(offset - 0x090)
            return self.io[address - 0x04000000]
        if 0x05000000 <= address <= 0x05FFFFFF:
            return self.palette_ram[(address - 0x05000000) & 0x3FF]
        if 0x06000000 <= address <= 0x06FFFFFF:
            offset = (address - 0x06000000) & 0x1FFFF
            if offset >= 0x18000:
                offset -= 0x8000
            return self.vram[offset]
        if 0x07000000 <= address <= 0x07FFFFFF:
            return self.oam[(address - 0x07000000) & 0x3FF]
        if 0x08000000 <= address <= 0x0DFFFFFF and self.cartridge.data:
            gpio = self._gpio_register_offset(address)
            if gpio is not None and self.cartridge.has_rtc and (self._gpio_control & 1):
                value = self._gpio_register_value(gpio)
                return (value >> (8 * (address & 1))) & 0xFF
            return self.cartridge.data[(address - 0x08000000) % len(self.cartridge.data)]
        if 0x0E000000 <= address <= 0x0FFFFFFF:
            if self._save_ram is None:
                return 0xFF
            save_type = self.cartridge.save_type
            offset = (address - 0x0E000000) & 0xFFFF
            if save_type in ("FLASH", "FLASH1M"):
                if self._flash_id_mode:
                    if offset == 0:
                        return 0xC2  # Macronix manufacturer ID
                    if offset == 1:
                        return 0x09 if save_type == "FLASH1M" else 0x1C
                    return 0xFF
                bank_offset = self._flash_bank * 0x10000 if save_type == "FLASH1M" else 0
                return self._save_ram[bank_offset + offset]
            return self._save_ram[(address - 0x0E000000) & 0x7FFF]

        # BIOS and unimplemented save-memory behavior are not mapped here.
        return 0

    def set_cpu_pc(self, address: int) -> None:
        """Tell BIOS protection which address the CPU is about to execute."""
        self._cpu_pc = address & 0xFFFFFFFF

    @staticmethod
    def _gamepak_waitstate(address: int) -> int | None:
        if 0x08000000 <= address < 0x0E000000:
            return (address >> 25) - 4
        return None

    def _gamepak_halfword_cycles(self, address: int, sequential: bool, waitcnt: int) -> int:
        if 0x08000000 <= address < 0x0E000000:
            bank = (address >> 25) - 4
        else:
            return 1
        if address & 0x1FFFF == 0:
            sequential = False
        if sequential:
            bit = (4, 7, 10)[bank]
            wait_states = ((2, 1), (4, 1), (8, 1))[bank][(waitcnt >> bit) & 1]
            return 1 + wait_states
        setting = (waitcnt >> (2, 5, 8)[bank]) & 0x3
        return 1 + (4, 3, 2, 8)[setting]

    def begin_instruction_access_tracking(self) -> None:
        self._instruction_access_tracking = True
        self._instruction_data_extra_cycles = 0
        self._last_instruction_data_access = None

    def finish_instruction_access_tracking(self) -> int:
        self._instruction_access_tracking = False
        extra_cycles = self._instruction_data_extra_cycles
        self._instruction_data_extra_cycles = 0
        self._last_instruction_data_access = None
        return extra_cycles

    def cancel_instruction_access_tracking(self) -> None:
        self._instruction_access_tracking = False
        self._instruction_data_extra_cycles = 0
        self._last_instruction_data_access = None

    def _record_instruction_data_access(self, address: int, width: int, *, write: bool) -> None:
        del write  # Current WAITCNT and internal-memory costs match for reads/writes.
        address &= ~(width - 1)
        # IWRAM is zero-wait internal memory. It cannot contribute extra
        # transfer clocks, and a following Game Pak access is never sequential
        # after it, so skip the general WAITCNT/display contention machinery.
        if 0x03000000 <= address <= 0x03FFFFFF:
            self._last_instruction_data_access = None
            return
        previous = self._last_instruction_data_access
        sequential = False
        if previous is not None:
            old_address, old_width = previous
            sequential = (
                old_address + old_width == address
                and self._gamepak_waitstate(old_address) == self._gamepak_waitstate(address)
            )

        waitcnt = self.io[0x204] | ((self.io[0x205] & 0x7F) << 8)
        bank = self._gamepak_waitstate(address)
        if bank is not None:
            first = self._gamepak_halfword_cycles(address & ~1, sequential, waitcnt)
            total = first
            if width == 4:
                total += self._gamepak_halfword_cycles(address + 2, True, waitcnt)
        elif 0x02000000 <= address <= 0x02FFFFFF:
            total = 6 if width == 4 else 3  # EWRAM: 16-bit external bus.
        elif 0x05000000 <= address <= 0x06FFFFFF:
            total = 2 if width == 4 else 1  # Palette/VRAM: 16-bit bus.
            display_active = (
                self._scanline < 160
                and self._scanline_cycle < 960
                and not (self._io16(0x000) & 0x0080)
            )
            if display_active:
                total += 1  # Approximate a video-memory contention wait.
        elif 0x07000000 <= address <= 0x07FFFFFF:
            total = 1  # OAM is on the internal 32-bit bus.
        elif 0x0E000000 <= address <= 0x0FFFFFFF:
            total = 1 + (4, 3, 2, 8)[waitcnt & 0x3]
        else:
            total = 1  # BIOS, IWRAM, I/O, and other internal bus accesses.

        self._instruction_data_extra_cycles += max(0, total - 1)
        self._last_instruction_data_access = (address, width)

    def gamepak_fetch_extra_cycles(
        self,
        address: int,
        width: int,
        previous_fetch: tuple[int, int, bool] | None,
        thumb: bool,
    ) -> int:
        """Estimate extra Game Pak instruction-fetch clocks from WAITCNT.

        The base instruction estimate already includes one fetch clock, so
        this returns only the additional wait clocks. ARM words use two
        16-bit cartridge bus halves. The prefetch buffer remains unmodeled.
        """
        address &= 0xFFFFFFFF
        if not 0x08000000 <= address < 0x0E000000:
            return 0
        bank = (address >> 25) - 4
        waitcnt = self.io[0x204] | ((self.io[0x205] & 0x7F) << 8)
        sequential = False
        if previous_fetch is not None:
            old_address, old_width, old_thumb = previous_fetch
            sequential = old_address + old_width == address and old_thumb == thumb
            if sequential:
                sequential = 0x08000000 <= old_address < 0x0E000000 and (old_address >> 25) - 4 == bank

        total = self._gamepak_halfword_cycles(address, sequential, waitcnt)
        if width == 4:
            total += self._gamepak_halfword_cycles(address + 2, True, waitcnt)
        return max(0, total - 1)

    def consume_low_power_request(self) -> str | None:
        """Return and clear a pending HALTCNT request for the CPU core."""
        request = self._low_power_request
        self._low_power_request = None
        return request

    @property
    def video_generation(self) -> int:
        """Version of video memory/register state used to cache rendered frames."""
        return self._video_generation

    def _mark_video_dirty(self) -> None:
        self._video_generation += 1

    def record_instruction_fetch(self, address: int, instruction: int, *, thumb: bool) -> None:
        """Keep the most recent BIOS opcode for protected reads outside BIOS."""
        if self.bios_data is None or address >= 0x00004000:
            return
        if thumb:
            instruction &= 0xFFFF
            self._bios_open_bus = instruction | (instruction << 16)
        else:
            self._bios_open_bus = instruction & 0xFFFFFFFF

    @staticmethod
    def _mapped_vram_offset(address: int) -> int:
        """Map the 128 KiB VRAM aperture onto the GBA's 96 KiB VRAM."""
        offset = (address - 0x06000000) & 0x1FFFF
        return offset - 0x8000 if offset >= 0x18000 else offset

    def _is_eeprom_address(self, address: int) -> bool:
        if self.cartridge.save_type != "EEPROM":
            return False
        address &= 0xFFFFFFFF
        if len(self.cartridge.data) > 16 * 1024 * 1024:
            return 0x0DFFFF00 <= address <= 0x0DFFFFFF
        return 0x0D000000 <= address <= 0x0DFFFFFF

    @_track_bus_access(2, write=False)
    def read16(self, address: int) -> int:
        """Read a little-endian halfword."""
        address &= ~1
        base = address & 0xFF000000
        if base == 0x02000000:
            offset = (address - 0x02000000) & 0x3FFFF
            return self.ewram[offset] | (self.ewram[(offset + 1) & 0x3FFFF] << 8)
        if base == 0x03000000:
            offset = (address - 0x03000000) & 0x7FFF
            return self.iwram[offset] | (self.iwram[(offset + 1) & 0x7FFF] << 8)
        if 0x04000000 <= address <= 0x040003FE:
            offset = address - 0x04000000
            if address == 0x04000130:
                return self.key_input & 0x03FF
            if address == 0x04000204:
                return self.io[offset] | ((self.io[offset + 1] & 0x7F) << 8)
            if 0x100 <= offset <= 0x10E and (offset & 3) == 0:
                return self._io16(offset)
            if not (0x090 <= offset <= 0x09F):
                return self.io[offset] | (self.io[offset + 1] << 8)
        if self.cartridge.save_type == "EEPROM" and self._is_eeprom_address(address):
            return self._eeprom_output_bits.popleft() if self._eeprom_output_bits else 1
        if 0x08000000 <= address <= 0x0DFFFFFF and self.cartridge.data:
            data = self.cartridge.data
            gpio_offset = (address - 0x08000000) % 0x02000000
            touches_gpio = (
                self.cartridge.has_rtc
                and bool(self._gpio_control & 1)
                and gpio_offset < 0xCA
                and gpio_offset + 2 > 0xC4
            )
            if not touches_gpio:
                offset = (address - 0x08000000) % len(data)
                if offset + 2 <= len(data):
                    return data[offset] | (data[offset + 1] << 8)
        if 0x0E000000 <= address <= 0x0FFFFFFF and self.cartridge.save_type in ("SRAM", "FLASH", "FLASH1M"):
            value = self.read8(address)
            return value | (value << 8)
        return self.read8(address) | (self.read8(address + 1) << 8)

    def enabled_interrupt_requests(self) -> int:
        """Return active IE/IF requests when master interrupts are enabled."""
        if not (self.io[0x208] & 1):
            return 0
        enabled = self.io[0x200] | (self.io[0x201] << 8)
        requested = self.io[0x202] | (self.io[0x203] << 8)
        return enabled & requested

    def cycles_until_next_interrupt(self, wait_mask: int | None = None) -> int:
        """Skip sleeping CPU time to the next interrupt the CPU is waiting on."""
        line_cycles = 1232
        frame_cycles = line_cycles * 228
        ie = self._io16(0x200)
        event_mask = ie if wait_mask is None else ie & wait_mask
        dispstat = self._io16(0x004)
        position = self._scanline * line_cycles + self._scanline_cycle
        candidates = [frame_cycles - position if position < frame_cycles else line_cycles]

        # DISPSTAT requests are raised at HBlank, VBlank start, and VCOUNT match.
        if event_mask & (1 << 1) and dispstat & (1 << 4):
            hblank = self._scanline * line_cycles + 960
            candidates.append(hblank - position if hblank > position else line_cycles - self._scanline_cycle + 960)
        if event_mask & 1 and dispstat & (1 << 3):
            lines = (160 - self._scanline) % 228 or 228
            delta = lines * line_cycles - self._scanline_cycle
            candidates.append(delta if delta > 0 else delta + frame_cycles)
        if event_mask & (1 << 2) and dispstat & (1 << 5):
            target = self.io[5]
            lines = (target - self._scanline) % 228 or 228
            delta = lines * line_cycles - self._scanline_cycle
            candidates.append(delta if delta > 0 else delta + frame_cycles)

        # Timer IRQs can wake the CPU between display events. Compute the next
        # overflow for a directly clocked timer; cascaded timers use one-line
        # checkpoints so their chained overflow is observed promptly.
        divisors = (1, 64, 256, 1024)
        for timer in range(4):
            if not (event_mask & (1 << (3 + timer)) and self._timer_enabled_mask & (1 << timer)):
                continue
            offset = 0x102 + timer * 4
            control = self._io16(offset)
            if timer and control & 0x04:
                candidates.append(line_cycles - self._scanline_cycle % line_cycles)
                continue
            count = self._io16(0x100 + timer * 4)
            cycles = (0x10000 - count) * divisors[control & 3] - self._timer_prescale_remainder[timer]
            candidates.append(max(1, cycles))
        return max(1, min(candidates))

    @_track_bus_access(4, write=False)
    def read32(self, address: int) -> int:
        """Read a little-endian word."""
        address &= ~3
        base = address & 0xFF000000
        if base == 0x02000000:
            data, offset, mask = self.ewram, (address - 0x02000000) & 0x3FFFF, 0x3FFFF
            return (data[offset] | (data[(offset + 1) & mask] << 8)
                    | (data[(offset + 2) & mask] << 16) | (data[(offset + 3) & mask] << 24))
        if base == 0x03000000:
            data, offset, mask = self.iwram, (address - 0x03000000) & 0x7FFF, 0x7FFF
            return (data[offset] | (data[(offset + 1) & mask] << 8)
                    | (data[(offset + 2) & mask] << 16) | (data[(offset + 3) & mask] << 24))
        if self.cartridge.save_type == "EEPROM" and self._is_eeprom_address(address):
            return self.read16(address) | (self.read16(address + 2) << 16)
        if 0x08000000 <= address <= 0x0DFFFFFF and self.cartridge.data:
            data = self.cartridge.data
            gpio_offset = (address - 0x08000000) % 0x02000000
            touches_gpio = (
                self.cartridge.has_rtc
                and bool(self._gpio_control & 1)
                and gpio_offset < 0xCA
                and gpio_offset + 4 > 0xC4
            )
            if not touches_gpio:
                offset = (address - 0x08000000) % len(data)
                if offset + 4 <= len(data):
                    return (
                        data[offset]
                        | (data[offset + 1] << 8)
                        | (data[offset + 2] << 16)
                        | (data[offset + 3] << 24)
                    )
        if 0x0E000000 <= address <= 0x0FFFFFFF and self.cartridge.save_type in ("SRAM", "FLASH", "FLASH1M"):
            value = self.read8(address)
            return value * 0x01010101
        return (
            self.read8(address)
            | (self.read8(address + 1) << 8)
            | (self.read8(address + 2) << 16)
            | (self.read8(address + 3) << 24)
        )

    @_track_bus_access(1, write=True)
    def write8(self, address: int, value: int) -> None:
        """Write one byte to writable RAM regions."""
        address &= 0xFFFFFFFF
        value &= 0xFF

        # The cartridge GPIO port is 16-bit; byte stores do not drive it.
        if 0x08000000 <= address <= 0x0DFFFFFF:
            return

        if 0x02000000 <= address <= 0x02FFFFFF:
            self.ewram[(address - 0x02000000) & 0x3FFFF] = value
        elif 0x03000000 <= address <= 0x03FFFFFF:
            self.iwram[(address - 0x03000000) & 0x7FFF] = value
        elif 0x04000000 <= address <= 0x040003FF:
            if address - 0x04000000 <= 0x055:
                self._mark_video_dirty()
            if address in (0x04000130, 0x04000131):
                return
            offset = address - 0x04000000
            if offset == 0x205:
                self.io[offset] = value & 0x7F  # WAITCNT bit 15 is read-only zero.
                return
            if address == 0x04000301:
                # HALTCNT is write-only: bit 7 selects Stop; zero selects Halt.
                # The CPU consumes this request at its next instruction boundary.
                self._low_power_request = "stop" if value & 0x80 else "halt"
                return
            if offset in (0x202, 0x203):
                # IF is write-one-to-clear.
                pending = self._io16(0x202)
                shift = 8 * (offset & 1)
                self._set_io16(0x202, pending & ~((value & 0xFF) << shift))
                return
            if offset in (0x132, 0x133):
                self.io[offset] = value
                self._update_keypad_irq()
                return
            if offset in (0x006, 0x007):
                return  # VCOUNT is read-only.
            if offset in (0x004, 0x005):
                if offset == 0x004:
                    self.io[offset] = (self.io[offset] & 0x07) | (value & 0x38)
                else:
                    self.io[offset] = value
                self._update_vcount_match(raise_irq=False)
                return
            if offset == 0x084:
                was_enabled = bool(self.io[offset] & 0x80)
                self.io[offset] = (self.io[offset] & 0x0F) | (value & 0x80)
                if was_enabled and not (value & 0x80):
                    self._psg.master_disabled()
                    self.io[0x060:0x082] = bytes(0x22)
                    self._psg.wave_ram[:] = bytes(len(self._psg.wave_ram))
                return
            if offset == 0x085:
                return
            if 0x090 <= offset <= 0x09F:
                self._psg.write_wave_ram(offset - 0x090, value)
                return
            if 0x060 <= offset <= 0x081:
                self.io[offset] = value
                self._psg.write_register(offset, value)
                return
            if 0x100 <= offset <= 0x10F and (offset & 3) < 2:
                timer = (offset - 0x100) // 4
                # TMxCNT_L reads the current count but writes the reload latch.
                shift = 8 * (offset & 1)
                reload_value = (self._timer_reload[timer] & ~(0xFF << shift)) | (value << shift)
                self._timer_reload[timer] = reload_value
                if not (self._io16(0x102 + timer * 4) & 0x80):
                    self._set_io16(0x100 + timer * 4, reload_value)
                return
            if 0x102 <= offset <= 0x10F and (offset & 3) >= 2:
                timer = (offset - 0x102) // 4
                old_control = self._io16(0x102 + timer * 4)
                shift = 8 * (offset & 1)
                control = (old_control & ~(0xFF << shift)) | (value << shift)
                control &= 0x00C7  # prescaler, cascade, IRQ, enable only
                self._set_io16(0x102 + timer * 4, control)
                if control & 0x80:
                    self._timer_enabled_mask |= 1 << timer
                else:
                    self._timer_enabled_mask &= ~(1 << timer)
                if not (old_control & 0x80) and (control & 0x80):
                    self._set_io16(0x100 + timer * 4, self._timer_reload[timer])
                    self._timer_prescale_remainder[timer] = 0
                return
            if 0x0A0 <= offset <= 0x0A7:
                fifo = self._sound_fifo_a if offset < 0x0A4 else self._sound_fifo_b
                if len(fifo) < fifo.maxlen:
                    fifo.append(value)
                return
            self.io[address - 0x04000000] = value
            if offset == 0x083:
                if value & 0x08:
                    self._sound_fifo_a.clear()
                    self._sound_last_a = 0
                    self.io[offset] &= ~0x08
                if value & 0x80:
                    self._sound_fifo_b.clear()
                    self._sound_last_b = 0
                    self.io[offset] &= ~0x80
            # DMAxCNT_H occupies the upper halfword of each 12-byte channel block.
            relative = address - 0x040000BB
            if relative >= 0 and relative % 12 == 0 and relative // 12 < 4:
                self._start_dma(relative // 12)
        elif 0x05000000 <= address <= 0x05FFFFFF:
            self._mark_video_dirty()
            offset = (address - 0x05000000) & 0x3FF
            offset &= ~1
            # Palette RAM is 16-bit wide: STRB duplicates its byte to both
            # halves of the addressed color halfword.
            self.palette_ram[offset] = value
            self.palette_ram[offset + 1] = value
        elif 0x06000000 <= address <= 0x06FFFFFF:
            self._mark_video_dirty()
            offset = self._mapped_vram_offset(address)
            mode = self._io16(0) & 0x7
            bg_end = 0x14000 if mode in (3, 4, 5) else 0x10000
            if offset < bg_end:
                # Byte writes work in BG VRAM and repeat into the other byte.
                # OBJ VRAM ignores STRB writes entirely.
                offset &= ~1
                self.vram[offset] = value
                self.vram[offset + 1] = value
        elif 0x07000000 <= address <= 0x07FFFFFF:
            return  # OAM ignores 8-bit writes.
        elif 0x0E000000 <= address <= 0x0FFFFFFF and self._save_ram is not None:
            if self.cartridge.save_type == "SRAM":
                self._save_ram[(address - 0x0E000000) & 0x7FFF] = value
                self._save_dirty = True
            elif self.cartridge.save_type in ("FLASH", "FLASH1M"):
                self._write_flash(address, value)
        # Writes to ROM, BIOS, and unsupported save-chip types are ignored.

    def _write_flash(self, address: int, value: int) -> None:
        """Handle the common Macronix/Sanyo-style Flash command sequence."""
        offset = (address - 0x0E000000) & 0xFFFF
        state = self._flash_state
        if value == 0xF0 and state != "program":
            self._flash_id_mode = False
            self._flash_state = "idle"
            return

        if state == "program":
            index = (self._flash_bank * 0x10000 if self.cartridge.save_type == "FLASH1M" else 0) + offset
            self._save_ram[index] &= value  # Flash programming can change 1 bits to 0.
            self._save_dirty = True
            self._flash_state = "idle"
        elif state == "bank":
            # FLASH1M selects its bank by writing bank 0/1 at 0x0E000000.
            # Ignore writes to other offsets; the command sequence is still consumed.
            if self.cartridge.save_type == "FLASH1M" and offset == 0:
                self._flash_bank = value & 1
            self._flash_state = "idle"
        elif state in ("unlock1", "erase_unlock1"):
            expected_address = 0x2AAA if state == "unlock1" else 0x5555
            if offset == expected_address and value == (0x55 if state == "unlock1" else 0xAA):
                self._flash_state = "unlock2" if state == "unlock1" else "erase_unlock2"
            else:
                self._flash_state = "idle"
        elif state in ("unlock2", "erase_unlock2"):
            expected_address = 0x5555 if state == "unlock2" else 0x2AAA
            if offset != expected_address or value != (0xAA if state == "unlock2" else 0x55):
                self._flash_state = "idle"
                return
            self._flash_state = "command" if state == "unlock2" else "erase_confirm"
        elif state == "command":
            self._flash_state = "idle"
            if offset != 0x5555:
                return
            if value == 0x90:
                self._flash_id_mode = True
            elif value == 0x80:
                self._flash_state = "erase_unlock1"
            elif value == 0xA0:
                self._flash_state = "program"
            elif value == 0xB0 and self.cartridge.save_type == "FLASH1M":
                self._flash_state = "bank"
        elif state == "erase_confirm":
            self._flash_state = "idle"
            bank_offset = self._flash_bank * 0x10000 if self.cartridge.save_type == "FLASH1M" else 0
            if offset == 0x5555 and value == 0x10:
                self._save_ram[:] = b"\xFF" * len(self._save_ram)
                self._save_dirty = True
            elif value == 0x30:
                sector = bank_offset + (offset & 0xF000)
                self._save_ram[sector : sector + 0x1000] = b"\xFF" * 0x1000
                self._save_dirty = True
        elif offset == 0x5555 and value == 0xAA:
            self._flash_state = "unlock1"

    def flush_save(self) -> None:
        """Persist changed SRAM to the .sav file beside its ROM, if configured."""
        if not self._save_dirty or self._save_ram is None or self.cartridge.save_path is None:
            return
        save_path: Path = self.cartridge.save_path
        temporary_path = save_path.with_suffix(save_path.suffix + ".tmp")
        temporary_path.write_bytes(self._save_ram)
        temporary_path.replace(save_path)
        self._save_dirty = False

    def trigger_dma(self, timing: int, channel_filter: int | None = None) -> None:
        """Run enabled repeating DMA channels waiting for this hardware event.

        Timing values are 1=VBlank, 2=HBlank, and 3=special. Video and sound
        devices can call this when their corresponding event is modeled.
        """
        for channel in range(4):
            if timing == 3 and channel not in (1, 2):
                continue  # Sound FIFO special DMA is available on channels 1/2.
            if channel_filter is not None and channel != channel_filter:
                continue
            control = self.read16(0x040000BA + channel * 12)
            if control & 0x8000 and ((control >> 12) & 3) == timing:
                self._run_dma(channel, control)

    def advance_cycles(self, cycles: int) -> None:
        """Advance display and timer hardware by CPU cycles.

        The desktop runner supplies approximate instruction cycle counts, so
        these devices work but are not yet cycle-accurate to commercial games.
        """
        if cycles <= 0:
            return
        self._advance_display(cycles)
        if self._timer_enabled_mask:
            cascaded_ticks = 0
            for timer in range(4):
                if not (self._timer_enabled_mask & (1 << timer)):
                    cascaded_ticks = 0
                    continue
                offset = 0x102 + timer * 4
                control = self.io[offset] | (self.io[offset + 1] << 8)
                if timer and (control & 0x04):
                    cascaded_ticks = self._tick_timer(timer, cascaded_ticks)
                    continue
                divisors = (1, 64, 256, 1024)
                total = self._timer_prescale_remainder[timer] + cycles
                ticks, self._timer_prescale_remainder[timer] = divmod(total, divisors[control & 3])
                cascaded_ticks = self._tick_timer(timer, ticks)
        # Most instruction batches end between 32.768 kHz sample points. In
        # that common case, only carry the fraction forward; invoke the mixer
        # when this batch reaches or crosses a sample boundary.
        if self._audio_cycle_remainder + cycles < 512:
            self._audio_cycle_remainder += cycles
        else:
            self._advance_audio(cycles)

    def initialize_post_bios_scanline(self) -> None:
        """Start the LCD at the line where the BIOS hands off to a cartridge."""
        self._scanline = 0x7E
        self._scanline_cycle = 0
        self.io[6] = self._scanline
        # DISPSTAT status bits are read-only; preserve the interrupt enables.
        self.io[4] &= ~0x07
        self._update_vcount_match(raise_irq=False)

    def _tick_timer(self, timer: int, ticks: int) -> int:
        """Apply timer ticks and return the number of overflows for cascading."""
        if ticks <= 0:
            return 0
        offset = 0x100 + timer * 4
        count = self._io16(offset)
        reload_value = self._timer_reload[timer]
        until_overflow = 0x10000 - count
        if ticks < until_overflow:
            self._set_io16(offset, count + ticks)
            return 0
        ticks -= until_overflow
        period = 0x10000 - reload_value
        overflows = 1 + ticks // period
        new_count = reload_value + ticks % period
        self._set_io16(offset, new_count)
        if self._io16(offset + 2) & 0x40:
            self._set_io16(0x202, self._io16(0x202) | (1 << (3 + timer)))
        for _ in range(overflows):
            self._clock_sound_timer(timer)
        return overflows

    def _clock_sound_timer(self, timer: int) -> None:
        """Consume direct-sound FIFO samples and request FIFO refill DMA."""
        if not (self._io16(0x084) & 0x0080):
            return
        control = self._io16(0x082)
        for fifo, timer_bit, fifo_index in (
            (self._sound_fifo_a, 10, 0),
            (self._sound_fifo_b, 14, 1),
        ):
            if ((control >> timer_bit) & 1) != timer:
                continue
            if fifo:
                sample = fifo.popleft()
                if fifo_index == 0:
                    self._sound_last_a = sample
                else:
                    self._sound_last_b = sample
            else:
                sample = self._sound_last_a if fifo_index == 0 else self._sound_last_b
            if len(fifo) <= 16:
                self.trigger_dma(3, channel_filter=fifo_index + 1)

    def _advance_audio(self, cycles: int) -> None:
        """Mix PSG and held FIFO levels at the GBA's default 32.768 kHz rate."""
        self._audio_cycle_remainder += cycles
        # When the sound master is off, no PSG or Direct Sound output can be
        # heard. Preserve the 512-cycle sample phase without iterating once
        # per silent sample during CPU batches.
        if not (self._io16(0x084) & 0x80):
            self._audio_cycle_remainder %= 512
            return
        if not any(self._psg.enabled) and not self._sound_last_a and not self._sound_last_b:
            samples, self._audio_cycle_remainder = divmod(self._audio_cycle_remainder, 512)
            frame_samples = self._psg.frame_sample_count + samples
            sequencer_ticks, self._psg.frame_sample_count = divmod(frame_samples, 64)
            for _ in range(sequencer_ticks):
                self._psg.clock_frame_sequencer()
            return
        if _run_native_psg_samples is not None:
            sample_count, self._audio_cycle_remainder = divmod(self._audio_cycle_remainder, 512)
            psg_samples = _run_native_psg_samples(self._psg, sample_count)
            control = self._io16(0x080)
            direct = self._io16(0x082)
            direct_levels = []
            for sample, volume_bit, left_bit, right_bit in (
                (self._sound_last_a, 2, 9, 8), (self._sound_last_b, 3, 13, 12),
            ):
                level = (sample - 0x100 if sample & 0x80 else sample) * 128
                if not direct & (1 << volume_bit):
                    level //= 2
                direct_levels.append((level if direct & (1 << left_bit) else 0,
                                      level if direct & (1 << right_bit) else 0))
            for psg_left, psg_right in psg_samples:
                left = psg_left + direct_levels[0][0] + direct_levels[1][0]
                right = psg_right + direct_levels[0][1] + direct_levels[1][1]
                self._audio_samples.append(((max(-32768, min(32767, left)),
                                             max(-32768, min(32767, right))), 32768))
            return

        while self._audio_cycle_remainder >= 512:
            self._audio_cycle_remainder -= 512
            psg_left, psg_right = self._psg.sample()
            control = self._io16(0x080)
            direct = self._io16(0x082)
            left, right = psg_left, psg_right
            for sample, volume_bit, left_bit, right_bit in (
                (self._sound_last_a, 2, 9, 8), (self._sound_last_b, 3, 13, 12),
            ):
                signed_sample = sample - 0x100 if sample & 0x80 else sample
                level = signed_sample * 128
                if not (direct & (1 << volume_bit)):
                    level //= 2
                if direct & (1 << left_bit):
                    left += level
                if direct & (1 << right_bit):
                    right += level
            stereo_sample = (max(-32768, min(32767, left)), max(-32768, min(32767, right)))
            self._audio_samples.append((stereo_sample, 32768))

    def drain_audio_samples(
        self, limit: int = 4096
    ) -> list[tuple[tuple[int, int], int]]:
        """Return queued ((left, right), approximate sample-rate) samples."""
        count = min(max(0, limit), len(self._audio_samples))
        return [self._audio_samples.popleft() for _ in range(count)]

    def _advance_display(self, cycles: int) -> None:
        """Advance LCD scanlines; visible lines are followed by VBlank."""
        # Most CPU instructions finish before the next LCD edge. Keep that
        # common case out of the boundary-processing loop; exact-edge and
        # multi-edge advances still use the full event path below.
        hblank_start = 960
        line_end = 1232
        boundary = hblank_start if self._scanline_cycle < hblank_start else line_end
        if cycles < boundary - self._scanline_cycle:
            self._scanline_cycle += cycles
            return

        remaining = cycles
        while remaining:
            boundary = hblank_start if self._scanline_cycle < hblank_start else line_end
            step = min(remaining, boundary - self._scanline_cycle)
            self._scanline_cycle += step
            remaining -= step

            if self._scanline_cycle == hblank_start:
                self.io[4] |= 0x02  # DISPSTAT HBlank flag
                if self.io[4] & 0x10:
                    self._request_interrupt(1)
                self.trigger_dma(2)

            if self._scanline_cycle == line_end:
                self._scanline_cycle = 0
                self.io[4] &= ~0x02
                self._scanline = (self._scanline + 1) % 228
                self.io[6] = self._scanline
                self.io[7] = 0
                if self._scanline == 160:
                    self.io[4] |= 0x01  # DISPSTAT VBlank flag
                    if self.io[4] & 0x08:
                        self._request_interrupt(0)
                    self.trigger_dma(1)
                elif self._scanline == 0:
                    self.io[4] &= ~0x01
                self._update_vcount_match(raise_irq=True)

    def _update_vcount_match(self, raise_irq: bool) -> None:
        if self._scanline == self.io[5]:
            self.io[4] |= 0x04
            if raise_irq and self.io[4] & 0x20:
                self._request_interrupt(2)
        else:
            self.io[4] &= ~0x04

    def _request_interrupt(self, bit: int) -> None:
        self._set_io16(0x202, self._io16(0x202) | (1 << bit))

    def _resize_eeprom(self, address_bits: int) -> None:
        size = 512 if address_bits == 6 else 8192
        if self._save_ram is None or len(self._save_ram) != size:
            old_data = self._save_ram or bytearray()
            resized = bytearray([0xFF]) * size
            resized[: min(size, len(old_data))] = old_data[:size]
            self._save_ram = resized
        self._eeprom_address_bits = address_bits

    @staticmethod
    def _bits_to_int(bits: list[int]) -> int:
        value = 0
        for bit in bits:
            value = (value << 1) | (bit & 1)
        return value

    def _finish_eeprom_transfer(self) -> None:
        """Decode one DMA3 EEPROM command and prepare its serial read response."""
        bits = self._eeprom_input_bits
        self._eeprom_input_bits = []
        if len(bits) < 9 or self._save_ram is None:
            return

        if bits[:2] == [1, 1]:  # Read request: command + block address + stop bit.
            address_bits = len(bits) - 3
            if address_bits not in (6, 14) or bits[-1] != 0:
                return
            self._resize_eeprom(address_bits)
            block = self._bits_to_int(bits[2:-1])
            start = (block * 8) % len(self._save_ram)
            data = self._save_ram[start : start + 8]
            self._eeprom_output_bits = deque([0] * 4)
            for byte in data:
                self._eeprom_output_bits.extend((byte >> shift) & 1 for shift in range(7, -1, -1))
        elif bits[:2] == [1, 0]:  # Write request: address, 64 data bits, stop bit.
            address_bits = len(bits) - 67
            if address_bits not in (6, 14) or bits[-1] != 0:
                return
            self._resize_eeprom(address_bits)
            block = self._bits_to_int(bits[2 : 2 + address_bits])
            start = (block * 8) % len(self._save_ram)
            data_bits = bits[2 + address_bits : -1]
            data = bytes(self._bits_to_int(data_bits[i : i + 8]) for i in range(0, 64, 8))
            self._save_ram[start : start + 8] = data
            self._save_dirty = True

    def _io16(self, offset: int) -> int:
        return self.io[offset] | (self.io[offset + 1] << 8)

    def _set_io16(self, offset: int, value: int) -> None:
        self.io[offset] = value & 0xFF
        self.io[offset + 1] = (value >> 8) & 0xFF

    def register_ram_reset(self, flags: int) -> None:
        """Apply the BIOS SWI 01h selective RAM and I/O reset flags."""
        flags &= 0xFF
        if flags & 0x01:
            self.ewram[:] = bytes(len(self.ewram))
        if flags & 0x02:
            # BIOS-owned stacks, IRQ callback and wait flags live in the final
            # 0x200 bytes and are deliberately preserved by RegisterRamReset.
            self.iwram[:-0x200] = bytes(len(self.iwram) - 0x200)
        if flags & 0x04:
            self.palette_ram[:] = bytes(len(self.palette_ram))
        if flags & 0x08:
            self.vram[:] = bytes(len(self.vram))
        if flags & 0x10:
            self.oam[:] = bytes(len(self.oam))

        if flags & 0x20:
            # Serial/link and Joy Bus registers. KEYINPUT/KEYCNT are separate
            # keypad registers and are not part of this SIO reset group.
            for start, end in ((0x128, 0x12C), (0x134, 0x136), (0x140, 0x142), (0x150, 0x160)):
                self.io[start:end] = bytes(end - start)

        if flags & 0x40:
            self.io[0x060:0x085] = bytes(0x25)
            self._set_io16(0x088, 0x0200)  # SOUNDBIAS reset level.
            self._psg.wave_ram[:] = bytes(len(self._psg.wave_ram))
            self._psg = GbaPsg(self.io)
            self._sound_fifo_a.clear()
            self._sound_fifo_b.clear()
            self._sound_last_a = self._sound_last_b = 0
            self._audio_cycle_remainder = 0
            self._audio_samples.clear()

        if flags & 0x80:
            # Video registers (including affine transform defaults), DMA,
            # timers, and IE/IF/WAITCNT/IME are the remaining reset group.
            self.io[0x004:0x020] = bytes(0x1C)
            for offset in (0x020, 0x026, 0x030, 0x036):
                self._set_io16(offset, 0x0100)
            for start, end in ((0x022, 0x026), (0x028, 0x030), (0x032, 0x036), (0x038, 0x056),
                               (0x0B0, 0x0E0), (0x100, 0x110), (0x200, 0x206), (0x208, 0x20A)):
                self.io[start:end] = bytes(end - start)
            self.io[0x006:0x008] = bytes(2)
            self._timer_reload[:] = [0] * 4
            self._timer_prescale_remainder[:] = [0] * 4
            self._timer_enabled_mask = 0
            self._scanline = 0
            self._scanline_cycle = 0

        # BIOS always forces a blank display during this reset service, even
        # when the caller did not request the general-register reset group.
        self._set_io16(0x000, 0x0080)
        self._mark_video_dirty()
        self._update_vcount_match(raise_irq=False)

    def initialize_bios_sound_area(self, address: int) -> None:
        """Initialize the small public header used by BIOS sound-driver calls."""
        address &= 0xFFFFFFFF
        area_size = 0xFB0
        if 0x02000000 <= address and address + area_size <= 0x02040000:
            offset = address - 0x02000000
            self.ewram[offset : offset + area_size] = bytes(area_size)
            self.write32(address, 0x68736D53)  # BIOS SoundArea ready marker, "Smsh".
        elif 0x03000000 <= address and address + area_size <= 0x03007E00:
            offset = address - 0x03000000
            self.iwram[offset : offset + area_size] = bytes(area_size)
            self.write32(address, 0x68736D53)
        # The BIOS keeps the work-area pointer here for its later SWIs.
        self.write32(0x03007FF0, address)

    def clear_sound_output(self) -> None:
        """Stop the currently emulated PSG and Direct Sound output."""
        self._psg.master_disabled()
        self.io[0x084] = 0
        self.io[0x080:0x084] = bytes(4)
        self._sound_fifo_a.clear()
        self._sound_fifo_b.clear()
        self._sound_last_a = self._sound_last_b = 0
        self._audio_samples.clear()

    def _start_dma(self, channel: int) -> None:
        if self._dma_active:
            return
        control_address = 0x040000BA + channel * 12
        control = self.read16(control_address)
        if control & 0x8000 and ((control >> 12) & 3) == 0:
            self._run_dma(channel, control)

    def _run_dma(self, channel: int, control: int) -> None:
        if self._dma_active:
            return
        base = 0x040000B0 + channel * 12
        source = self.read32(base)
        destination = self.read32(base + 4)
        initial_source = source
        sound_fifo_dma = channel in (1, 2) and ((control >> 12) & 3) == 3
        count = self.read16(base + 8) & (0xFFFF if channel == 3 else 0x3FFF)
        if count == 0:
            count = 0x10000 if channel == 3 else 0x4000

        if sound_fifo_dma:
            count = 4  # FIFO timing always transfers four 32-bit words.
        word_transfer = sound_fifo_dma or bool(control & (1 << 10))
        width = 4 if word_transfer else 2
        source_mode = (control >> 7) & 3
        destination_mode = 2 if sound_fifo_dma else (control >> 5) & 3
        initial_destination = destination
        source &= ~(width - 1)
        destination &= ~(width - 1)
        read = self.read32 if word_transfer else self.read16
        write = self.write32 if word_transfer else self.write16

        self._dma_active = True
        try:
            for _ in range(count):
                write(destination, read(source))
                if source_mode == 0:
                    source += width
                elif source_mode == 1:
                    source -= width
                # Modes 2 and reserved 3 act fixed in this first pass.
                if destination_mode in (0, 3):
                    destination += width
                elif destination_mode == 1:
                    destination -= width
                source &= 0xFFFFFFFF
                destination &= 0xFFFFFFFF
        finally:
            self._dma_active = False

        if self._is_eeprom_address(initial_source) or self._is_eeprom_address(initial_destination):
            self._finish_eeprom_transfer()

        self.write32(base, source)
        if destination_mode == 3 and (control & (1 << 9)) and ((control >> 12) & 3) != 0:
            self.write32(base + 4, initial_destination)
        else:
            self.write32(base + 4, destination)

        # Immediate DMA is one-shot even if Repeat was requested.
        if not (control & (1 << 9)) or ((control >> 12) & 3) == 0:
            control &= ~(1 << 15)
            offset = 0x04000000 + (0xBA + channel * 12)
            self.io[offset - 0x04000000] = control & 0xFF
            self.io[offset + 1 - 0x04000000] = (control >> 8) & 0xFF

    def set_key(self, key: str, pressed: bool) -> None:
        """Set one GBA button; KEYINPUT bits are active-low."""
        bit = KEY_BITS.get(key.lower())
        if bit is None:
            return
        mask = 1 << bit
        if pressed:
            self.key_input &= ~mask
        else:
            self.key_input |= mask
        self.key_input |= 0xFC00
        self._update_keypad_irq()

    def _update_keypad_irq(self) -> None:
        """Request the keypad IRQ when the KEYCNT button condition is met."""
        control = self._io16(0x132)
        if not (control & 0x8000):
            return
        selected = control & 0x03FF
        if not selected:
            return
        pressed = (~self.key_input) & 0x03FF
        if control & 0x4000:
            matched = (pressed & selected) == selected
        else:
            matched = bool(pressed & selected)
        if matched:
            self._set_io16(0x202, self._io16(0x202) | (1 << 12))

    @_track_bus_access(2, write=True)
    def write16(self, address: int, value: int) -> None:
        """Write a little-endian halfword, aligned as on the GBA."""
        original_address = address & 0xFFFFFFFF
        address &= ~1
        value &= 0xFFFF
        base = address & 0xFF000000
        if base == 0x02000000:
            offset = (address - 0x02000000) & 0x3FFFF
            self.ewram[offset] = value & 0xFF
            self.ewram[(offset + 1) & 0x3FFFF] = value >> 8
            return
        if base == 0x03000000:
            offset = (address - 0x03000000) & 0x7FFF
            self.iwram[offset] = value & 0xFF
            self.iwram[(offset + 1) & 0x7FFF] = value >> 8
            return
        gpio = self._gpio_register_offset(address)
        if gpio is not None and self.cartridge.has_rtc:
            if gpio == 0xC4:
                self._gpio_data = value & 0x0F
                self._rtc_gpio_changed()
            elif gpio == 0xC6:
                self._gpio_direction = value & 0x0F
            elif gpio == 0xC8:
                self._gpio_control = value & 1
            return
        if self.cartridge.save_type == "EEPROM" and self._is_eeprom_address(address):
            self._eeprom_input_bits.append(value & 1)
            return
        if 0x0E000000 <= address <= 0x0FFFFFFF and self.cartridge.save_type in ("SRAM", "FLASH", "FLASH1M"):
            self.write8(address, value >> ((original_address & 1) * 8))
            return
        if 0x05000000 <= address <= 0x05FFFFFF:
            self._mark_video_dirty()
            offset = (address - 0x05000000) & 0x3FF
            self.palette_ram[offset] = value & 0xFF
            self.palette_ram[offset + 1] = value >> 8
            return
        if 0x06000000 <= address <= 0x06FFFFFF:
            self._mark_video_dirty()
            offset = self._mapped_vram_offset(address)
            self.vram[offset] = value & 0xFF
            self.vram[offset + 1] = value >> 8
            return
        if 0x07000000 <= address <= 0x07FFFFFF:
            self._mark_video_dirty()
            offset = (address - 0x07000000) & 0x3FF
            self.oam[offset] = value & 0xFF
            self.oam[offset + 1] = value >> 8
            return
        self.write8(address, value)
        self.write8(address + 1, value >> 8)

    @_track_bus_access(4, write=True)
    def write32(self, address: int, value: int) -> None:
        """Write a little-endian word, aligned as on the GBA."""
        original_address = address & 0xFFFFFFFF
        address &= ~3
        value &= 0xFFFFFFFF
        base = address & 0xFF000000
        if base == 0x02000000:
            data, offset, mask = self.ewram, (address - 0x02000000) & 0x3FFFF, 0x3FFFF
            data[offset] = value & 0xFF
            data[(offset + 1) & mask] = (value >> 8) & 0xFF
            data[(offset + 2) & mask] = (value >> 16) & 0xFF
            data[(offset + 3) & mask] = value >> 24
            return
        if base == 0x03000000:
            data, offset, mask = self.iwram, (address - 0x03000000) & 0x7FFF, 0x7FFF
            data[offset] = value & 0xFF
            data[(offset + 1) & mask] = (value >> 8) & 0xFF
            data[(offset + 2) & mask] = (value >> 16) & 0xFF
            data[(offset + 3) & mask] = value >> 24
            return
        if 0x08000000 <= address <= 0x0DFFFFFF:
            self.write16(address, value)
            self.write16(address + 2, value >> 16)
            return
        if self.cartridge.save_type == "EEPROM" and self._is_eeprom_address(address):
            self.write16(address, value)
            self.write16(address + 2, value >> 16)
            return
        if 0x0E000000 <= address <= 0x0FFFFFFF and self.cartridge.save_type in ("SRAM", "FLASH", "FLASH1M"):
            self.write8(address, value >> ((original_address & 3) * 8))
            return
        if any(
            start <= address <= end
            for start, end in (
                (0x05000000, 0x05FFFFFF),
                (0x06000000, 0x06FFFFFF),
                (0x07000000, 0x07FFFFFF),
            )
        ):
            self.write16(address, value)
            self.write16(address + 2, value >> 16)
            return
        self.write8(address, value)
        self.write8(address + 1, value >> 8)
        self.write8(address + 2, value >> 16)
        self.write8(address + 3, value >> 24)

    @staticmethod
    def _gpio_register_offset(address: int) -> int | None:
        """Return the ROM-window offset for one of the cartridge GPIO regs."""
        offset = (address - 0x08000000) % 0x02000000
        if offset in (0xC4, 0xC6, 0xC8):
            return offset
        if offset in (0xC5, 0xC7, 0xC9):
            return offset - 1
        return None

    def _gpio_register_value(self, register: int) -> int:
        if register == 0xC4:
            pins = self._gpio_data
            if not (self._gpio_direction & 0x02):
                pins = (pins & ~0x02) | (self._rtc_data_output << 1)
            return pins & 0x0F
        if register == 0xC6:
            return self._gpio_direction
        return self._gpio_control

    def _rtc_gpio_changed(self) -> None:
        """Clock the cartridge RTC's simple three-wire serial interface."""
        pins = self._gpio_data & 0x07
        clock = pins & 1
        chip_select = (pins >> 2) & 1
        if not chip_select:
            if clock:
                self._rtc_phase = 1
            else:
                self._rtc_phase = 0
                self._rtc_bit_count = 0
                self._rtc_byte = 0
                self._rtc_tx_bits.clear()
            self._rtc_last_clock = clock
            return
        if self._rtc_phase == 1:
            self._rtc_phase = 2 if clock else 1
            self._rtc_last_clock = clock
            return
        if self._rtc_phase != 2 or clock == self._rtc_last_clock:
            self._rtc_last_clock = clock
            return
        if not clock and not self._rtc_reading:
            # The host presents each SIO bit while SCK is low.
            bit = (pins >> 1) & 1
            self._rtc_byte = (self._rtc_byte & ~(1 << self._rtc_bit_count)) | (bit << self._rtc_bit_count)
        if clock:  # RTC output is driven and the bit advances on the rising edge.
            if self._rtc_reading:
                self._rtc_data_output = self._rtc_tx_bits.popleft() if self._rtc_tx_bits else 0
            self._rtc_bit_count += 1
            if self._rtc_bit_count == 8:
                self._rtc_accept_byte(self._rtc_byte)
                self._rtc_byte = 0
                self._rtc_bit_count = 0
        self._rtc_last_clock = clock

    @staticmethod
    def _bcd(value: int) -> int:
        return ((value // 10) << 4) | (value % 10)

    def _rtc_datetime_bytes(self) -> bytes:
        now = datetime.now().astimezone()
        hour = now.hour
        if not (self._rtc_control & 0x40):
            hour %= 12
        return bytes((
            self._bcd(now.year % 100), self._bcd(now.month), self._bcd(now.day),
            (now.weekday() + 1) % 7, self._bcd(hour), self._bcd(now.minute), self._bcd(now.second),
        ))

    def _rtc_accept_byte(self, value: int) -> None:
        if self._rtc_bytes_left:
            if not self._rtc_reading and self._rtc_command == 4:
                self._rtc_control = value & 0xFF
            self._rtc_bytes_left -= 1
            if not self._rtc_bytes_left:
                self._rtc_command = 0
                self._rtc_reading = False
            return
        if (value & 0x0F) == 0x06:
            self._rtc_command = (value >> 4) & 7
            self._rtc_reading = bool(value & 0x80)
            self._rtc_bytes_left = (0, 0, 7, 0, 1, 0, 3, 0)[self._rtc_command]
            self._rtc_tx_bits.clear()
            if self._rtc_command == 0:
                self._rtc_control = 0
            if self._rtc_command in (2, 6):
                self._rtc_snapshot = self._rtc_datetime_bytes()
            if self._rtc_reading:
                if self._rtc_command == 2:
                    payload = self._rtc_snapshot
                elif self._rtc_command == 4:
                    payload = bytes((self._rtc_control,))
                elif self._rtc_command == 6:
                    payload = self._rtc_snapshot[4:]
                else:
                    payload = b""
                self._rtc_tx_bits = deque((byte >> bit) & 1 for byte in payload for bit in range(8))
            if not self._rtc_bytes_left:
                self._rtc_command = 0
                self._rtc_reading = False
