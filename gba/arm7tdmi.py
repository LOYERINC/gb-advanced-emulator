"""Early ARM7TDMI core with initial ARM and Thumb instruction subsets."""

from __future__ import annotations

from math import atan, atan2, exp2, isqrt, pi, sin
from typing import NamedTuple

from gba.memory import MemoryBus


MASK32 = 0xFFFFFFFF
FLAG_N = 1 << 31
FLAG_Z = 1 << 30
FLAG_C = 1 << 29
FLAG_V = 1 << 28
FLAG_T = 1 << 5
FLAG_F = 1 << 6
FLAG_I = 1 << 7
MODE_MASK = 0x1F
MODE_USER = 0x10
MODE_FIQ = 0x11
MODE_SVC = 0x13
MODE_ABORT = 0x17
MODE_UNDEFINED = 0x1B
MODE_SYSTEM = 0x1F
MODE_IRQ = 0x12
PRIVILEGED_MODES = (MODE_FIQ, MODE_IRQ, MODE_SVC, MODE_ABORT, MODE_UNDEFINED)
VALID_MODES = (MODE_USER, *PRIVILEGED_MODES, MODE_SYSTEM)
_CONDITION_TABLE = tuple(
    (
        z, not z, c, not c, n, not n, v, not v,
        c and not z, not c or z, n == v, n != v,
        not z and n == v, z or n != v, True, False,
    )
    for n, z, c, v in (
        (bool(flags & 8), bool(flags & 4), bool(flags & 2), bool(flags & 1))
        for flags in range(16)
    )
)


class UnsupportedInstruction(RuntimeError):
    """Raised when execution reaches an instruction not implemented yet."""

    def __init__(self, address: int, instruction: int) -> None:
        self.address = address & MASK32
        self.instruction = instruction & MASK32
        super().__init__(
            f"unsupported ARM instruction 0x{self.instruction:08X} "
            f"at 0x{self.address:08X}"
        )


class UndefinedInstruction(UnsupportedInstruction):
    """An opcode that ARM7TDMI sends to its undefined-instruction vector."""


class StepResult(NamedTuple):
    address: int
    instruction: int
    executed: bool
    operation: str
    cycles: int | None = None
    fetch_cycles: int = 0
    data_cycles: int = 0


def _estimated_instruction_cycles(operation: str, instruction: int, *, thumb: bool = False) -> int:
    """Return the runner's interim cycle estimate inside the CPU hot path."""
    if thumb:
        if instruction & 0xF000 == 0xD000:
            condition = (instruction >> 8) & 0xF
            if condition == 0xF:
                return 3 if operation.startswith("SWI") else 2
            if condition == 0xE:
                return 2
            return 3 if operation == "B conditional taken" else 1
        if instruction & 0xF800 == 0xE000 or instruction & 0xF800 == 0xF800:
            return 3
        if instruction & 0xF800 == 0xF000:
            return 1
        if instruction & 0xE000 == 0x6000:
            return 3 if instruction & (1 << 11) else 2
        if instruction & 0xF000 == 0x5000:
            return 2 if (instruction >> 9) & 0x7 < 3 else 3
        if instruction & 0xF000 == 0x8000:
            return 3 if instruction & (1 << 11) else 2
        if instruction & 0xF000 == 0x9000:
            return 3 if instruction & (1 << 11) else 2
        if instruction & 0xF800 == 0x4800:
            return 3
        if instruction & 0xFE00 in (0xB400, 0xBC00):
            pop = bool(instruction & (1 << 11))
            count = (instruction & 0x01FF).bit_count()
            return max(2, count + (2 if pop else 1))
        if instruction & 0xF000 == 0xC000:
            count = (instruction & 0x00FF).bit_count() or 1
            return count + (2 if instruction & (1 << 11) else 1)
        if instruction & 0xFC00 == 0x4400 and (instruction >> 8) & 0x3 == 3:
            return 3
        return 2
    if operation in ("CMP", "ADD", "MOV", "ORR", "SUB", "LSR", "LSL", "AND"):
        return 2
    if operation in ("HALT idle", "IntrWait idle"):
        return 32
    if operation == "condition skipped" or operation == "BL prefix" or operation.endswith("not taken"):
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
        count = (instruction & 0x01FF).bit_count()
        return max(2, count + (2 if operation == "POP" else 1))
    if operation.startswith(("LDM", "STM")):
        mask = instruction & (0x00FF if operation.endswith("(Thumb)") else 0xFFFF)
        count = mask.bit_count() or 1
        return count + (2 if operation.startswith("LDM") else 1)
    return 2


class Arm7Tdmi:
    """A small, incrementally implemented ARM7TDMI CPU core.

    The default startup jumps directly to the cartridge entry point in ARM
    state. When asked to start at the BIOS reset vector, it begins in
    Supervisor ARM mode with IRQ/FIQ masked. The core models ARM7TDMI's
    register banks for supported mode switches and uses the bus's local BIOS
    IRQ trampoline when no real BIOS image is present.
    """

    def __init__(
        self,
        bus: MemoryBus,
        entry_point: int = 0x08000000,
        *,
        start_at_bios: bool = False,
    ) -> None:
        self.bus = bus
        self.registers = [0] * 16
        self.cpsr = (MODE_SVC | FLAG_I | FLAG_F) if start_at_bios else MODE_SYSTEM
        self._user_sp_lr = [0, 0]
        self._mode_sp_lr = {
            MODE_FIQ: [0, 0],
            # The BIOS normally initializes this bank before launching a game.
            MODE_IRQ: [0 if start_at_bios else 0x03007FA0, 0],
            MODE_SVC: [0 if start_at_bios else 0x03007FE0, 0],
            MODE_ABORT: [0, 0],
            MODE_UNDEFINED: [0, 0],
        }
        self._irq_sp_lr = self._mode_sp_lr[MODE_IRQ]
        self._user_r8_r12 = [0] * 5
        self._fiq_r8_r14 = [0] * 7
        self._spsr = {mode: 0 for mode in PRIVILEGED_MODES}
        self._irq_return_pc = 0
        self._intr_wait_mask = 0
        self._halted = False
        self._stopped = False
        self._last_gamepak_fetch: tuple[int, int, bool] | None = None
        if start_at_bios:
            self.registers[15] = 0
        else:
            # Match the useful post-BIOS stack layout for a cartridge started
            # directly, without requiring a copyrighted firmware image.
            self.registers[13] = 0x03007F00
            self.registers[15] = entry_point & MASK32
            # POSTFLG tells software that the BIOS boot phase has completed.
            # A direct-to-cartridge launch skips that write, so reproduce its
            # post-boot state here. Leave it clear when an actual BIOS is about
            # to run; that firmware owns the transition in that path.
            if self.bus.bios_data is None:
                self.bus.write8(0x04000300, 1)
                self.bus.initialize_post_bios_scanline()

    @property
    def thumb_state(self) -> bool:
        return bool(self.cpsr & FLAG_T)

    @property
    def stopped(self) -> bool:
        """Whether BIOS Stop has suspended CPU and system clocks."""
        return self._stopped

    def _flag(self, mask: int) -> bool:
        return bool(self.cpsr & mask)

    def _set_mode(self, mode: int) -> None:
        """Switch ARM7TDMI's banked registers between processor modes."""
        old_mode = self.cpsr & MODE_MASK
        mode &= MODE_MASK
        if old_mode == mode:
            return
        if old_mode == MODE_FIQ:
            self._fiq_r8_r14[:] = self.registers[8:15]
        elif old_mode in self._mode_sp_lr:
            self._mode_sp_lr[old_mode][:] = self.registers[13:15]
        else:
            self._user_sp_lr[:] = self.registers[13:15]

        if mode == MODE_FIQ:
            self._user_r8_r12[:] = self.registers[8:13]
            self.registers[8:15] = self._fiq_r8_r14[:]
        else:
            if old_mode == MODE_FIQ:
                self.registers[8:13] = self._user_r8_r12[:]
            bank = self._mode_sp_lr.get(mode, self._user_sp_lr)
            self.registers[13:15] = bank[:]
        self.cpsr = (self.cpsr & ~MODE_MASK) | mode

    def _restore_cpsr(self, value: int, address: int, instruction: int) -> None:
        mode = value & MODE_MASK
        if mode not in VALID_MODES:
            raise UnsupportedInstruction(address, instruction)
        self._set_mode(mode)
        self.cpsr = value & MASK32

    def _get_spsr(self, mode: int | None = None) -> int:
        mode = (self.cpsr & MODE_MASK) if mode is None else mode & MODE_MASK
        if mode not in self._spsr:
            raise UnsupportedInstruction(self.registers[15], 0)
        return self._spsr[mode]

    def _set_spsr(self, value: int, mode: int | None = None) -> None:
        mode = (self.cpsr & MODE_MASK) if mode is None else mode & MODE_MASK
        if mode not in self._spsr:
            raise UnsupportedInstruction(self.registers[15], 0)
        self._spsr[mode] = value & MASK32

    def _read_user_register(self, index: int, address: int) -> int:
        """Read a register from User/System mode while in a privileged bank."""
        mode = self.cpsr & MODE_MASK
        if index == 15:
            return self._read_register(index, address)
        if mode == MODE_FIQ and 8 <= index <= 12:
            return self._user_r8_r12[index - 8]
        if mode in PRIVILEGED_MODES and index in (13, 14):
            return self._user_sp_lr[index - 13]
        return self.registers[index] & MASK32

    def _write_user_register(self, index: int, value: int) -> None:
        """Write a register in the User/System bank during a privileged transfer."""
        value &= MASK32
        mode = self.cpsr & MODE_MASK
        if mode == MODE_FIQ and 8 <= index <= 12:
            self._user_r8_r12[index - 8] = value
        elif mode in PRIVILEGED_MODES and index in (13, 14):
            self._user_sp_lr[index - 13] = value
        else:
            self.registers[index] = value

    def _enter_irq(self, pending: int | None = None) -> StepResult | None:
        if pending is None:
            pending = self.bus.enabled_interrupt_requests()
        if not pending or self._flag(FLAG_I):
            return None

        interrupted_pc = self.registers[15] & MASK32
        # The BIOS IRQ dispatcher records which sources have occurred here;
        # IntrWait and VBlankIntrWait watch this word.
        check_flags = self.bus.read32(0x03007FF8)
        self.bus.write32(0x03007FF8, check_flags | pending)
        old_cpsr = self.cpsr
        self._set_spsr(old_cpsr, MODE_IRQ)
        self._irq_return_pc = interrupted_pc
        self._set_mode(MODE_IRQ)
        self.registers[14] = (interrupted_pc + 4) & MASK32
        self.cpsr = (self.cpsr | FLAG_I) & ~FLAG_T
        self.registers[15] = 0x18
        return StepResult(interrupted_pc, 0, True, "IRQ entry")

    def _enter_swi(self, address: int, instruction: int, *, was_thumb: bool) -> StepResult:
        """Vector SWI into an optional, user-supplied GBA BIOS image."""
        old_cpsr = self.cpsr
        self._set_spsr(old_cpsr, MODE_SVC)
        self._set_mode(MODE_SVC)
        self.registers[14] = (address + (2 if was_thumb else 4)) & MASK32
        self.cpsr = (self.cpsr | FLAG_I) & ~FLAG_T
        self.registers[15] = 0x08
        return StepResult(address, instruction, True, "SWI exception")

    def _enter_undefined(self, address: int, instruction: int, *, was_thumb: bool) -> StepResult:
        """Enter the ARM7TDMI undefined-instruction exception vector."""
        old_cpsr = self.cpsr
        self._set_spsr(old_cpsr, MODE_UNDEFINED)
        self._set_mode(MODE_UNDEFINED)
        # ARM7TDMI's recommended MOVS PC, LR return uses +4 in ARM state and
        # +2 in Thumb state to resume after the instruction that trapped.
        self.registers[14] = (address + (2 if was_thumb else 4)) & MASK32
        self.cpsr = (self.cpsr | FLAG_I) & ~FLAG_T
        self.registers[15] = 0x04
        return StepResult(address, instruction, True, "Undefined instruction exception")

    def _condition_passes(self, condition: int) -> bool:
        condition &= 0xF
        # Index the condition result using NZCV directly. Building a fresh
        # 16-item tuple for every conditional branch made hot polling loops
        # needlessly expensive.
        return _CONDITION_TABLE[(self.cpsr >> 28) & 0xF][condition]

    def _finish_instruction_fetch(
        self,
        result: StepResult,
        address: int,
        width: int,
        thumb: bool,
    ) -> StepResult:
        extra_cycles = self.bus.gamepak_fetch_extra_cycles(
            address, width, self._last_gamepak_fetch, thumb
        )
        # ALU, branch, and register-only instructions never touch the data
        # bus. Their handlers skip tracking setup, so avoid calling the bus
        # cleanup method for the overwhelmingly common case.
        data_cycles = (
            self.bus.finish_instruction_access_tracking()
            if self.bus._instruction_access_tracking
            else 0
        )
        explicit_cycles = result.cycles is not None
        if explicit_cycles:
            # High-level BIOS replacements already carry their own bulk-work
            # estimate; do not charge those same reads/writes twice.
            data_cycles = 0
        expected_next = (address + width) & MASK32
        branch_taken = result.operation in ("B", "BL") or result.operation == "B conditional taken"
        branch_taken |= result.operation.startswith(("BX", "SWI"))
        remains_sequential = (
            self.registers[15] == expected_next
            and self.thumb_state == thumb
            and not branch_taken
        )
        self._last_gamepak_fetch = (address, width, thumb) if remains_sequential else None
        cycles = result.cycles
        if cycles is None:
            cycles = _estimated_instruction_cycles(result.operation, result.instruction, thumb=thumb)
        return result._replace(cycles=cycles, fetch_cycles=extra_cycles, data_cycles=data_cycles)

    def _read_register(self, index: int, instruction_address: int) -> int:
        # In ARM state, reads of r15 see the current instruction address + 8.
        if index == 15:
            return (instruction_address + 8) & MASK32
        return self.registers[index] & MASK32

    def _set_nz(self, result: int) -> None:
        result &= MASK32
        self.cpsr = (self.cpsr & ~(FLAG_N | FLAG_Z))
        if result & 0x80000000:
            self.cpsr |= FLAG_N
        if result == 0:
            self.cpsr |= FLAG_Z

    def _shift_operand(self, instruction: int, instruction_address: int) -> tuple[int, bool]:
        """Decode ARM's immediate or immediate-shifted-register operand."""
        old_carry = self._flag(FLAG_C)
        if instruction & (1 << 25):
            value = instruction & 0xFF
            rotation = ((instruction >> 8) & 0xF) * 2
            if rotation == 0:
                return value, old_carry
            result = ((value >> rotation) | (value << (32 - rotation))) & MASK32
            return result, bool(result & 0x80000000)

        by_register = bool(instruction & (1 << 4))
        source_register = instruction & 0xF
        if by_register and source_register == 15:
            # ARM7TDMI exposes PC+12 as Rm when the barrel-shift amount is
            # supplied by a register; the usual data-processing PC value is +8.
            value = (instruction_address + 12) & MASK32
        else:
            value = self._read_register(source_register, instruction_address)
        shift_type = (instruction >> 5) & 0x3
        if by_register:
            if instruction & (1 << 7):
                raise UnsupportedInstruction(instruction_address, instruction)
            shift_register = (instruction >> 8) & 0xF
            if shift_register == 15:
                raise UnsupportedInstruction(instruction_address, instruction)
            amount = self.registers[shift_register] & 0xFF
            if amount == 0:
                return value, old_carry
        else:
            amount = (instruction >> 7) & 0x1F

        if shift_type == 0:  # LSL
            if amount == 0:
                return value, old_carry
            if amount == 32:
                return 0, bool(value & 1)
            if amount > 32:
                return 0, False
            return (value << amount) & MASK32, bool(value & (1 << (32 - amount)))

        if shift_type == 1:  # LSR; an encoded zero means 32 places
            if amount == 0 and not by_register:
                amount = 32
            if amount == 32:
                return 0, bool(value & 0x80000000)
            if amount > 32:
                return 0, False
            return value >> amount, bool(value & (1 << (amount - 1)))

        if shift_type == 2:  # ASR; an encoded zero means 32 places
            if amount == 0 and not by_register:
                amount = 32
            sign = bool(value & 0x80000000)
            if amount >= 32:
                return (MASK32 if sign else 0), sign
            signed_value = value if not sign else value - (1 << 32)
            return (signed_value >> amount) & MASK32, bool(value & (1 << (amount - 1)))

        # Register ROR by a multiple of 32 preserves the value and sets carry
        # from bit 31; immediate ROR with zero is RRX.
        if amount == 0:
            result = (int(old_carry) << 31) | (value >> 1)
            return result & MASK32, bool(value & 1)
        rotation = amount & 0x1F
        if by_register and rotation == 0:
            return value, bool(value & 0x80000000)
        result = ((value >> rotation) | (value << (32 - rotation))) & MASK32
        return result, bool(result & 0x80000000)

    def _set_arithmetic_flags(self, result: int, carry: bool, overflow: bool) -> None:
        self._set_nz(result)
        self.cpsr &= ~(FLAG_C | FLAG_V)
        if carry:
            self.cpsr |= FLAG_C
        if overflow:
            self.cpsr |= FLAG_V

    def _transfer_offset(self, instruction: int, address: int) -> int:
        """Get the 12-bit immediate or shifted-register address offset."""
        if not instruction & (1 << 25):
            return instruction & 0xFFF

        if instruction & (1 << 4):
            raise UnsupportedInstruction(address, instruction)
        value = self._read_register(instruction & 0xF, address)
        shift_type = (instruction >> 5) & 0x3
        amount = (instruction >> 7) & 0x1F

        if shift_type == 0:  # LSL
            return (value << amount) & MASK32
        if shift_type == 1:  # LSR
            return 0 if amount == 0 else value >> amount
        if shift_type == 2:  # ASR
            signed_value = value if value < 0x80000000 else value - (1 << 32)
            return (signed_value >> (amount or 32)) & MASK32
        if amount == 0:  # RRX
            return ((int(self._flag(FLAG_C)) << 31) | (value >> 1)) & MASK32
        return ((value >> amount) | (value << (32 - amount))) & MASK32

    def _execute_single_data_transfer(self, instruction: int, address: int) -> StepResult:
        register_offset = bool(instruction & (1 << 25))
        pre_index = bool(instruction & (1 << 24))
        add_offset = bool(instruction & (1 << 23))
        byte_access = bool(instruction & (1 << 22))
        writeback_bit = bool(instruction & (1 << 21))
        load = bool(instruction & (1 << 20))
        rn = (instruction >> 16) & 0xF
        rd = (instruction >> 12) & 0xF

        # Register offsets permit an immediate shift, never a shift amount in a register.
        offset = self._transfer_offset(instruction, address) if register_offset else instruction & 0xFFF
        base = self._read_register(rn, address)
        adjusted = (base + offset if add_offset else base - offset) & MASK32
        effective_address = adjusted if pre_index else base
        writeback = writeback_bit or not pre_index

        if writeback and (rn == 15 or (load and rn == rd)):
            raise UnsupportedInstruction(address, instruction)

        if load:
            if byte_access:
                value = self.bus.read8(effective_address)
            else:
                value = self.bus.read32(effective_address)
                rotation = (effective_address & 3) * 8
                if rotation:
                    value = ((value >> rotation) | (value << (32 - rotation))) & MASK32
            if rd == 15:
                self.registers[15] = value & ~3
            else:
                self.registers[rd] = value & MASK32
        else:
            value = self._read_register(rd, address)
            if rd == 15:
                value = (address + 12) & MASK32
            if byte_access:
                self.bus.write8(effective_address, value)
            else:
                self.bus.write32(effective_address, value)

        if writeback:
            self.registers[rn] = adjusted

        operation = "LDR" if load else "STR"
        if byte_access:
            operation += "B"
        return StepResult(address, instruction, True, operation)

    def _execute_halfword_transfer(self, instruction: int, address: int) -> StepResult:
        pre_index = bool(instruction & (1 << 24))
        add_offset = bool(instruction & (1 << 23))
        immediate = bool(instruction & (1 << 22))
        writeback_bit = bool(instruction & (1 << 21))
        load = bool(instruction & (1 << 20))
        rn = (instruction >> 16) & 0xF
        rd = (instruction >> 12) & 0xF
        signed = bool(instruction & (1 << 6))
        halfword = bool(instruction & (1 << 5))

        if immediate:
            offset = (((instruction >> 8) & 0xF) << 4) | (instruction & 0xF)
        else:
            offset = self._read_register(instruction & 0xF, address)

        base = self._read_register(rn, address)
        adjusted = (base + offset if add_offset else base - offset) & MASK32
        location = adjusted if pre_index else base
        writeback = writeback_bit or not pre_index
        if writeback and (rn == 15 or (load and rn == rd)):
            raise UnsupportedInstruction(address, instruction)

        if not halfword and not signed:
            raise UnsupportedInstruction(address, instruction)
        if signed and not load:
            raise UnsupportedInstruction(address, instruction)
        if rd == 15 and (not load or signed):
            raise UnsupportedInstruction(address, instruction)

        if load:
            if halfword:
                value = self._load_arm7_halfword(location, signed=signed)
            else:
                value = self.bus.read8(location)
                if signed and value & 0x80:
                    value -= 0x100
            if rd == 15:
                self.registers[15] = value & ~3
            else:
                self.registers[rd] = value & MASK32
        else:
            self.bus.write16(location, self._read_register(rd, address))

        if writeback:
            self.registers[rn] = adjusted
        if signed:
            name = "LDRSH" if halfword else "LDRSB"
        else:
            name = "LDRH" if load else "STRH"
        return StepResult(address, instruction, True, name)

    def _execute_block_transfer(self, instruction: int, address: int) -> StepResult:
        pre_index = bool(instruction & (1 << 24))
        increment = bool(instruction & (1 << 23))
        user_bank = bool(instruction & (1 << 22))
        writeback = bool(instruction & (1 << 21))
        load = bool(instruction & (1 << 20))
        base_register = (instruction >> 16) & 0xF
        register_list = instruction & 0xFFFF
        selected = [reg for reg in range(16) if register_list & (1 << reg)]
        empty_register_list = not selected
        if empty_register_list:
            # ARMv4T transfers PC for an empty list, but writeback still moves
            # the base as though all sixteen registers had been selected.
            selected = [15]
        if writeback and base_register == 15:
            raise UnsupportedInstruction(address, instruction)
        exception_return = load and user_bank and 15 in selected
        current_mode = self.cpsr & MODE_MASK
        if exception_return and current_mode not in PRIVILEGED_MODES:
            raise UnsupportedInstruction(address, instruction)
        transfer_user_bank = user_bank and not exception_return

        base = self._read_register(base_register, address)
        count = len(selected)
        writeback_count = 16 if empty_register_list else count
        if increment:
            first_address = base + (4 if pre_index else 0)
            updated_base = base + writeback_count * 4
            name = "LDMIB" if load and pre_index else "STMIB" if pre_index else "LDMIA" if load else "STMIA"
        else:
            first_address = base - count * 4 + (0 if pre_index else 4)
            updated_base = base - writeback_count * 4
            name = "LDMDB" if load and pre_index else "STMDB" if pre_index else "LDMDA" if load else "STMDA"

        location = first_address & MASK32
        for register in selected:
            if load:
                value = self.bus.read32(location)
                if register == 15:
                    if exception_return:
                        self.registers[15] = value & MASK32
                    else:
                        self.registers[15] = value & ~3
                elif transfer_user_bank:
                    self._write_user_register(register, value)
                else:
                    self.registers[register] = value
            else:
                if register == 15:
                    value = (address + 12) & MASK32
                elif register == base_register and writeback and not transfer_user_bank:
                    # ARMv4 STM writes the old base if it is the first
                    # register, otherwise the updated base value.
                    value = base if selected[0] == base_register else updated_base
                elif transfer_user_bank:
                    value = self._read_user_register(register, address)
                else:
                    value = self.registers[register]
                self.bus.write32(location, value)
            location = (location + 4) & MASK32

        if writeback and not (load and base_register in selected):
            # ARMv4 LDM with Rn in the list leaves Rn as the loaded value.
            self.registers[base_register] = updated_base & MASK32
        if exception_return:
            # ARM's LDM^ form with PC restores CPSR from the current mode's
            # SPSR. The restored T bit decides whether PC is halfword/word
            # aligned and which instruction set executes next.
            target = self.registers[15]
            self._restore_cpsr(self._get_spsr(current_mode), address, instruction)
            self.registers[15] = target & (~1 if self.thumb_state else ~3) & MASK32
        return StepResult(address, instruction, True, name)

    @staticmethod
    def _multiply_internal_cycles(value: int, *, signed: bool) -> int:
        value &= MASK32
        if signed:
            if (value >> 8) in (0, 0xFFFFFF):
                return 1
            if (value >> 16) in (0, 0xFFFF):
                return 2
            if (value >> 24) in (0, 0xFF):
                return 3
        else:
            if value >> 8 == 0:
                return 1
            if value >> 16 == 0:
                return 2
            if value >> 24 == 0:
                return 3
        return 4

    def _execute_multiply(self, instruction: int, address: int) -> StepResult:
        if instruction & 0x0F8000F0 == 0x00800090:
            signed = bool(instruction & (1 << 22))
            accumulate = bool(instruction & (1 << 21))
            set_flags = bool(instruction & (1 << 20))
            high_register = (instruction >> 16) & 0xF
            low_register = (instruction >> 12) & 0xF
            source_register = (instruction >> 8) & 0xF
            operand_register = instruction & 0xF
            if high_register == 15 or low_register == 15 or high_register == low_register:
                raise UnsupportedInstruction(address, instruction)
            left = self.registers[operand_register] & MASK32
            right = self.registers[source_register] & MASK32
            multiply_cycles = self._multiply_internal_cycles(right, signed=signed)
            if signed:
                if left & 0x80000000:
                    left -= 1 << 32
                if right & 0x80000000:
                    right -= 1 << 32
            result = left * right
            if accumulate:
                result += ((self.registers[high_register] & MASK32) << 32) | (self.registers[low_register] & MASK32)
            result &= 0xFFFFFFFFFFFFFFFF
            self.registers[low_register] = result & MASK32
            self.registers[high_register] = (result >> 32) & MASK32
            if set_flags:
                self.cpsr &= ~(FLAG_N | FLAG_Z)
                if result & (1 << 63):
                    self.cpsr |= FLAG_N
                if result == 0:
                    self.cpsr |= FLAG_Z
            operation = "SMLAL" if signed and accumulate else "SMULL" if signed else "UMLAL" if accumulate else "UMULL"
            extra_cycles = 2 if accumulate else 1
            return StepResult(address, instruction, True, operation, 1 + multiply_cycles + extra_cycles)

        accumulate = bool(instruction & (1 << 21))
        set_flags = bool(instruction & (1 << 20))
        destination = (instruction >> 16) & 0xF
        accumulate_register = (instruction >> 12) & 0xF
        source_register = (instruction >> 8) & 0xF
        operand_register = instruction & 0xF
        if destination == 15 or source_register == 15 or operand_register == 15:
            raise UnsupportedInstruction(address, instruction)
        multiply_cycles = self._multiply_internal_cycles(self.registers[source_register], signed=True)
        result = (self.registers[operand_register] * self.registers[source_register]) & MASK32
        if accumulate:
            result = (result + self.registers[accumulate_register]) & MASK32
        self.registers[destination] = result
        if set_flags:
            self._set_nz(result)
        operation = "MLA" if accumulate else "MUL"
        return StepResult(address, instruction, True, operation, 1 + multiply_cycles + int(accumulate))

    def _execute_swap(self, instruction: int, address: int) -> StepResult:
        byte_access = bool(instruction & (1 << 22))
        base_register = (instruction >> 16) & 0xF
        destination = (instruction >> 12) & 0xF
        source = instruction & 0xF
        if base_register == 15 or destination == 15 or source == 15:
            raise UnsupportedInstruction(address, instruction)
        location = self.registers[base_register] & MASK32
        if byte_access:
            old_value = self.bus.read8(location)
            self.bus.write8(location, self.registers[source])
        else:
            old_value = self.bus.read32(location)
            self.bus.write32(location, self.registers[source])
            rotation = (location & 3) * 8
            if rotation:
                old_value = ((old_value >> rotation) | (old_value << (32 - rotation))) & MASK32
        self.registers[destination] = old_value
        return StepResult(address, instruction, True, "SWPB" if byte_access else "SWP")

    @staticmethod
    def _signed32(value: int) -> int:
        value &= MASK32
        return value - (1 << 32) if value & 0x80000000 else value

    @staticmethod
    def _signed16(value: int) -> int:
        value &= 0xFFFF
        return value - 0x10000 if value & 0x8000 else value

    @staticmethod
    def _sine_q14(index: int) -> int:
        return int(round(sin((index & 0xFF) * (2 * pi / 256)) * 0x4000))

    def _bios_bg_affine_set(self) -> None:
        source = self.registers[0] & MASK32
        destination = self.registers[1] & MASK32
        count = self.registers[2] & MASK32
        for _ in range(count):
            center_x = self._signed32(self.bus.read32(source))
            center_y = self._signed32(self.bus.read32(source + 4))
            display_x = self._signed16(self.bus.read16(source + 8))
            display_y = self._signed16(self.bus.read16(source + 10))
            scale_x = self._signed16(self.bus.read16(source + 12))
            scale_y = self._signed16(self.bus.read16(source + 14))
            angle = self.bus.read16(source + 16) >> 8
            cosine = self._sine_q14(angle + 0x40)
            sine = self._sine_q14(angle)

            pa = (scale_x * cosine) >> 14
            pb_magnitude = (scale_x * sine) >> 14
            pc = (scale_y * sine) >> 14
            pd = (scale_y * cosine) >> 14
            self.bus.write16(destination, pa)
            self.bus.write16(destination + 2, -pb_magnitude)
            self.bus.write16(destination + 4, pc)
            self.bus.write16(destination + 6, pd)
            start_x = (center_x - pa * display_x + pb_magnitude * display_y) & MASK32
            start_y = (center_y - pc * display_x - pd * display_y) & MASK32
            self.bus.write32(destination + 8, start_x)
            self.bus.write32(destination + 12, start_y)
            source += 20
            destination += 28

    def _bios_obj_affine_set(self) -> None:
        source = self.registers[0] & MASK32
        destination = self.registers[1] & MASK32
        count = self.registers[2] & MASK32
        stride = self.registers[3] & MASK32
        for _ in range(count):
            scale_x = self._signed16(self.bus.read16(source))
            scale_y = self._signed16(self.bus.read16(source + 2))
            angle = self.bus.read16(source + 4) >> 8
            cosine = self._sine_q14(angle + 0x40)
            sine = self._sine_q14(angle)
            pa = (scale_x * cosine) >> 14
            pb_magnitude = (scale_x * sine) >> 14
            pc = (scale_y * sine) >> 14
            pd = (scale_y * cosine) >> 14
            for offset, value in enumerate((pa, -pb_magnitude, pc, pd)):
                self.bus.write16(destination + offset * stride, value)
            source += 8
            destination += 4 * stride

    def _decompress_lz77(self, source: int) -> bytearray:
        header = self.bus.read32(source)
        if header & 0xF0 != 0x10:
            raise ValueError("not GBA LZ77 data")
        output_size = (header >> 8) & 0xFFFFFF
        cursor = source + 4
        output = bytearray()
        while len(output) < output_size:
            flags = self.bus.read8(cursor)
            cursor += 1
            for bit in range(7, -1, -1):
                if len(output) >= output_size:
                    break
                if flags & (1 << bit):
                    first = self.bus.read8(cursor)
                    second = self.bus.read8(cursor + 1)
                    cursor += 2
                    run_length = (first >> 4) + 3
                    distance = (((first & 0x0F) << 8) | second) + 1
                    if distance > len(output):
                        raise ValueError("invalid GBA LZ77 back-reference")
                    for _ in range(min(run_length, output_size - len(output))):
                        output.append(output[-distance])
                else:
                    output.append(self.bus.read8(cursor))
                    cursor += 1
        return output

    def _decompress_rl(self, source: int) -> bytearray:
        header = self.bus.read32(source)
        if header & 0xF0 != 0x30:
            raise ValueError("not GBA run-length data")
        output_size = (header >> 8) & 0xFFFFFF
        cursor = source + 4
        output = bytearray()
        while len(output) < output_size:
            control = self.bus.read8(cursor)
            cursor += 1
            if control & 0x80:
                value = self.bus.read8(cursor)
                cursor += 1
                output.extend(bytes((value,)) * min((control & 0x7F) + 3, output_size - len(output)))
            else:
                count = min((control & 0x7F) + 1, output_size - len(output))
                output.extend(self.bus.read8(cursor + index) for index in range(count))
                cursor += count
        return output

    def _bit_unpack(self, source: int, info_address: int) -> bytearray:
        source_length = self.bus.read16(info_address)
        source_width = self.bus.read8(info_address + 2)
        destination_width = self.bus.read8(info_address + 3)
        offset_info = self.bus.read32(info_address + 4)
        valid_source_widths = (1, 2, 4, 8)
        valid_destination_widths = (1, 2, 4, 8, 16, 32)
        if (
            source_width not in valid_source_widths
            or destination_width not in valid_destination_widths
            or destination_width < source_width
            or (source_length * 8 * destination_width // source_width) % 32
        ):
            raise ValueError("invalid GBA bit-unpack parameters")

        mask = (1 << destination_width) - 1 if destination_width < 32 else MASK32
        data_offset = offset_info & 0x7FFFFFFF
        add_to_zero = bool(offset_info & 0x80000000)
        output = bytearray()
        word = 0
        bit_position = 0
        source_mask = (1 << source_width) - 1
        units_per_byte = 8 // source_width
        for byte_index in range(source_length):
            source_byte = self.bus.read8(source + byte_index)
            for unit_index in range(units_per_byte):
                value = (source_byte >> (unit_index * source_width)) & source_mask
                if value or add_to_zero:
                    value = (value + data_offset) & mask
                word |= value << bit_position
                bit_position += destination_width
                if bit_position == 32:
                    output.extend(word.to_bytes(4, "little"))
                    word = 0
                    bit_position = 0
        return output

    def _difference_unpack(self, number: int, source: int) -> bytearray:
        header = self.bus.read32(source)
        unit_width = 16 if number == 0x18 else 8
        unit_bytes = unit_width // 8
        if header & 0xF0 != 0x80 or (header & 0xF) != unit_bytes:
            raise ValueError("invalid GBA difference-filter header")
        output_size = (header >> 8) & 0xFFFFFF
        if output_size % unit_bytes:
            raise ValueError("invalid GBA difference-filter size")
        cursor = source + 4
        output = bytearray()
        previous = 0
        mask = (1 << unit_width) - 1
        for index in range(output_size // unit_bytes):
            difference = self.bus.read8(cursor) if unit_width == 8 else self.bus.read16(cursor)
            cursor += unit_bytes
            previous = (previous + difference) & mask
            output.extend(previous.to_bytes(unit_bytes, "little"))
        return output

    def _decompress_huffman(self, source: int) -> bytearray:
        source &= ~3
        header = self.bus.read32(source)
        symbol_width = header & 0xF
        output_size = (header >> 8) & 0xFFFFFF
        if (header >> 4) & 0xF != 2 or symbol_width not in (4, 8) or output_size % 4:
            raise ValueError("invalid GBA Huffman header")

        tree_size = (self.bus.read8(source + 4) << 1) + 1
        tree_base = source + 5
        tree_end = tree_base + tree_size
        bitstream = (tree_end + 3) & ~3
        root_node = self.bus.read8(tree_base)
        symbols_needed = (output_size * 8) // symbol_width
        output = bytearray()
        word = 0
        word_bits = 0
        bits_consumed = 0
        word_mask = (1 << symbol_width) - 1

        for _ in range(symbols_needed):
            node = root_node
            pointer = tree_base
            leaf_value = None
            # The tree has at most 256 nodes; this also protects against a bad cycle.
            for _depth in range(256):
                node = self.bus.read8(pointer)
                next_node = (pointer & ~1) + (node & 0x3F) * 2 + 2
                byte_index = bits_consumed // 32
                bit_index = 31 - (bits_consumed % 32)
                bits = self.bus.read32(bitstream + byte_index * 4)
                branch_right = bool(bits & (1 << bit_index))
                bits_consumed += 1
                if branch_right:
                    if node & 0x40:
                        leaf_value = self.bus.read8(next_node + 1)
                        break
                    pointer = next_node + 1
                else:
                    if node & 0x80:
                        leaf_value = self.bus.read8(next_node)
                        break
                    pointer = next_node
                if pointer < tree_base or pointer >= tree_end:
                    raise ValueError("invalid GBA Huffman tree link")
            if leaf_value is None:
                raise ValueError("GBA Huffman tree did not reach a leaf")

            symbol = leaf_value & word_mask
            word |= symbol << word_bits
            word_bits += symbol_width
            if word_bits == 32:
                output.extend(word.to_bytes(4, "little"))
                word = 0
                word_bits = 0
        if len(output) != output_size:
            raise ValueError("invalid GBA Huffman output size")
        return output

    def _bios_soft_reset(self) -> None:
        """Reset the CPU and jump to the BIOS-selected reset target."""
        use_ewram = self.bus.read8(0x03007FFA) != 0
        self.bus.iwram[-0x200:] = bytes(0x200)
        self.registers[:] = [0] * 16
        self.cpsr = MODE_SYSTEM
        self._user_sp_lr[:] = [0, 0]
        for bank in self._mode_sp_lr.values():
            bank[:] = [0, 0]
        self._irq_sp_lr[:] = [0x03007FA0, 0]
        self._user_r8_r12[:] = [0] * 5
        self._fiq_r8_r14[:] = [0] * 7
        for mode in self._spsr:
            self._spsr[mode] = 0
        self._irq_return_pc = 0
        self._intr_wait_mask = 0
        self._halted = False
        self._stopped = False
        self.registers[13] = 0x03007F00
        self.registers[15] = 0x02000000 if use_ewram else 0x08000000

    def _execute_bios_call(self, number: int, address: int, instruction: int) -> StepResult:
        """Handle a small set of common BIOS calls without bundled BIOS data."""
        if number == 0x00:  # SoftReset
            self._bios_soft_reset()
            return StepResult(address, instruction, True, "SWI SoftReset")

        if number == 0x26:  # HardReset
            # A desktop HLE cannot power-cycle the console or replay its boot
            # animation; restart the cartridge through the existing reset path.
            self._bios_soft_reset()
            return StepResult(address, instruction, True, "SWI HardReset (reset approximation)")

        if number == 0x01:  # RegisterRamReset
            self.bus.register_ram_reset(self.registers[0])
            return StepResult(address, instruction, True, "SWI RegisterRamReset")

        if number == 0x02:  # Halt
            self._halted = True
            return StepResult(address, instruction, True, "SWI Halt")

        if number == 0x03:  # Stop
            self._stopped = True
            return StepResult(address, instruction, True, "SWI Stop")

        if number == 0x27:  # CustomHalt: HALTCNT mode is supplied in R2.
            halt_mode = self.registers[2] & 0xFF
            if halt_mode == 0x00:
                self._halted = True
                operation = "SWI CustomHalt (Halt)"
            elif halt_mode == 0x80:
                self._stopped = True
                operation = "SWI CustomHalt (Stop)"
            else:
                operation = "SWI CustomHalt (reserved mode)"
            return StepResult(address, instruction, True, operation)

        if number in (0x04, 0x05):  # IntrWait / VBlankIntrWait
            if number == 0x05:
                mask = 1  # IRQ_VBLANK
                discard_old = True
            else:
                mask = self.registers[1] & 0x3FFF
                discard_old = bool(self.registers[0] & 1)
            flags = self.bus.read32(0x03007FF8)
            if discard_old:
                flags &= ~mask
                self.bus.write32(0x03007FF8, flags)
            self._intr_wait_mask = mask if not (flags & mask) else 0
            return StepResult(address, instruction, True, "SWI IntrWait" if number == 0x04 else "SWI VBlankIntrWait")

        if number in (0x06, 0x07):  # Div / DivArm
            if number == 0x06:
                numerator = self._signed32(self.registers[0])
                denominator = self._signed32(self.registers[1])
            else:
                denominator = self._signed32(self.registers[0])
                numerator = self._signed32(self.registers[1])
            if denominator == 0:
                raise UnsupportedInstruction(address, instruction)
            quotient = abs(numerator) // abs(denominator)
            if (numerator < 0) != (denominator < 0):
                quotient = -quotient
            remainder = numerator - quotient * denominator
            self.registers[0] = quotient & MASK32
            self.registers[1] = remainder & MASK32
            self.registers[3] = abs(quotient) & MASK32
            return StepResult(address, instruction, True, "SWI Div" if number == 0x06 else "SWI DivArm")

        if number == 0x08:  # Sqrt
            self.registers[0] = isqrt(self.registers[0] & MASK32)
            return StepResult(address, instruction, True, "SWI Sqrt")

        if number == 0x09:  # ArcTan, signed 1.14 input -> signed 1.14 radians/pi
            tangent = self.registers[0] & 0xFFFF
            if tangent & 0x8000:
                tangent -= 0x10000
            angle = atan(tangent / 16384.0)
            self.registers[0] = round(angle * (32768.0 / pi)) & MASK32
            return StepResult(address, instruction, True, "SWI ArcTan")

        if number == 0x0A:  # ArcTan2, signed 1.14 X/Y -> unsigned 16-bit turn
            x = self.registers[0] & 0xFFFF
            y = self.registers[1] & 0xFFFF
            if x & 0x8000:
                x -= 0x10000
            if y & 0x8000:
                y -= 0x10000
            angle = atan2(y, x) % (2.0 * pi)
            self.registers[0] = round(angle * (32768.0 / pi)) & 0xFFFF
            return StepResult(address, instruction, True, "SWI ArcTan2")

        if number == 0x0D:  # GetBiosChecksum
            self.registers[0] = 0xBAAE187F
            return StepResult(address, instruction, True, "SWI GetBiosChecksum")

        if number == 0x19:  # SoundBias
            bias = self.bus.read16(0x04000088)
            level = 0x0200 if self.registers[0] else 0
            self.bus.write16(0x04000088, (bias & ~0x03FF) | level)
            return StepResult(address, instruction, True, "SWI SoundBias")

        if number == 0x1A:  # SoundDriverInit (work-area compatibility stub)
            self.bus.initialize_bios_sound_area(self.registers[0])
            return StepResult(address, instruction, True, "SWI SoundDriverInit")

        if number in (0x1B, 0x1C, 0x1D, 0x28, 0x29, 0x2A):
            # The BIOS MPlay software mixer is not implemented. Recognize its
            # setup/frame-sync calls so they do not stop game logic as unknown SWIs.
            names = {
                0x1B: "SoundDriverMode",
                0x1C: "SoundDriverMain",
                0x1D: "SoundDriverVSync",
                0x28: "SoundDriverVSyncOff",
                0x29: "SoundDriverVSyncOn",
                0x2A: "SoundGetJumpList",
            }
            return StepResult(address, instruction, True, f"SWI {names[number]} (stub)")

        if number in (0x20, 0x21, 0x22, 0x23, 0x24):
            # MPlay is the BIOS software music player; recognize its calls so
            # games can continue even though this build has no MPlay mixer.
            names = {
                0x20: "MusicPlayerOpen",
                0x21: "MusicPlayerStart",
                0x22: "MusicPlayerStop",
                0x23: "MusicPlayerContinue",
                0x24: "MusicPlayerFadeOut",
            }
            return StepResult(address, instruction, True, f"SWI {names[number]} (silent stub)")

        if number == 0x1E:  # SoundChannelClear
            self.bus.clear_sound_output()
            return StepResult(address, instruction, True, "SWI SoundChannelClear")

        if number == 0x1F:  # MidiKey2Freq
            wave_data = self.registers[0] & MASK32
            base_frequency = self.bus.read32(wave_data + 4)
            midi_key = self.registers[1]
            fine_tune = self.registers[2]
            exponent = (180.0 - midi_key - fine_tune / 256.0) / 12.0
            self.registers[0] = int(base_frequency / exp2(exponent)) & MASK32
            return StepResult(address, instruction, True, "SWI MidiKey2Freq")

        if number == 0x25:  # MultiBoot
            # Link-cable transfers are not emulated. Return the documented
            # failure code so callers can fall back to the cartridge game.
            self.registers[0] = 1
            return StepResult(address, instruction, True, "SWI MultiBoot (unavailable)")

        if number == 0x0E:  # BgAffineSet
            self._bios_bg_affine_set()
            return StepResult(address, instruction, True, "SWI BgAffineSet")

        if number == 0x0F:  # ObjAffineSet
            self._bios_obj_affine_set()
            return StepResult(address, instruction, True, "SWI ObjAffineSet")

        if number == 0x10:  # BitUnPack
            try:
                output = self._bit_unpack(self.registers[0] & MASK32, self.registers[2] & MASK32)
            except ValueError:
                raise UnsupportedInstruction(address, instruction) from None
            destination = self.registers[1] & ~3
            for offset in range(0, len(output), 4):
                self.bus.write32(destination + offset, int.from_bytes(output[offset : offset + 4], "little"))
            return StepResult(address, instruction, True, "SWI BitUnPack", 3 + len(output) * 3)

        if number == 0x13:  # HuffUnComp
            try:
                output = self._decompress_huffman(self.registers[0] & MASK32)
            except ValueError:
                raise UnsupportedInstruction(address, instruction) from None
            destination = self.registers[1] & ~3
            for offset in range(0, len(output), 4):
                self.bus.write32(destination + offset, int.from_bytes(output[offset : offset + 4], "little"))
            return StepResult(address, instruction, True, "SWI HuffUnComp", 3 + len(output) * 4)

        if number in (0x16, 0x17, 0x18):  # Diff8/16 unfilter
            try:
                output = self._difference_unpack(number, self.registers[0] & MASK32)
            except ValueError:
                raise UnsupportedInstruction(address, instruction) from None
            destination = self.registers[1] & MASK32
            if number == 0x17:
                destination &= ~1
                for offset in range(0, len(output), 2):
                    low = output[offset]
                    high = output[offset + 1] if offset + 1 < len(output) else 0
                    self.bus.write16(destination + offset, low | (high << 8))
            else:
                for offset, value in enumerate(output):
                    self.bus.write8(destination + offset, value)
            name = {0x16: "SWI Diff8bitUnFilterWram", 0x17: "SWI Diff8bitUnFilterVram", 0x18: "SWI Diff16bitUnFilter"}[number]
            return StepResult(address, instruction, True, name, 3 + len(output) * 3)

        if number in (0x11, 0x12, 0x14, 0x15):  # LZ77/RL, WRAM/VRAM destinations
            source = self.registers[0] & MASK32
            destination = self.registers[1] & MASK32
            is_lz77 = number in (0x11, 0x12)
            to_vram = number in (0x12, 0x15)
            try:
                output = self._decompress_lz77(source) if is_lz77 else self._decompress_rl(source)
            except ValueError:
                raise UnsupportedInstruction(address, instruction) from None
            if to_vram:
                destination &= ~1
                for offset in range(0, len(output), 2):
                    low = output[offset]
                    high = output[offset + 1] if offset + 1 < len(output) else 0
                    self.bus.write16(destination + offset, low | (high << 8))
            else:
                for offset, value in enumerate(output):
                    self.bus.write8(destination + offset, value)
            operation = ("SWI LZ77UnComp" if is_lz77 else "SWI RLUnComp") + ("Vram" if to_vram else "Wram")
            cycles_per_byte = 4 if is_lz77 else 2
            return StepResult(address, instruction, True, operation, 3 + len(output) * cycles_per_byte)

        if number in (0x0B, 0x0C):  # CpuSet / CpuFastSet
            source = self.registers[0] & MASK32
            destination = self.registers[1] & MASK32
            mode = self.registers[2] & MASK32
            fixed_source = bool(mode & (1 << 24))
            if number == 0x0C:
                word_count = mode & 0x1FFFFF
                word_count = (word_count + 7) & ~7  # GBA CpuFastSet works in 32-byte blocks.
                unit = 4
            else:
                unit = 4 if mode & (1 << 26) else 2
                word_count = mode & 0x1FFFFF
            source &= ~(unit - 1)
            destination &= ~(unit - 1)
            if not word_count:
                return StepResult(address, instruction, True, "SWI CpuFastSet" if number == 0x0C else "SWI CpuSet")
            read = self.bus.read32 if unit == 4 else self.bus.read16
            write = self.bus.write32 if unit == 4 else self.bus.write16
            value = read(source)
            for index in range(word_count):
                if not fixed_source and index:
                    value = read(source + index * unit)
                write(destination + index * unit, value)
            cycles_per_unit = 3 if not fixed_source else 2
            if number == 0x0C:
                cycles_per_unit = 3  # Eight-word block copy/fill path.
            cycles = 3 + word_count * cycles_per_unit
            return StepResult(address, instruction, True, "SWI CpuFastSet" if number == 0x0C else "SWI CpuSet", cycles)

        raise UndefinedInstruction(address, instruction)

    def _execute_data_processing(self, instruction: int, address: int) -> StepResult:
        # Multiply and halfword-transfer instructions share this broad encoding
        # area with register-form data processing. Immediate-form instructions
        # reuse these low bits as literal data, so do not classify them here.
        if not (instruction & (1 << 25)) and instruction & 0x00000090 == 0x00000090:
            raise UnsupportedInstruction(address, instruction)

        operation = (instruction >> 21) & 0xF
        set_flags = bool(instruction & (1 << 20))
        rn = (instruction >> 16) & 0xF
        rd = (instruction >> 12) & 0xF
        if instruction & (1 << 25) and operation in (0x2, 0x4) and rn != 15 and rd != 15:
            immediate = instruction & 0xFF
            rotation = ((instruction >> 8) & 0xF) * 2
            operand = (
                ((immediate >> rotation) | (immediate << (32 - rotation))) & MASK32
                if rotation else immediate
            )
            left = self.registers[rn] & MASK32
            if operation == 0x2:
                result = (left - operand) & MASK32
                if set_flags:
                    carry = left >= operand
                    overflow = bool((left ^ operand) & (left ^ result) & 0x80000000)
            else:
                total = left + operand
                result = total & MASK32
                if set_flags:
                    carry = total > MASK32
                    overflow = bool(~(left ^ operand) & (left ^ result) & 0x80000000)
            self.registers[rd] = result
            if set_flags:
                self._set_arithmetic_flags(result, carry, overflow)
            return StepResult(address, instruction, True, "SUB" if operation == 0x2 else "ADD")

        left = self._read_register(rn, address)
        operand, shifter_carry = self._shift_operand(instruction, address)
        old_carry = self._flag(FLAG_C)

        # AND, EOR, SUB, RSB, ADD, ADC, SBC, RSC, TST, TEQ, CMP, CMN,
        # ORR, MOV, BIC, MVN.
        if operation == 0x0:
            result, carry, overflow, kind = left & operand, shifter_carry, self._flag(FLAG_V), "AND"
        elif operation == 0x1:
            result, carry, overflow, kind = left ^ operand, shifter_carry, self._flag(FLAG_V), "EOR"
        elif operation in (0x2, 0xA):
            result = (left - operand) & MASK32
            carry = left >= operand
            overflow = bool(((left ^ operand) & (left ^ result) & 0x80000000))
            kind = "CMP" if operation == 0xA else "SUB"
        elif operation == 0x3:
            result = (operand - left) & MASK32
            carry = operand >= left
            overflow = bool(((operand ^ left) & (operand ^ result) & 0x80000000))
            kind = "RSB"
        elif operation in (0x4, 0xB):
            total = left + operand
            result = total & MASK32
            carry = total > MASK32
            overflow = bool((~(left ^ operand) & (left ^ result) & 0x80000000))
            kind = "CMN" if operation == 0xB else "ADD"
        elif operation == 0x5:
            total = left + operand + int(old_carry)
            result = total & MASK32
            carry = total > MASK32
            overflow = bool((~(left ^ operand) & (left ^ result) & 0x80000000))
            kind = "ADC"
        elif operation == 0x6:
            borrow = 1 - int(old_carry)
            result = (left - operand - borrow) & MASK32
            carry = left >= operand + borrow
            overflow = bool(((left ^ operand) & (left ^ result) & 0x80000000))
            kind = "SBC"
        elif operation == 0x7:
            borrow = 1 - int(old_carry)
            result = (operand - left - borrow) & MASK32
            carry = operand >= left + borrow
            overflow = bool(((operand ^ left) & (operand ^ result) & 0x80000000))
            kind = "RSC"
        elif operation == 0x8:
            result, carry, overflow, kind = left & operand, shifter_carry, self._flag(FLAG_V), "TST"
        elif operation == 0x9:
            result, carry, overflow, kind = left ^ operand, shifter_carry, self._flag(FLAG_V), "TEQ"
        elif operation == 0xC:
            result, carry, overflow, kind = left | operand, shifter_carry, self._flag(FLAG_V), "ORR"
        elif operation == 0xD:
            result, carry, overflow, kind = operand, shifter_carry, self._flag(FLAG_V), "MOV"
        elif operation == 0xE:
            result, carry, overflow, kind = left & ~operand, shifter_carry, self._flag(FLAG_V), "BIC"
        else:  # 0xF
            result, carry, overflow, kind = ~operand, shifter_carry, self._flag(FLAG_V), "MVN"

        result &= MASK32
        test_only = operation in (0x8, 0x9, 0xA, 0xB)
        if test_only:
            set_flags = True
        current_mode = self.cpsr & MODE_MASK
        if rd == 15 and set_flags and not test_only and current_mode not in PRIVILEGED_MODES:
            raise UnsupportedInstruction(address, instruction)

        exception_return = rd == 15 and set_flags and not test_only
        if set_flags and not exception_return:
            if operation in (0x0, 0x1, 0x8, 0x9, 0xC, 0xD, 0xE, 0xF):
                self._set_nz(result)
                self.cpsr = (self.cpsr & ~FLAG_C) | (FLAG_C if carry else 0)
            else:
                self._set_arithmetic_flags(result, carry, overflow)

        if not test_only:
            if rd == 15:
                if exception_return:
                    self._restore_cpsr(self._get_spsr(current_mode), address, instruction)
                    alignment = 1 if self.thumb_state else 3
                    self.registers[15] = result & ~alignment
                else:
                    self.registers[15] = result & ~3
            else:
                self.registers[rd] = result
        return StepResult(address, instruction, True, kind)

    def _execute_arm(self, address: int, instruction: int) -> StepResult:
        condition = instruction >> 28
        if not self._condition_passes(condition):
            return StepResult(address, instruction, False, "condition skipped")

        if instruction & 0x0F000000 == 0x0F000000:
            if self.bus.bios_data is not None:
                return self._enter_swi(address, instruction, was_thumb=False)
            return self._execute_bios_call((instruction >> 16) & 0xFF, address, instruction)

        # These large ARM opcode groups are common in game code. Handle them
        # before the narrower status-register and multiply encodings below.
        if instruction & 0x0C000000 == 0x04000000:
            return self._execute_single_data_transfer(instruction, address)
        if instruction & 0x0E000000 == 0x0A000000:
            link = bool(instruction & (1 << 24))
            offset = instruction & 0x00FFFFFF
            if offset & 0x00800000:
                offset -= 1 << 24
            if link:
                self.registers[14] = (address + 4) & MASK32
            self.registers[15] = (address + 8 + (offset << 2)) & MASK32
            return StepResult(address, instruction, True, "BL" if link else "B")
        if instruction & 0x0E000000 == 0x08000000:
            return self._execute_block_transfer(instruction, address)
        if instruction & 0x0FB00FF0 == 0x01000090:
            return self._execute_swap(instruction, address)
        if instruction & 0x0F8000F0 == 0x00800090 or instruction & 0x0FC000F0 == 0x00000090:
            return self._execute_multiply(instruction, address)
        if instruction & 0x0E000090 == 0x00000090:
            return self._execute_halfword_transfer(instruction, address)

        # MRS reads CPSR/SPSR; MSR writes selected status-register bytes.
        # SPSR is available in privileged exception modes, not User/System.
        if instruction & 0x0FBF0FFF == 0x010F0000:
            saved = bool(instruction & (1 << 22))
            destination = (instruction >> 12) & 0xF
            current_mode = self.cpsr & MODE_MASK
            if destination == 15 or (saved and current_mode not in PRIVILEGED_MODES):
                raise UnsupportedInstruction(address, instruction)
            self.registers[destination] = self._get_spsr(current_mode) if saved else self.cpsr
            return StepResult(address, instruction, True, "MRS SPSR" if saved else "MRS CPSR")

        msr_register = instruction & 0x0FB0FFF0 == 0x0120F000
        msr_immediate = instruction & 0x0FB0F000 == 0x0320F000
        if msr_register or msr_immediate:
            saved = bool(instruction & (1 << 22))
            fields = (instruction >> 16) & 0xF
            if msr_immediate:
                immediate = instruction & 0xFF
                rotate = ((instruction >> 8) & 0xF) * 2
                value = ((immediate >> rotate) | (immediate << (32 - rotate))) & MASK32 if rotate else immediate
            else:
                value = self._read_register(instruction & 0xF, address)

            byte_mask = sum((0xFF << (byte * 8)) for byte in range(4) if fields & (1 << byte))
            if saved:
                current_mode = self.cpsr & MODE_MASK
                if current_mode not in PRIVILEGED_MODES:
                    raise UnsupportedInstruction(address, instruction)
                old_spsr = self._get_spsr(current_mode)
                self._set_spsr((old_spsr & ~byte_mask) | (value & byte_mask), current_mode)
            else:
                if fields & 0x8:
                    self.cpsr = (self.cpsr & 0x0FFFFFFF) | (value & 0xF0000000)
                if fields & 0x1 and (self.cpsr & MODE_MASK) != MODE_USER:
                    mode = value & MODE_MASK
                    if mode not in VALID_MODES:
                        raise UnsupportedInstruction(address, instruction)
                    self._set_mode(mode)
                    self.cpsr = (self.cpsr & ~0xFF) | (value & 0xFF)
            return StepResult(address, instruction, True, "MSR SPSR" if saved else "MSR CPSR")

        # BX Rm: exchange between ARM and Thumb instruction sets.
        if instruction & 0x0FFFFFF0 == 0x012FFF10:
            source = instruction & 0xF
            target = self.registers[source] & MASK32
            self.cpsr = (self.cpsr | FLAG_T) if target & 1 else (self.cpsr & ~FLAG_T)
            alignment = 1 if self.thumb_state else 3
            self.registers[15] = target & ~alignment & MASK32
            return StepResult(address, instruction, True, f"BX r{source}")

        # B and BL use a signed, PC-relative 24-bit word offset.
        if instruction & 0x0E000000 == 0x0A000000:
            link = bool(instruction & (1 << 24))
            offset = instruction & 0x00FFFFFF
            if offset & 0x00800000:
                offset -= 1 << 24
            target = (address + 8 + (offset << 2)) & MASK32
            if link:
                self.registers[14] = (address + 4) & MASK32
            self.registers[15] = target
            return StepResult(address, instruction, True, "BL" if link else "B")

        # Swap a register with one memory location.
        if instruction & 0x0FB00FF0 == 0x01000090:
            return self._execute_swap(instruction, address)

        # 32x32 multiply instructions use a separate ARM encoding group.
        if instruction & 0x0F8000F0 == 0x00800090 or instruction & 0x0FC000F0 == 0x00000090:
            return self._execute_multiply(instruction, address)

        # Halfword and signed byte/halfword transfers.
        if instruction & 0x0E000090 == 0x00000090:
            return self._execute_halfword_transfer(instruction, address)

        # Single data transfer: LDR/STR and their byte forms.
        if instruction & 0x0C000000 == 0x04000000:
            return self._execute_single_data_transfer(instruction, address)

        # Load/store a selected register list in ascending register order.
        if instruction & 0x0E000000 == 0x08000000:
            return self._execute_block_transfer(instruction, address)

        # Data-processing operations (logical and arithmetic instructions).
        if instruction & 0x0C000000 == 0:
            return self._execute_data_processing(instruction, address)

        raise UndefinedInstruction(address, instruction)

    def _thumb_register(self, index: int, address: int) -> int:
        if index == 15:
            return (address + 4) & ~3
        return self.registers[index] & MASK32

    def _thumb_write_register(self, index: int, value: int) -> None:
        value &= MASK32
        self.registers[index] = (value & ~1) if index == 15 else value

    def _thumb_load_word(self, address: int) -> int:
        """Read a word using ARM7TDMI's align-down-and-rotate behavior."""
        value = self.bus.read32(address)
        rotation = (address & 3) * 8
        if rotation:
            value = ((value >> rotation) | (value << (32 - rotation))) & MASK32
        return value

    def _load_arm7_halfword(self, address: int, *, signed: bool = False) -> int:
        """Read an ARM7 halfword, including its odd-address edge cases."""
        if address & 1 and signed:
            value = self.bus.read8(address)
            return (value - 0x100 if value & 0x80 else value) & MASK32
        value = self.bus.read16(address)
        if address & 1:
            value = ((value >> 8) | (value << 24)) & MASK32
        elif signed and value & 0x8000:
            value -= 0x10000
        return value & MASK32

    def _thumb_add(self, left: int, right: int, carry_in: int = 0) -> int:
        total = (left & MASK32) + (right & MASK32) + carry_in
        result = total & MASK32
        carry = total > MASK32
        overflow = bool((~(left ^ right) & (left ^ result) & 0x80000000))
        self._set_arithmetic_flags(result, carry, overflow)
        return result

    def _thumb_sub(self, left: int, right: int, borrow_in: int = 0) -> int:
        left &= MASK32
        right &= MASK32
        result = (left - right - borrow_in) & MASK32
        carry = left >= right + borrow_in
        overflow = bool(((left ^ right) & (left ^ result) & 0x80000000))
        self._set_arithmetic_flags(result, carry, overflow)
        return result

    def _execute_thumb(self, address: int, instruction: int) -> StepResult:
        # Hot Thumb encodings used heavily in game startup and polling loops
        # are dispatched first so they do not scan the full decoder chain.
        if instruction & 0xE000 == 0x2000:
            operation = (instruction >> 11) & 0x3
            destination = (instruction >> 8) & 0x7
            immediate = instruction & 0xFF
            old = self.registers[destination] & MASK32
            if operation == 0:
                self.registers[destination] = immediate
                self._set_nz(immediate)
                name = "MOV"
            elif operation == 1:
                self._thumb_sub(old, immediate)
                name = "CMP"
            elif operation == 2:
                self.registers[destination] = self._thumb_add(old, immediate)
                name = "ADD"
            else:
                self.registers[destination] = self._thumb_sub(old, immediate)
                name = "SUB"
            return StepResult(address, instruction, True, name)

        if instruction & 0xE000 == 0x6000:
            byte_access = bool(instruction & (1 << 12))
            load = bool(instruction & (1 << 11))
            offset = (instruction >> 6) & 0x1F
            if not byte_access:
                offset <<= 2
            base = (instruction >> 3) & 0x7
            destination = instruction & 0x7
            location = (self.registers[base] + offset) & MASK32
            if load:
                self.registers[destination] = (
                    self.bus.read8(location) if byte_access else self._thumb_load_word(location)
                )
            elif byte_access:
                self.bus.write8(location, self.registers[destination])
            else:
                self.bus.write32(location, self.registers[destination])
            operation = (
                "LDRB" if load and byte_access else
                "STRB" if byte_access else
                "LDR" if load else "STR"
            )
            return StepResult(address, instruction, True, operation)

        if instruction & 0xF000 == 0xD000:
            condition = (instruction >> 8) & 0xF
            if condition == 0xF:
                if self.bus.bios_data is not None:
                    return self._enter_swi(address, instruction, was_thumb=True)
                return self._execute_bios_call(instruction & 0xFF, address, instruction)
            if condition == 0xE:
                raise UndefinedInstruction(address, instruction)
            taken = self._condition_passes(condition)
            if taken:
                offset = instruction & 0xFF
                if offset & 0x80:
                    offset -= 0x100
                self.registers[15] = (address + 4 + (offset << 1)) & MASK32
            return StepResult(address, instruction, True, "B conditional taken" if taken else "B conditional not taken")

        # Shift by an immediate: LSL, LSR, and ASR.
        if instruction & 0xF800 in (0x0000, 0x0800, 0x1000):
            shift_type = (instruction >> 11) & 0x3
            amount = (instruction >> 6) & 0x1F
            source = (instruction >> 3) & 0x7
            destination = instruction & 0x7
            value = self.registers[source] & MASK32
            old_carry = self._flag(FLAG_C)
            if shift_type == 0:
                if amount == 0:
                    result, carry = value, old_carry
                else:
                    result = (value << amount) & MASK32
                    carry = bool(value & (1 << (32 - amount)))
                operation = "LSL"
            elif shift_type == 1:
                amount = amount or 32
                if amount == 32:
                    result, carry = 0, bool(value & 0x80000000)
                else:
                    result, carry = value >> amount, bool(value & (1 << (amount - 1)))
                operation = "LSR"
            elif shift_type == 2:
                amount = amount or 32
                sign = bool(value & 0x80000000)
                if amount == 32:
                    result, carry = (MASK32 if sign else 0), sign
                else:
                    signed = value if not sign else value - (1 << 32)
                    result = (signed >> amount) & MASK32
                    carry = bool(value & (1 << (amount - 1)))
                operation = "ASR"
            else:
                raise UnsupportedInstruction(address, instruction)
            self.registers[destination] = result
            self._set_nz(result)
            self.cpsr = (self.cpsr & ~FLAG_C) | (FLAG_C if carry else 0)
            return StepResult(address, instruction, True, operation)

        # Add/subtract a register or a small immediate.
        if instruction & 0xF800 == 0x1800:
            immediate = bool(instruction & (1 << 10))
            subtract = bool(instruction & (1 << 9))
            operand_field = (instruction >> 6) & 0x7
            source = (instruction >> 3) & 0x7
            destination = instruction & 0x7
            right = operand_field if immediate else self.registers[operand_field]
            left = self.registers[source]
            result = self._thumb_sub(left, right) if subtract else self._thumb_add(left, right)
            self.registers[destination] = result
            return StepResult(address, instruction, True, "SUB" if subtract else "ADD")

        # Register ALU operations: logical, arithmetic, shifts, multiply.
        if instruction & 0xFC00 == 0x4000:
            operation = (instruction >> 6) & 0xF
            source = (instruction >> 3) & 0x7
            destination = instruction & 0x7
            left = self.registers[destination] & MASK32
            right = self.registers[source] & MASK32
            name = (
                "AND", "EOR", "LSL", "LSR", "ASR", "ADC", "SBC", "ROR",
                "TST", "NEG", "CMP", "CMN", "ORR", "MUL", "BIC", "MVN",
            )[operation]
            write_result = True

            if operation == 0:
                result = left & right
                self._set_nz(result)
            elif operation == 1:
                result = left ^ right
                self._set_nz(result)
            elif operation in (2, 3, 4, 7):
                amount = right & 0xFF
                old_carry = self._flag(FLAG_C)
                if amount == 0:
                    result, carry = left, old_carry
                elif operation == 2:
                    if amount < 32:
                        result, carry = (left << amount) & MASK32, bool(left & (1 << (32 - amount)))
                    elif amount == 32:
                        result, carry = 0, bool(left & 1)
                    else:
                        result, carry = 0, False
                elif operation == 3:
                    if amount < 32:
                        result, carry = left >> amount, bool(left & (1 << (amount - 1)))
                    elif amount == 32:
                        result, carry = 0, bool(left & 0x80000000)
                    else:
                        result, carry = 0, False
                elif operation == 4:
                    sign = bool(left & 0x80000000)
                    signed = left if not sign else left - (1 << 32)
                    if amount >= 32:
                        result, carry = (MASK32 if sign else 0), sign
                    else:
                        result = (signed >> amount) & MASK32
                        carry = bool(left & (1 << (amount - 1)))
                else:
                    rotation = amount & 0x1F
                    if rotation == 0:
                        result, carry = left, bool(left & 0x80000000)
                    else:
                        result = ((left >> rotation) | (left << (32 - rotation))) & MASK32
                        carry = bool(result & 0x80000000)
                self._set_nz(result)
                if amount:
                    self.cpsr = (self.cpsr & ~FLAG_C) | (FLAG_C if carry else 0)
            elif operation == 5:
                result = self._thumb_add(left, right, int(self._flag(FLAG_C)))
            elif operation == 6:
                result = self._thumb_sub(left, right, 1 - int(self._flag(FLAG_C)))
            elif operation == 8:
                result = left & right
                self._set_nz(result)
                write_result = False
            elif operation == 9:
                result = self._thumb_sub(0, right)
            elif operation == 10:
                result = self._thumb_sub(left, right)
                write_result = False
            elif operation == 11:
                result = self._thumb_add(left, right)
                write_result = False
            elif operation == 12:
                result = left | right
                self._set_nz(result)
            elif operation == 13:
                result = (left * right) & MASK32
                self._set_nz(result)
            elif operation == 14:
                result = left & ~right
                self._set_nz(result)
            else:
                result = ~right & MASK32
                self._set_nz(result)

            if write_result:
                self.registers[destination] = result & MASK32
            cycles = (
                1 + self._multiply_internal_cycles(right, signed=True)
                if operation == 13
                else None
            )
            return StepResult(address, instruction, True, name, cycles)

        # High registers and BX.
        if instruction & 0xFC00 == 0x4400:
            operation = (instruction >> 8) & 0x3
            source = ((instruction >> 3) & 0xF)
            destination = (instruction & 0x7) | ((instruction >> 4) & 0x8)
            right = self._thumb_register(source, address)
            if operation == 3:
                self.cpsr = (self.cpsr | FLAG_T) if right & 1 else (self.cpsr & ~FLAG_T)
                self.registers[15] = right & (~1 if self.thumb_state else ~3)
                return StepResult(address, instruction, True, f"BX r{source}")
            left = self._thumb_register(destination, address)
            if operation == 0:
                self._thumb_write_register(destination, left + right)
                name = "ADD"
            elif operation == 1:
                self._thumb_sub(left, right)
                name = "CMP"
            else:
                self._thumb_write_register(destination, right)
                name = "MOV"
            return StepResult(address, instruction, True, name)

        # PC-relative word load.
        if instruction & 0xF800 == 0x4800:
            destination = (instruction >> 8) & 0x7
            literal_address = ((address + 4) & ~3) + ((instruction & 0xFF) << 2)
            self.registers[destination] = self.bus.read32(literal_address)
            return StepResult(address, instruction, True, "LDR literal")

        # Register-offset word/halfword/byte and signed-byte/halfword transfers.
        if instruction & 0xF000 == 0x5000:
            operation = (instruction >> 9) & 0x7
            offset_register = (instruction >> 6) & 0x7
            base_register = (instruction >> 3) & 0x7
            destination = instruction & 0x7
            location = (self.registers[base_register] + self.registers[offset_register]) & MASK32
            if operation in (0, 1, 2):  # STR, STRH, STRB
                value = self.registers[destination]
                if operation == 0:
                    self.bus.write32(location, value)
                elif operation == 1:
                    self.bus.write16(location, value)
                else:
                    self.bus.write8(location, value)
            elif operation == 3:  # LDSB
                value = self.bus.read8(location)
                self.registers[destination] = (value - 0x100 if value & 0x80 else value) & MASK32
            elif operation == 4:
                self.registers[destination] = self._thumb_load_word(location)
            elif operation == 5:
                self.registers[destination] = self._load_arm7_halfword(location)
            elif operation == 7:  # LDSH; odd addresses behave like LDSB
                self.registers[destination] = self._load_arm7_halfword(location, signed=True)
            else:
                self.registers[destination] = self.bus.read8(location)
            return StepResult(address, instruction, True, ("STR" if operation < 3 else "LDR") + " register-offset")

        # Immediate-offset halfword transfers.
        if instruction & 0xF000 == 0x8000:
            load = bool(instruction & (1 << 11))
            offset = ((instruction >> 6) & 0x1F) << 1
            base = (instruction >> 3) & 0x7
            destination = instruction & 0x7
            location = (self.registers[base] + offset) & MASK32
            if load:
                self.registers[destination] = self._load_arm7_halfword(location)
            else:
                self.bus.write16(location, self.registers[destination])
            return StepResult(address, instruction, True, "LDRH" if load else "STRH")

        # SP-relative word transfers.
        if instruction & 0xF000 == 0x9000:
            load = bool(instruction & (1 << 11))
            destination = (instruction >> 8) & 0x7
            location = (self.registers[13] + ((instruction & 0xFF) << 2)) & MASK32
            if load:
                self.registers[destination] = self.bus.read32(location)
            else:
                self.bus.write32(location, self.registers[destination])
            return StepResult(address, instruction, True, "LDR SP-relative" if load else "STR SP-relative")

        # Load an address relative to PC or SP.
        if instruction & 0xF000 == 0xA000:
            use_sp = bool(instruction & (1 << 11))
            destination = (instruction >> 8) & 0x7
            base = self.registers[13] if use_sp else ((address + 4) & ~3)
            self.registers[destination] = (base + ((instruction & 0xFF) << 2)) & MASK32
            return StepResult(address, instruction, True, "ADD address")

        # Add or subtract a small, word-scaled value from SP.
        if instruction & 0xFF00 == 0xB000:
            amount = (instruction & 0x7F) << 2
            if instruction & (1 << 7):
                self.registers[13] = (self.registers[13] - amount) & MASK32
                name = "SUB SP"
            else:
                self.registers[13] = (self.registers[13] + amount) & MASK32
                name = "ADD SP"
            return StepResult(address, instruction, True, name)

        # PUSH and POP save/restore a list of low registers and optionally LR/PC.
        if instruction & 0xFE00 in (0xB400, 0xBC00):
            pop = bool(instruction & (1 << 11))
            extra = bool(instruction & (1 << 8))
            selected = [reg for reg in range(8) if instruction & (1 << reg)]
            if pop:
                location = self.registers[13]
                for reg in selected:
                    self.registers[reg] = self.bus.read32(location)
                    location += 4
                if extra:
                    self.registers[15] = self.bus.read32(location) & ~1
                    location += 4
                self.registers[13] = location & MASK32
                name = "POP"
            else:
                count = len(selected) + int(extra)
                location = (self.registers[13] - count * 4) & MASK32
                cursor = location
                for reg in selected:
                    self.bus.write32(cursor, self.registers[reg])
                    cursor += 4
                if extra:
                    self.bus.write32(cursor, self.registers[14])
                self.registers[13] = location
                name = "PUSH"
            return StepResult(address, instruction, True, name)

        # Multiple load/store of a low-register list.
        if instruction & 0xF000 == 0xC000:
            load = bool(instruction & (1 << 11))
            base_register = (instruction >> 8) & 0x7
            location = self.registers[base_register] & MASK32
            selected = [reg for reg in range(8) if instruction & (1 << reg)]
            base = location
            empty_register_list = not selected
            if empty_register_list:
                # ARMv4T treats an empty block-transfer list as a PC transfer,
                # while advancing the base as if all sixteen registers moved.
                if load:
                    self.registers[15] = self.bus.read32(location) & ~1
                else:
                    self.bus.write32(location, (address + 4) & ~3)
                location = (base + 0x40) & MASK32
            else:
                updated_base = (base + len(selected) * 4) & MASK32
                for reg in selected:
                    if load:
                        self.registers[reg] = self.bus.read32(location)
                    else:
                        value = self.registers[reg]
                        if reg == base_register and selected[0] != base_register:
                            # On ARMv4T, STM stores the updated base when its
                            # base register appears after the first list entry.
                            value = updated_base
                        self.bus.write32(location, value)
                    location = (location + 4) & MASK32
                # Thumb LDM with its base in the list keeps the loaded value;
                # it does not replace it with the incremented address.
                if not (load and base_register in selected):
                    self.registers[base_register] = updated_base
            return StepResult(address, instruction, True, "LDMIA (Thumb)" if load else "STMIA (Thumb)")

        # Unconditional branch.
        if instruction & 0xF800 == 0xE000:
            offset = instruction & 0x7FF
            if offset & 0x400:
                offset -= 0x800
            self.registers[15] = (address + 4 + (offset << 1)) & MASK32
            return StepResult(address, instruction, True, "B")

        # Thumb long branch with link (two halfword instructions).
        if instruction & 0xF800 == 0xF000:
            high = instruction & 0x7FF
            if high & 0x400:
                high -= 0x800
            self.registers[14] = (address + 4 + (high << 12)) & MASK32
            return StepResult(address, instruction, True, "BL prefix")
        if instruction & 0xF800 == 0xF800:
            target = (self.registers[14] + ((instruction & 0x7FF) << 1)) & MASK32
            self.registers[14] = ((address + 2) | 1) & MASK32
            self.registers[15] = target & ~1
            return StepResult(address, instruction, True, "BL suffix")

        raise UndefinedInstruction(address, instruction)

    def _step_thumb(self) -> StepResult:
        address = self.registers[15] & MASK32
        instruction = self.bus.read16(address)
        if self.bus.bios_data is not None:
            self.bus.record_instruction_fetch(address, instruction, thumb=True)
        # Most Thumb instructions only change registers and flags. Starting
        # bus tracking for them used two Python calls per instruction despite
        # there being no data access to measure.
        tracks_data = (
            instruction & 0xE000 == 0x6000
            or instruction & 0xF000 in (0x5000, 0x8000, 0x9000, 0xC000)
            or instruction & 0xFE00 in (0xB400, 0xBC00)
            or instruction & 0xF000 == 0xD000 and (instruction >> 8) & 0xF == 0xF
        )
        if tracks_data:
            self.bus.begin_instruction_access_tracking()
        self.registers[15] = (address + 2) & MASK32
        try:
            return self._execute_thumb(address, instruction)
        except UnsupportedInstruction:
            self.bus.cancel_instruction_access_tracking()
            self.registers[15] = address
            raise

    def step(self) -> StepResult:
        """Service a pending IRQ, then fetch and execute one instruction."""
        low_power_request = self.bus._low_power_request
        if low_power_request is not None:
            self.bus._low_power_request = None
            if low_power_request == "stop":
                self._stopped = True
            elif low_power_request == "halt":
                self._halted = True
        if self._stopped:
            # STOP wakes from an enabled keypad, cartridge, or serial IRQ.
            # Only keypad wake is modeled here; IME still controls whether the
            # CPU vectors to its handler after the sleep state ends.
            keypad_request = (
                self.bus.read16(0x04000200)
                & self.bus.read16(0x04000202)
                & (1 << 12)
            )
            if not keypad_request:
                return StepResult(self.registers[15] & MASK32, 0, True, "STOP idle")
            self._stopped = False
        if self._halted:
            # GBA Halt wakes on an interrupt enabled in IE and requested in IF;
            # IME and the CPSR IRQ mask determine whether the CPU takes it.
            pending = self.bus.read16(0x04000200) & self.bus.read16(0x04000202)
            if not pending:
                return StepResult(self.registers[15] & MASK32, 0, True, "HALT idle")
            self._halted = False
        irq_result = None
        if not (self.cpsr & FLAG_I) and self.bus.io[0x208] & 1:
            pending = (
                (self.bus.io[0x200] | (self.bus.io[0x201] << 8))
                & (self.bus.io[0x202] | (self.bus.io[0x203] << 8))
            )
            if pending:
                irq_result = self._enter_irq(pending)
        if irq_result is not None:
            self._last_gamepak_fetch = None
            return irq_result
        if self._intr_wait_mask:
            flags = self.bus.read32(0x03007FF8)
            if flags & self._intr_wait_mask:
                self.bus.write32(0x03007FF8, flags & ~self._intr_wait_mask)
                self._intr_wait_mask = 0
            else:
                return StepResult(self.registers[15] & MASK32, 0, True, "IntrWait idle")
        if self.bus.bios_data is not None:
            self.bus.set_cpu_pc(self.registers[15])
        if self.cpsr & FLAG_T:
            address = self.registers[15] & MASK32
            try:
                result = self._step_thumb()
                return self._finish_instruction_fetch(result, address, 2, True)
            except UndefinedInstruction as exc:
                if self.bus.bios_data is None:
                    raise
                result = self._enter_undefined(exc.address, exc.instruction, was_thumb=True)
                return self._finish_instruction_fetch(result, address, 2, True)

        address = self.registers[15] & MASK32
        instruction = self.bus.read32(address)
        if self.bus.bios_data is not None:
            self.bus.record_instruction_fetch(address, instruction, thumb=False)
        tracks_data = (
            instruction & 0x0C000000 == 0x04000000
            or instruction & 0x0E000000 == 0x08000000
            or instruction & 0x0FB00FF0 == 0x01000090
            or instruction & 0x0E000090 == 0x00000090
            or instruction & 0x0F000000 == 0x0F000000
        )
        if tracks_data:
            self.bus.begin_instruction_access_tracking()
        self.registers[15] = (address + 4) & MASK32
        try:
            result = self._execute_arm(address, instruction)
            return self._finish_instruction_fetch(result, address, 4, False)
        except UndefinedInstruction as exc:
            self.registers[15] = address
            if self.bus.bios_data is None:
                self.bus.cancel_instruction_access_tracking()
                raise
            result = self._enter_undefined(exc.address, exc.instruction, was_thumb=False)
            return self._finish_instruction_fetch(result, address, 4, False)
        except UnsupportedInstruction:
            # Leave the CPU at the instruction it cannot execute, so a later
            # instruction implementation can resume cleanly at the same spot.
            self.bus.cancel_instruction_access_tracking()
            self.registers[15] = address
            raise

    def run_steps(self, count: int) -> list[StepResult]:
        """Execute up to ``count`` instructions, stopping on unsupported code."""
        if count < 0:
            raise ValueError("instruction count must be non-negative")
        return [self.step() for _ in range(count)]
