"""Small instruction-level checks for the early ARM/Thumb CPU core."""

import unittest

from gba.arm7tdmi import Arm7Tdmi, FLAG_T
from gba.cartridge import Cartridge
from gba.memory import MemoryBus


def cartridge_with_code(code: bytes) -> Cartridge:
    rom = bytearray(0xC0 + len(code))
    rom[: len(code)] = code
    return Cartridge(bytes(rom), "TEST", "TST0", "00", 0, 0, 0, 0, 0)


class CpuSmokeTests(unittest.TestCase):
    def test_arm_math_and_store(self) -> None:
        # MOV r1,#5; ADD r0,r1,#7; STR r0,[r2]
        code = b"".join(word.to_bytes(4, "little") for word in (0xE3A01005, 0xE2810007, 0xE5820000))
        bus = MemoryBus(cartridge_with_code(code))
        cpu = Arm7Tdmi(bus)
        cpu.registers[2] = 0x02000000

        cpu.run_steps(3)

        self.assertEqual(bus.read32(0x02000000), 12)

    def test_thumb_math_and_store(self) -> None:
        # MOV r0,#5; ADD r0,#7; STR r0,[r1,#0]
        code = b"".join(half.to_bytes(2, "little") for half in (0x2005, 0x3007, 0x6008))
        bus = MemoryBus(cartridge_with_code(code))
        cpu = Arm7Tdmi(bus)
        cpu.cpsr |= FLAG_T
        cpu.registers[1] = 0x02000000

        cpu.run_steps(3)

        self.assertEqual(bus.read32(0x02000000), 12)

    def test_arm_condition_can_skip_instruction(self) -> None:
        # MOVEQ r0,#1 is skipped because reset Z is clear.
        code = (0x03A00001).to_bytes(4, "little")
        cpu = Arm7Tdmi(MemoryBus(cartridge_with_code(code)))

        result = cpu.step()

        self.assertFalse(result.executed)
        self.assertEqual(cpu.registers[0], 0)
        self.assertEqual(cpu.registers[15], 0x08000004)

    def test_arm_multiply(self) -> None:
        # MUL r0,r1,r2
        code = (0xE0000291).to_bytes(4, "little")
        cpu = Arm7Tdmi(MemoryBus(cartridge_with_code(code)))
        cpu.registers[1] = 6
        cpu.registers[2] = 7

        cpu.step()

        self.assertEqual(cpu.registers[0], 42)

    def test_arm_unsigned_long_multiply(self) -> None:
        # UMULL r0,r1,r2,r3: a 64-bit product in r1:r0.
        code = (0xE0810392).to_bytes(4, "little")
        cpu = Arm7Tdmi(MemoryBus(cartridge_with_code(code)))
        cpu.registers[2] = 0xFFFFFFFF
        cpu.registers[3] = 2

        cpu.step()

        self.assertEqual(cpu.registers[0], 0xFFFFFFFE)
        self.assertEqual(cpu.registers[1], 1)

    def test_arm_block_store_and_load(self) -> None:
        # STMIA r4!,{r0-r2}; LDMIA r5!,{r0-r2}
        words = (0xE8A40007, 0xE8B50007)
        code = b"".join(word.to_bytes(4, "little") for word in words)
        bus = MemoryBus(cartridge_with_code(code))
        cpu = Arm7Tdmi(bus)
        cpu.registers[0:3] = [11, 22, 33]
        cpu.registers[4] = 0x02000000
        cpu.registers[5] = 0x02000000

        cpu.step()
        self.assertEqual([bus.read32(0x02000000 + 4 * i) for i in range(3)], [11, 22, 33])
        cpu.registers[0:3] = [0, 0, 0]
        cpu.registers[15] = 0x08000004
        cpu.step()

        self.assertEqual(cpu.registers[0:3], [11, 22, 33])
        self.assertEqual(cpu.registers[4], 0x0200000C)
        self.assertEqual(cpu.registers[5], 0x0200000C)

    def test_arm_halfword_and_signed_byte_loads(self) -> None:
        # LDRH r0,[r1]; LDRSB r2,[r1]
        words = (0xE1D100B0, 0xE1D120D0)
        code = b"".join(word.to_bytes(4, "little") for word in words)
        bus = MemoryBus(cartridge_with_code(code))
        cpu = Arm7Tdmi(bus)
        cpu.registers[1] = 0x02000000
        bus.write16(0x02000000, 0xAB80)

        cpu.step()
        cpu.step()

        self.assertEqual(cpu.registers[0], 0xAB80)
        self.assertEqual(cpu.registers[2], 0xFFFFFF80)

    def test_arm_register_controlled_shift(self) -> None:
        # MOVS r2,r0,LSL r1
        code = (0xE1B02110).to_bytes(4, "little")
        cpu = Arm7Tdmi(MemoryBus(cartridge_with_code(code)))
        cpu.registers[0] = 3
        cpu.registers[1] = 2

        cpu.step()

        self.assertEqual(cpu.registers[2], 12)

    def test_arm_swap(self) -> None:
        # SWP r0,r1,[r2]
        code = (0xE1020091).to_bytes(4, "little")
        bus = MemoryBus(cartridge_with_code(code))
        cpu = Arm7Tdmi(bus)
        cpu.registers[1] = 22
        cpu.registers[2] = 0x02000000
        bus.write32(0x02000000, 11)

        cpu.step()

        self.assertEqual(cpu.registers[0], 11)
        self.assertEqual(bus.read32(0x02000000), 22)

    def test_arm_bios_division_and_square_root(self) -> None:
        instructions = (0xEF060000, 0xEF080000)  # SWI Div; SWI Sqrt
        code = b"".join(word.to_bytes(4, "little") for word in instructions)
        cpu = Arm7Tdmi(MemoryBus(cartridge_with_code(code)))
        cpu.registers[0] = (-1234) & 0xFFFFFFFF
        cpu.registers[1] = 10

        cpu.step()
        self.assertEqual(cpu.registers[0], (-123) & 0xFFFFFFFF)
        self.assertEqual(cpu.registers[1], (-4) & 0xFFFFFFFF)
        self.assertEqual(cpu.registers[3], 123)

        cpu.registers[0] = 1000
        cpu.step()
        self.assertEqual(cpu.registers[0], 31)

    def test_thumb_bios_divarm_uses_reversed_arguments(self) -> None:
        code = (0xDF07).to_bytes(2, "little")  # SWI DivArm
        cpu = Arm7Tdmi(MemoryBus(cartridge_with_code(code)))
        cpu.cpsr |= FLAG_T
        cpu.registers[0] = 7  # Divisor
        cpu.registers[1] = 100  # Numerator

        result = cpu.step()

        self.assertEqual(result.operation, "SWI DivArm")
        self.assertEqual(cpu.registers[0], 14)
        self.assertEqual(cpu.registers[1], 2)
        self.assertEqual(cpu.registers[3], 14)

    def test_cpu_set_copies_words_and_fast_set_rounds_to_eight_words(self) -> None:
        code = b"".join(word.to_bytes(4, "little") for word in (0xEF0B0000, 0xEF0C0000))
        bus = MemoryBus(cartridge_with_code(code))
        cpu = Arm7Tdmi(bus)
        source = 0x02000000
        destination = 0x02001000
        for index in range(16):
            bus.write32(source + index * 4, 0xA5000000 + index)

        cpu.registers[0:3] = [source, destination, (1 << 26) | 3]
        cpu.step()
        self.assertEqual([bus.read32(destination + i * 4) for i in range(3)], [0xA5000000, 0xA5000001, 0xA5000002])

        cpu.registers[0:3] = [source, destination + 0x100, 1]  # FastSet count rounds up to 8 words.
        cpu.registers[15] = 0x08000004
        cpu.step()
        self.assertEqual(
            [bus.read32(destination + 0x100 + i * 4) for i in range(8)],
            [0xA5000000 + i for i in range(8)],
        )

    def test_arm_bios_lz77_decompresses_literals_and_back_references(self) -> None:
        code = (0xEF110000).to_bytes(4, "little")  # SWI LZ77UnCompWram
        bus = MemoryBus(cartridge_with_code(code))
        source = 0x02000000
        destination = 0x02001000
        encoded = bytes((0x10, 6, 0, 0, 0x10, ord("A"), ord("B"), ord("C"), 0, 2))
        for offset, value in enumerate(encoded):
            bus.write8(source + offset, value)
        cpu = Arm7Tdmi(bus)
        cpu.registers[0:2] = [source, destination]

        result = cpu.step()

        self.assertEqual(result.operation, "SWI LZ77UnCompWram")
        self.assertEqual(bytes(bus.read8(destination + i) for i in range(6)), b"ABCABC")

    def test_thumb_bios_rl_decompresses_repeated_and_literal_runs(self) -> None:
        code = (0xDF14).to_bytes(2, "little")  # SWI RLUnCompWram
        bus = MemoryBus(cartridge_with_code(code))
        source = 0x02000000
        destination = 0x02001000
        encoded = bytes((0x30, 6, 0, 0, 0x80, 0xAA, 0x02, 1, 2, 3))
        for offset, value in enumerate(encoded):
            bus.write8(source + offset, value)
        cpu = Arm7Tdmi(bus)
        cpu.cpsr |= FLAG_T
        cpu.registers[0:2] = [source, destination]

        result = cpu.step()

        self.assertEqual(result.operation, "SWI RLUnCompWram")
        self.assertEqual(bytes(bus.read8(destination + i) for i in range(6)), bytes((0xAA, 0xAA, 0xAA, 1, 2, 3)))

    def test_lz77_vram_writes_halfwords_and_pads_odd_output(self) -> None:
        code = (0xEF120000).to_bytes(4, "little")  # SWI LZ77UnCompVram
        bus = MemoryBus(cartridge_with_code(code))
        source = 0x02000000
        destination = 0x06000000
        encoded = bytes((0x10, 3, 0, 0, 0, ord("G"), ord("B"), ord("A")))
        for offset, value in enumerate(encoded):
            bus.write8(source + offset, value)
        cpu = Arm7Tdmi(bus)
        cpu.registers[0:2] = [source, destination]

        result = cpu.step()

        self.assertEqual(result.operation, "SWI LZ77UnCompVram")
        self.assertEqual(bus.read16(destination), ord("G") | (ord("B") << 8))
        self.assertEqual(bus.read16(destination + 2), ord("A"))

    def test_bit_unpack_expands_small_values_and_applies_palette_offset(self) -> None:
        code = (0xEF100000).to_bytes(4, "little")  # SWI BitUnPack
        bus = MemoryBus(cartridge_with_code(code))
        source = 0x02000000
        destination = 0x02001000
        info = 0x02000010
        bus.write8(source, 0b11100100)  # 2-bit values 0, 1, 2, 3, low bits first.
        bus.write16(info, 1)  # One source byte.
        bus.write8(info + 2, 2)
        bus.write8(info + 3, 8)
        bus.write32(info + 4, 0x10)  # Add 0x10 to nonzero colors only.
        cpu = Arm7Tdmi(bus)
        cpu.registers[0:3] = [source, destination, info]

        result = cpu.step()

        self.assertEqual(result.operation, "SWI BitUnPack")
        self.assertEqual(bus.read32(destination), 0x13121100)

    def test_difference_filter_restores_incrementing_bytes(self) -> None:
        code = (0xEF160000).to_bytes(4, "little")  # SWI Diff8bitUnFilterWram
        bus = MemoryBus(cartridge_with_code(code))
        source = 0x02000000
        destination = 0x02001000
        encoded = bytes((0x81, 4, 0, 0, 10, 1, 1, 1))  # Type 8, 4 output bytes.
        for offset, value in enumerate(encoded):
            bus.write8(source + offset, value)
        cpu = Arm7Tdmi(bus)
        cpu.registers[0:2] = [source, destination]

        result = cpu.step()

        self.assertEqual(result.operation, "SWI Diff8bitUnFilterWram")
        self.assertEqual(bytes(bus.read8(destination + i) for i in range(4)), bytes((10, 11, 12, 13)))

    def test_huffman_bios_decompresses_8bit_symbols(self) -> None:
        code = (0xEF130000).to_bytes(4, "little")  # SWI HuffUnComp
        bus = MemoryBus(cartridge_with_code(code))
        source = 0x02000000
        destination = 0x02001000
        header = (4 << 8) | (2 << 4) | 8
        encoded = header.to_bytes(4, "little") + bytes((1, 0xC0, ord("A"), ord("B"))) + (0x30000000).to_bytes(4, "little")
        for offset, value in enumerate(encoded):
            bus.write8(source + offset, value)
        cpu = Arm7Tdmi(bus)
        cpu.registers[0:2] = [source, destination]

        result = cpu.step()

        self.assertEqual(result.operation, "SWI HuffUnComp")
        self.assertEqual(bytes(bus.read8(destination + i) for i in range(4)), b"AABB")

    def test_huffman_bios_packs_4bit_symbols_into_nibbles(self) -> None:
        code = (0xEF130000).to_bytes(4, "little")  # SWI HuffUnComp
        bus = MemoryBus(cartridge_with_code(code))
        source = 0x02000000
        destination = 0x02001000
        header = (4 << 8) | (2 << 4) | 4
        encoded = header.to_bytes(4, "little") + bytes((1, 0xC0, 3, 7)) + (0x55555555).to_bytes(4, "little")
        for offset, value in enumerate(encoded):
            bus.write8(source + offset, value)
        cpu = Arm7Tdmi(bus)
        cpu.registers[0:2] = [source, destination]

        cpu.step()

        self.assertEqual(bytes(bus.read8(destination + i) for i in range(4)), bytes((0x73,) * 4))

    def test_huffman_bios_follows_a_deeper_tree_and_word_alignment(self) -> None:
        code = (0xEF130000).to_bytes(4, "little")
        bus = MemoryBus(cartridge_with_code(code))
        source = 0x02000000
        destination = 0x02001000
        header = (4 << 8) | (2 << 4) | 8
        # Root chooses A directly or a second node; the stream decodes ABCB.
        encoded = (
            header.to_bytes(4, "little")
            + bytes((2, 0x80, ord("A"), 0xC0, ord("B"), ord("C"), 0, 0))
            + (0x5C000000).to_bytes(4, "little")
        )
        for offset, value in enumerate(encoded):
            bus.write8(source + offset, value)
        cpu = Arm7Tdmi(bus)
        cpu.registers[0:2] = [source, destination]

        cpu.step()

        self.assertEqual(bytes(bus.read8(destination + i) for i in range(4)), b"ABCB")

    def test_bg_affine_bios_builds_scaled_matrix_and_center_offset(self) -> None:
        code = (0xEF0E0000).to_bytes(4, "little")  # SWI BgAffineSet
        bus = MemoryBus(cartridge_with_code(code))
        source = 0x02000000
        destination = 0x02000100
        bus.write32(source, 10 << 8)  # Original texture center, 24.8 fixed point.
        bus.write32(source + 4, 20 << 8)
        bus.write16(source + 8, 5)  # Display center.
        bus.write16(source + 10, 7)
        bus.write16(source + 12, 0x0200)  # X scale 2.0
        bus.write16(source + 14, 0x0100)  # Y scale 1.0
        bus.write16(source + 16, 0)  # No rotation.
        cpu = Arm7Tdmi(bus)
        cpu.registers[0:3] = [source, destination, 1]

        result = cpu.step()

        self.assertEqual(result.operation, "SWI BgAffineSet")
        self.assertEqual([bus.read16(destination + i * 2) for i in range(4)], [0x0200, 0, 0, 0x0100])
        self.assertEqual(bus.read32(destination + 8), 0)
        self.assertEqual(bus.read32(destination + 12), (13 << 8))

    def test_obj_affine_bios_writes_matrix_at_oam_stride(self) -> None:
        code = (0xEF0F0000).to_bytes(4, "little")  # SWI ObjAffineSet
        bus = MemoryBus(cartridge_with_code(code))
        source = 0x02000000
        destination = 0x02000100
        bus.write16(source, 0x0100)
        bus.write16(source + 2, 0x0200)
        bus.write16(source + 4, 0x4000)  # Quarter turn.
        for offset in range(0, 32, 2):
            bus.write16(destination + offset, 0x5555)
        cpu = Arm7Tdmi(bus)
        cpu.registers[0:4] = [source, destination, 1, 8]

        result = cpu.step()

        self.assertEqual(result.operation, "SWI ObjAffineSet")
        self.assertEqual([bus.read16(destination + i * 8) for i in range(4)], [0, 0xFF00, 0x0200, 0])
        self.assertEqual(bus.read16(destination + 2), 0x5555)  # OAM padding stays untouched.


if __name__ == "__main__":
    unittest.main()
