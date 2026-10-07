"""Build a tiny original GBA homebrew ROM for end-to-end emulator checks."""

from __future__ import annotations

from pathlib import Path


ROM_SIZE = 0x200
PROGRAM_OFFSET = 0xC0


def build_rom() -> bytes:
    rom = bytearray(ROM_SIZE)
    # The GBA begins at 0x08000000. This branch enters ARM code at 0x080000C0.
    rom[0:4] = (0xEA00002E).to_bytes(4, "little")
    rom[0xA0:0xAC] = b"PYGBA DEMO\0\0"
    rom[0xAC:0xB0] = b"PGDE"
    rom[0xB0:0xB2] = b"00"
    rom[0xB2] = 0x96
    rom[0xBC] = 0
    rom[0xBD] = (0xE7 - sum(rom[0xA0:0xBD])) & 0xFF

    # Enable Mode 3, then fill all 240x160 pixels with incrementing BGR555
    # values. The final branch holds the program at a harmless idle loop.
    instructions = (
        0xE59F0024,  # LDR r0, =DISPCNT
        0xE59F2024,  # LDR r2, =MODE3_BG2
        0xE1C020B0,  # STRH r2, [r0]
        0xE59F0020,  # LDR r0, =VRAM
        0xE3A01000,  # MOV r1, #0
        0xE59F201C,  # LDR r2, =240*160
        0xE0C010B2,  # STRH r1, [r0], #2
        0xE2811001,  # ADD r1, r1, #1
        0xE2522001,  # SUBS r2, r2, #1
        0x1AFFFFFB,  # BNE to the first pixel write
        0xEAFFFFFE,  # B to this idle instruction
        0x04000000,  # Literal: DISPCNT
        0x00000403,  # Literal: Mode 3 + BG2 enable
        0x06000000,  # Literal: VRAM
        240 * 160,   # Literal: screen pixel count
    )
    for index, instruction in enumerate(instructions):
        offset = PROGRAM_OFFSET + index * 4
        rom[offset : offset + 4] = instruction.to_bytes(4, "little")
    return bytes(rom)


if __name__ == "__main__":
    output = Path(__file__).with_name("gradient_demo.gba")
    output.write_bytes(build_rom())
    print(f"Wrote {output}")
