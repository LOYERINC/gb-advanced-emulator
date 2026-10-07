"""Entry point for the GBA emulator learning project."""

from __future__ import annotations

import argparse
import sys

from gba.cartridge import Cartridge, CartridgeError
from gba.memory import MemoryBus


def main() -> int:
    parser = argparse.ArgumentParser(description="GBA emulator learning project")
    parser.add_argument("rom", help="path to a .gba ROM file")
    args = parser.parse_args()

    try:
        cartridge = Cartridge.load(args.rom)
    except (OSError, CartridgeError) as exc:
        print(f"Could not load ROM: {exc}", file=sys.stderr)
        return 1

    print("ROM file loaded")
    print(f"  Title:     {cartridge.title or '(no title)'}")
    print(f"  Game code: {cartridge.game_code or '(unknown)'}")
    print(f"  Maker:     {cartridge.maker_code or '(unknown)'}")
    print(f"  Version:   {cartridge.version}")
    print(f"  Size:      {len(cartridge.data):,} bytes")
    print(f"  Save type: {cartridge.save_type or '(not identified)'}")
    print(
        "  Header:    "
        f"{'valid' if not cartridge.header_warnings else 'check warnings'} "
        f"(checksum 0x{cartridge.header_checksum:02X})"
    )
    for warning in cartridge.header_warnings:
        print(f"  Warning:   {warning}")
    bus = MemoryBus(cartridge)
    first_word = bus.read32(0x08000000)
    print(f"  First ROM word: 0x{first_word:08X} (read through the memory bus)")
    print("\nTo run the ROM in the desktop emulator, use: python emulator_app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
