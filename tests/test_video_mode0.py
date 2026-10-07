"""Checks for GBA mode 0 text-background tile and palette rendering."""

import unittest

from gba.cartridge import Cartridge
from gba.memory import MemoryBus
from gba.video import DISPCNT, GbaVideo, SCREEN_WIDTH


def mode0_bus() -> MemoryBus:
    cartridge = Cartridge(bytes(0xC0), "TEST", "TST0", "00", 0, 0, 0, 0, 0)
    bus = MemoryBus(cartridge)
    bus.write16(DISPCNT, 1 << 8)  # Enable BG0 in mode 0.
    return bus


def pixel(frame: bytes, x: int, y: int = 0) -> tuple[int, int, int]:
    start = (y * SCREEN_WIDTH + x) * 3
    return tuple(frame[start : start + 3])


class Mode0VideoTests(unittest.TestCase):
    def test_4bpp_tiles_use_palette_bank_and_transparency(self) -> None:
        bus = mode0_bus()
        bus.write16(0x04000008, 0)  # BG0 uses char/map block zero, 4bpp.
        bus.write16(0x06000000, 0x1001)  # Tile 1, palette bank 1.
        bus.write8(0x06000020, 0x21)  # Pixel 0=index 1; pixel 1=index 2.
        bus.write16(0x05000022, 0x001F)  # Palette index 17: red.
        bus.write16(0x05000024, 0x03E0)  # Palette index 18: green.
        bus.write16(0x05000020, 0x7C00)  # Palette index 16: blue (nibble 0 stays transparent).

        frame = GbaVideo(bus).frame_rgb()

        self.assertEqual(pixel(frame, 0), (255, 0, 0))
        self.assertEqual(pixel(frame, 1), (0, 255, 0))
        self.assertEqual(pixel(frame, 2), (0, 0, 0))

    def test_8bpp_tiles_use_direct_palette_index(self) -> None:
        bus = mode0_bus()
        bus.write16(0x04000008, 1 << 7)  # BG0 8bpp.
        bus.write16(0x06000000, 1)  # First map entry points to tile 1.
        bus.write8(0x06000040, 2)  # First tile pixel uses palette index 2.
        bus.write16(0x05000004, 0x03FF)  # Palette index 2: yellow.

        frame = GbaVideo(bus).frame_rgb()

        self.assertEqual(pixel(frame, 0), (255, 255, 0))


if __name__ == "__main__":
    unittest.main()
