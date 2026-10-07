"""Checks for affine tiled background sampling."""

import unittest

from gba.cartridge import Cartridge
from gba.memory import MemoryBus
from gba.video import GbaVideo, SCREEN_WIDTH


class AffineVideoTests(unittest.TestCase):
    def test_mode2_bg2_identity_matrix_samples_8bpp_tiles(self) -> None:
        cartridge = Cartridge(bytes(0xC0), "TEST", "TST0", "00", 0, 0, 0, 0, 0)
        bus = MemoryBus(cartridge)
        bus.write16(0x04000000, 2 | (1 << 10))  # Mode 2, enable BG2.
        bus.write16(0x0400000C, 1 << 2)  # BG2 uses character block 1.
        bus.write16(0x04000020, 0x0100)  # PA=1.0
        bus.write16(0x04000026, 0x0100)  # PD=1.0
        bus.write8(0x06000000, 0)  # Map pixel (0, 0) to tile 0.
        bus.write8(0x06004000, 1)  # Tile pixel 0 uses palette color 1.
        bus.write16(0x05000002, 0x001F)  # Red.

        frame = GbaVideo(bus).frame_rgb()

        self.assertEqual(frame[:3], bytes((255, 0, 0)))
        self.assertEqual(frame[SCREEN_WIDTH * 3 : SCREEN_WIDTH * 3 + 3], bytes((0, 0, 0)))

    def test_affine_reference_point_can_scroll_source(self) -> None:
        cartridge = Cartridge(bytes(0xC0), "TEST", "TST0", "00", 0, 0, 0, 0, 0)
        bus = MemoryBus(cartridge)
        bus.write16(0x04000000, 2 | (1 << 10))
        bus.write16(0x0400000C, 1 << 2)
        bus.write16(0x04000020, 0x0100)
        bus.write16(0x04000026, 0x0100)
        bus.write32(0x04000028, 0x00000100)  # BG2X = 1 pixel.
        bus.write8(0x06000000, 0)
        bus.write8(0x06000001, 0)
        bus.write8(0x06004000, 1)
        bus.write8(0x06004001, 2)
        bus.write16(0x05000002, 0x001F)  # Red.
        bus.write16(0x05000004, 0x03E0)  # Green.

        frame = GbaVideo(bus).frame_rgb()

        self.assertEqual(frame[:3], bytes((0, 255, 0)))


if __name__ == "__main__":
    unittest.main()
