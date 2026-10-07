"""Pixel checks for the first GBA bitmap render modes."""

import unittest

from gba.cartridge import Cartridge
from gba.memory import MemoryBus
from gba.video import BG2_ENABLE, DISPCNT, FRAME_SELECT, GbaVideo, SCREEN_WIDTH


def empty_bus() -> MemoryBus:
    cartridge = Cartridge(bytes(0xC0), "TEST", "TST0", "00", 0, 0, 0, 0, 0)
    return MemoryBus(cartridge)


class VideoTests(unittest.TestCase):
    def test_mode3_renders_direct_color(self) -> None:
        bus = empty_bus()
        bus.write16(DISPCNT, 3 | BG2_ENABLE)
        bus.write16(0x06000000, 0x001F)  # Full red in BGR555.

        frame = GbaVideo(bus).frame_rgb()

        self.assertEqual(frame[:3], bytes((255, 0, 0)))

    def test_mode4_uses_palette_and_selected_page(self) -> None:
        bus = empty_bus()
        bus.write16(DISPCNT, 4 | BG2_ENABLE | FRAME_SELECT)
        bus.palette_ram[4:6] = (0x03E0).to_bytes(2, "little")  # Palette index 2: green.
        bus.write8(0x0600A000, 2)

        frame = GbaVideo(bus).frame_rgb()

        self.assertEqual(frame[:3], bytes((0, 255, 0)))

    def test_mode5_centers_small_frame(self) -> None:
        bus = empty_bus()
        bus.write16(DISPCNT, 5 | BG2_ENABLE)
        bus.write16(0x06000000, 0x7C00)  # Full blue in BGR555.

        frame = GbaVideo(bus).frame_rgb()
        centered_pixel = ((16 * SCREEN_WIDTH) + 40) * 3

        self.assertEqual(frame[:3], bytes((0, 0, 0)))
        self.assertEqual(frame[centered_pixel : centered_pixel + 3], bytes((0, 0, 255)))


if __name__ == "__main__":
    unittest.main()
