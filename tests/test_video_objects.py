"""Checks for simple non-affine GBA object/sprite drawing."""

import unittest

from gba.cartridge import Cartridge
from gba.memory import MemoryBus
from gba.video import BG2_ENABLE, DISPCNT, GbaVideo, OBJ_ENABLE, SCREEN_WIDTH


def new_bus() -> MemoryBus:
    cartridge = Cartridge(bytes(0xC0), "TEST", "TST0", "00", 0, 0, 0, 0, 0)
    return MemoryBus(cartridge)


def first_pixel(frame: bytes, x: int = 0) -> tuple[int, int, int]:
    start = x * 3
    return tuple(frame[start : start + 3])


def set_obj(bus: MemoryBus, attr2: int = 0, attr1: int = 0) -> None:
    bus.write16(0x07000000, 0)  # At (0,0), regular 8x8 object.
    bus.write16(0x07000002, attr1)
    bus.write16(0x07000004, attr2)


class VideoObjectTests(unittest.TestCase):
    def test_obj_palette_and_transparent_pixel(self) -> None:
        bus = new_bus()
        bus.write16(DISPCNT, OBJ_ENABLE)
        set_obj(bus)
        bus.write8(0x06010000, 0x01)  # x=0 uses color 1; x=1 uses transparent color 0.
        bus.write16(0x05000202, 0x001F)  # OBJ palette color 1: red.

        frame = GbaVideo(bus).frame_rgb()

        self.assertEqual(first_pixel(frame, 0), (255, 0, 0))
        self.assertEqual(first_pixel(frame, 1), (0, 0, 0))

    def test_obj_priority_against_text_background(self) -> None:
        bus = new_bus()
        bus.write16(DISPCNT, OBJ_ENABLE | (1 << 8))
        bus.write16(0x04000008, 0)  # BG0 priority 0.
        bus.write16(0x06000000, 1)  # BG0 tile map points to tile 1.
        bus.write8(0x06000020, 0x01)  # Its first pixel uses BG color 1.
        bus.write16(0x05000002, 0x7C00)  # BG color 1: blue.
        bus.write16(0x06010000, 0x01)  # OBJ first pixel uses OBJ color 1.
        bus.write16(0x05000202, 0x001F)  # OBJ color 1: red.
        set_obj(bus, attr2=1 << 10)  # OBJ priority 1 is behind BG priority 0.

        frame = GbaVideo(bus).frame_rgb()
        self.assertEqual(first_pixel(frame), (0, 0, 255))

        bus.write16(0x07000004, 0)  # OBJ priority 0 wins a same-priority tie.
        frame = GbaVideo(bus).frame_rgb()
        self.assertEqual(first_pixel(frame), (255, 0, 0))

    def test_bitmap_mode_obj_uses_upper_vram_tiles(self) -> None:
        bus = new_bus()
        bus.write16(DISPCNT, 3 | OBJ_ENABLE)  # Mode 3 with BG2 off, OBJ on.
        set_obj(bus, attr2=512)  # Bitmap modes reserve OBJ tiles 0-511 for BG data.
        bus.write8(0x06014000, 0x01)
        bus.write16(0x05000202, 0x001F)

        frame = GbaVideo(bus).frame_rgb()

        self.assertEqual(first_pixel(frame), (255, 0, 0))
        self.assertTrue(GbaVideo(bus).display_output_enabled)

    def test_affine_obj_rotates_source_pixels(self) -> None:
        bus = new_bus()
        bus.write16(DISPCNT, OBJ_ENABLE)
        bus.write16(0x07000000, 1 << 8)  # Affine, at (0,0), 8x8.
        bus.write16(0x07000002, 0)  # Matrix 0.
        bus.write16(0x07000004, 0)
        # A quarter turn maps source pixel (0,4) to screen pixel (4,0).
        for address, value in zip((0x07000006, 0x0700000E, 0x07000016, 0x0700001E), (0, 0x0100, 0xFF00, 0)):
            bus.write16(address, value)
        bus.write8(0x06010000 + 4 * 4, 0x01)
        bus.write16(0x05000202, 0x001F)

        frame = GbaVideo(bus).frame_rgb()

        self.assertEqual(first_pixel(frame, 4), (255, 0, 0))
        self.assertEqual(first_pixel(frame, 0), (0, 0, 0))

    def test_affine_double_size_sprite_is_enabled_and_centered(self) -> None:
        bus = new_bus()
        bus.write16(DISPCNT, OBJ_ENABLE)
        bus.write16(0x07000000, (1 << 8) | (1 << 9))  # Affine + double-size.
        bus.write16(0x07000002, 0)
        bus.write16(0x07000004, 0)
        for address, value in zip((0x07000006, 0x0700000E, 0x07000016, 0x0700001E), (0x0100, 0, 0, 0x0100)):
            bus.write16(address, value)
        bus.write8(0x06010000, 0x01)  # Source pixel (0,0).
        bus.write16(0x05000202, 0x03E0)  # Green.

        frame = GbaVideo(bus).frame_rgb()
        pixel = ((4 * SCREEN_WIDTH) + 4) * 3
        offset_pixel = ((0 * SCREEN_WIDTH) + 0) * 3

        self.assertEqual(tuple(frame[pixel : pixel + 3]), (0, 255, 0))
        self.assertEqual(tuple(frame[offset_pixel : offset_pixel + 3]), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
