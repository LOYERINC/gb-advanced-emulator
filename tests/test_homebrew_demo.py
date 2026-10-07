"""End-to-end check for the included original homebrew ROM."""

import unittest
from pathlib import Path

from gba.arm7tdmi import Arm7Tdmi
from gba.cartridge import Cartridge
from gba.memory import MemoryBus
from gba.video import DISPCNT, GbaVideo, SCREEN_HEIGHT, SCREEN_WIDTH


DEMO_ROM = Path(__file__).parents[1] / "examples" / "gradient_demo.gba"


class HomebrewDemoTests(unittest.TestCase):
    def test_rom_executes_and_fills_the_display_frame(self) -> None:
        cartridge = Cartridge.load(DEMO_ROM)
        self.assertEqual(cartridge.title, "PYGBA DEMO")
        self.assertTrue(cartridge.checksum_valid)
        bus = MemoryBus(cartridge)
        cpu = Arm7Tdmi(bus)

        # Header branch + setup + four instructions for each frame pixel.
        instruction_count = 1 + 6 + (SCREEN_WIDTH * SCREEN_HEIGHT * 4)
        for _ in range(instruction_count):
            cpu.step()

        self.assertEqual(cpu.registers[2], 0)
        self.assertEqual(cpu.registers[15], 0x080000E8)  # Idle loop after the draw.
        self.assertEqual(bus.read16(DISPCNT), 3 | (1 << 10))
        self.assertEqual(bus.read16(0x06000000), 0)
        self.assertEqual(bus.read16(0x06000002), 1)
        last_pixel_address = 0x06000000 + 2 * (SCREEN_WIDTH * SCREEN_HEIGHT - 1)
        self.assertEqual(bus.read16(last_pixel_address), 0x95FF)

        frame = GbaVideo(bus).frame_rgb()
        self.assertEqual(frame[:3], bytes((0, 0, 0)))
        second_pixel = frame[3:6]
        self.assertEqual(second_pixel, bytes((8, 0, 0)))


if __name__ == "__main__":
    unittest.main()
