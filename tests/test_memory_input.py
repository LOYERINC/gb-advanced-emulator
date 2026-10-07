"""Checks for the active-low GBA keypad input register."""

import unittest

from gba.cartridge import Cartridge
from gba.memory import MemoryBus


class MemoryInputTests(unittest.TestCase):
    def test_keyinput_is_active_low_and_read_only(self) -> None:
        cartridge = Cartridge(bytes(0xC0), "TEST", "TST0", "00", 0, 0, 0, 0, 0)
        bus = MemoryBus(cartridge)

        self.assertEqual(bus.read16(0x04000130), 0x03FF)
        bus.set_key("a", True)
        bus.set_key("left", True)
        self.assertEqual(bus.read16(0x04000130), 0x03DE)
        bus.write16(0x04000130, 0)
        self.assertEqual(bus.read16(0x04000130), 0x03DE)
        bus.set_key("a", False)
        bus.set_key("left", False)
        self.assertEqual(bus.read16(0x04000130), 0x03FF)


if __name__ == "__main__":
    unittest.main()
