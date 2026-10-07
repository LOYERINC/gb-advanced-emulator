"""Checks immediate and event-triggered DMA behavior."""

import unittest

from gba.cartridge import Cartridge
from gba.memory import MemoryBus


def make_bus() -> MemoryBus:
    cartridge = Cartridge(bytes(0xC0), "TEST", "TST0", "00", 0, 0, 0, 0, 0)
    return MemoryBus(cartridge)


class DmaTests(unittest.TestCase):
    def test_immediate_halfword_dma_copies_and_disables_channel(self) -> None:
        bus = make_bus()
        source = 0x02000000
        destination = 0x02001000
        bus.write16(source, 0x1234)
        bus.write16(source + 2, 0xABCD)
        bus.write32(0x040000B0, source)
        bus.write32(0x040000B4, destination)
        bus.write16(0x040000B8, 2)
        bus.write16(0x040000BA, 0x8000)  # Enable immediately.

        self.assertEqual(bus.read16(destination), 0x1234)
        self.assertEqual(bus.read16(destination + 2), 0xABCD)
        self.assertEqual(bus.read32(0x040000B0), source + 4)
        self.assertEqual(bus.read32(0x040000B4), destination + 4)
        self.assertEqual(bus.read16(0x040000BA) & 0x8000, 0)

    def test_word_dma_can_fill_using_fixed_source(self) -> None:
        bus = make_bus()
        source = 0x02000000
        destination = 0x02001000
        bus.write32(source, 0xDEADBEEF)
        bus.write32(0x040000BC, source)  # DMA1 source
        bus.write32(0x040000C0, destination)
        bus.write16(0x040000C4, 3)
        control = 0x8000 | (1 << 10) | (2 << 7)  # Immediate, 32-bit, fixed source.
        bus.write16(0x040000C6, control)

        self.assertEqual([bus.read32(destination + i * 4) for i in range(3)], [0xDEADBEEF] * 3)

    def test_vblank_repeat_dma_reloads_destination_until_triggered(self) -> None:
        bus = make_bus()
        source = 0x02000000
        destination = 0x02001000
        bus.write32(source, 0x11111111)
        bus.write32(source + 4, 0x22222222)
        bus.write32(0x040000B0, source)
        bus.write32(0x040000B4, destination)
        bus.write16(0x040000B8, 1)
        control = 0x8000 | (1 << 9) | (3 << 5) | (1 << 10) | (1 << 12)
        bus.write16(0x040000BA, control)  # Repeat, 32-bit, destination reload, VBlank.

        self.assertEqual(bus.read32(destination), 0)
        bus.trigger_dma(2)  # HBlank does not match.
        self.assertEqual(bus.read32(destination), 0)
        bus.trigger_dma(1)
        self.assertEqual(bus.read32(destination), 0x11111111)
        self.assertEqual(bus.read32(0x040000B4), destination)
        bus.trigger_dma(1)
        self.assertEqual(bus.read32(destination), 0x22222222)
        self.assertTrue(bus.read16(0x040000BA) & 0x8000)


if __name__ == "__main__":
    unittest.main()
