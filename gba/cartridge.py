"""Game Pak ROM loading and GBA cartridge-header validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile, ZipFile


HEADER_END = 0xC0
HEADER_START = 0xA0
HEADER_CHECKSUM_OFFSET = 0xBD
HEADER_FIXED_VALUE_OFFSET = 0xB2
HEADER_FIXED_VALUE = 0x96


class CartridgeError(ValueError):
    """Raised when a file cannot be treated as a GBA cartridge ROM."""


def _read_ascii(data: bytes, start: int, length: int) -> str:
    return data[start : start + length].split(b"\0", 1)[0].decode("ascii", errors="replace").strip()


@dataclass(frozen=True)
class Cartridge:
    data: bytes
    title: str
    game_code: str
    maker_code: str
    unit_code: int
    device_type: int
    version: int
    header_checksum: int
    expected_checksum: int
    save_type: str | None = None
    save_path: Path | None = None
    has_rtc: bool = False

    @property
    def checksum_valid(self) -> bool:
        return self.header_checksum == self.expected_checksum

    @property
    def header_warnings(self) -> tuple[str, ...]:
        warnings = []
        if self.data[HEADER_FIXED_VALUE_OFFSET] != HEADER_FIXED_VALUE:
            warnings.append(
                "fixed header byte is "
                f"0x{self.data[HEADER_FIXED_VALUE_OFFSET]:02X}, expected 0x{HEADER_FIXED_VALUE:02X}"
            )
        if not self.checksum_valid:
            warnings.append(
                f"header checksum is 0x{self.header_checksum:02X}, "
                f"expected 0x{self.expected_checksum:02X}"
            )
        return tuple(warnings)

    @classmethod
    def load(cls, path: str | Path) -> "Cartridge":
        rom_path = Path(path).expanduser()
        if rom_path.suffix.lower() == ".zip":
            try:
                with ZipFile(rom_path) as archive:
                    candidates = [
                        item for item in archive.infolist()
                        if not item.is_dir() and Path(item.filename).suffix.lower() in (".gba", ".agb")
                    ]
                    if not candidates:
                        raise CartridgeError("ZIP file contains no .gba or .agb ROM")
                    if len(candidates) > 1:
                        raise CartridgeError("ZIP file contains multiple GBA ROMs; choose one ROM per ZIP")
                    rom_entry = candidates[0]
                    if rom_entry.file_size > 32 * 1024 * 1024:
                        raise CartridgeError("GBA ROM inside ZIP is larger than the 32 MiB cartridge limit")
                    data = archive.read(rom_entry)
            except (BadZipFile, NotImplementedError, RuntimeError) as exc:
                raise CartridgeError(f"could not read ZIP ROM: {exc}") from exc
        else:
            data = rom_path.read_bytes()
        if not data:
            raise CartridgeError("file is empty")
        if len(data) < HEADER_END:
            raise CartridgeError(
                f"file is only {len(data)} bytes; a GBA ROM needs at least {HEADER_END} bytes"
            )

        # The complement byte is chosen so the sum of header bytes A0-BD,
        # plus 0x19, wraps to zero in an 8-bit value.
        expected_checksum = (0xE7 - sum(data[HEADER_START:HEADER_CHECKSUM_OFFSET])) & 0xFF
        save_markers = (
            (b"FLASH1M_V", "FLASH1M"),
            (b"FLASH512_V", "FLASH"),
            (b"FLASH_V", "FLASH"),
            (b"SRAM_F_V", "SRAM"),
            (b"SRAM_V", "SRAM"),
            (b"EEPROM512_V", "EEPROM"),
            (b"EEPROM_V", "EEPROM"),
        )
        save_type = next(
            (kind for marker, kind in save_markers if marker in data),
            None,
        )

        return cls(
            data=data,
            title=_read_ascii(data, 0xA0, 12),
            game_code=_read_ascii(data, 0xAC, 4),
            maker_code=_read_ascii(data, 0xB0, 2),
            unit_code=data[0xB3],
            device_type=data[0xB4],
            version=data[0xBC],
            header_checksum=data[HEADER_CHECKSUM_OFFSET],
            expected_checksum=expected_checksum,
            save_type=save_type,
            save_path=rom_path.with_suffix(".sav"),
            has_rtc=b"SIIRTC_V" in data,
        )
