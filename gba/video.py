"""First-pass renderer for text backgrounds and bitmap display modes."""

from __future__ import annotations

from dataclasses import dataclass, field

from gba.memory import MemoryBus
try:
    from gba._native_cpu import render_mode0_fast as _render_native_mode0
except ImportError:  # Keep the source-only Python renderer usable.
    _render_native_mode0 = None


SCREEN_WIDTH = 240
SCREEN_HEIGHT = 160
DISPCNT = 0x04000000
BG2_ENABLE = 1 << 10
FRAME_SELECT = 1 << 4
OBJ_ENABLE = 1 << 12
OBJ_VRAM_BASE = 0x10000
_RGB555_TRANSLATION = {
    color: "".join(
        chr(component)
        for component in (
            ((color & 0x1F) << 3) | ((color & 0x1F) >> 2),
            (((color >> 5) & 0x1F) << 3) | (((color >> 5) & 0x1F) >> 2),
            (((color >> 10) & 0x1F) << 3) | (((color >> 10) & 0x1F) >> 2),
        )
    )
    for color in range(0x8000)
}


@dataclass
class GbaVideo:
    """Turn the current bitmap framebuffer into RGB pixels for the UI."""

    bus: MemoryBus
    _cached_generation: int = field(default=-1, init=False, repr=False)
    _cached_frame: bytes = field(default=b"", init=False, repr=False)

    @property
    def mode(self) -> int:
        return self.bus.read16(DISPCNT) & 0x7

    @property
    def display_output_enabled(self) -> bool:
        control = self.bus.read16(DISPCNT)
        if control & (1 << 7):
            return True  # Forced blank is itself a visible white display.
        mode = control & 0x7
        if mode == 0:
            return bool(control & 0x0F00) or bool(control & OBJ_ENABLE)
        if mode in (1, 2):
            return bool(control & 0x1F00)
        if mode in (3, 4, 5):
            return bool(control & BG2_ENABLE) or bool(control & OBJ_ENABLE)
        return False

    def _rgb555(self, value: int) -> tuple[int, int, int]:
        red5 = value & 0x1F
        green5 = (value >> 5) & 0x1F
        blue5 = (value >> 10) & 0x1F
        return (
            (red5 << 3) | (red5 >> 2),
            (green5 << 3) | (green5 >> 2),
            (blue5 << 3) | (blue5 >> 2),
        )

    def _palette_color(self, index: int, object_palette: bool = False) -> tuple[int, int, int]:
        address = (0x200 if object_palette else 0) + (index & 0xFF) * 2
        color = self.bus.palette_ram[address] | (self.bus.palette_ram[address + 1] << 8)
        return self._rgb555(color)

    @staticmethod
    def _inside_window_axis(coordinate: int, bounds: int) -> bool:
        start = (bounds >> 8) & 0xFF
        end = bounds & 0xFF
        if start < end:
            return start <= coordinate < end
        if start > end:
            return coordinate >= start or coordinate < end
        return False

    def _object_window_pixels(self, control: int) -> bytearray:
        """Rasterize opaque OBJ-window texels into a per-pixel mask."""
        mask = bytearray(SCREEN_WIDTH * SCREEN_HEIGHT)
        if not (control & OBJ_ENABLE) or not (control & (1 << 15)):
            return mask
        shape_sizes = (
            ((8, 8), (16, 16), (32, 32), (64, 64)),
            ((16, 8), (32, 8), (32, 16), (64, 32)),
            ((8, 16), (8, 32), (16, 32), (32, 64)),
        )
        mapping_1d = bool(control & (1 << 6))
        bitmap_mode = (control & 0x7) >= 3
        mosaic = self.bus.read16(0x0400004C)
        obj_mosaic_h = ((mosaic >> 8) & 0xF) + 1
        obj_mosaic_v = ((mosaic >> 12) & 0xF) + 1
        tile_rows: dict[tuple[int, int, bool], tuple[int, ...]] = {}
        for index in range(128):
            offset = index * 8
            attr0 = self.bus.oam[offset] | (self.bus.oam[offset + 1] << 8)
            if ((attr0 >> 10) & 3) != 2:
                continue
            attr1 = self.bus.oam[offset + 2] | (self.bus.oam[offset + 3] << 8)
            attr2 = self.bus.oam[offset + 4] | (self.bus.oam[offset + 5] << 8)
            affine = bool(attr0 & (1 << 8))
            double_size = affine and bool(attr0 & (1 << 9))
            if not affine and (attr0 & (1 << 9)):
                continue
            shape = (attr0 >> 14) & 3
            size = (attr1 >> 14) & 3
            if shape == 3:
                continue
            width, height = shape_sizes[shape][size]
            draw_width = width * (2 if double_size else 1)
            draw_height = height * (2 if double_size else 1)
            x = attr1 & 0x1FF
            y = attr0 & 0xFF
            if x >= 256:
                x -= 512
            if y >= 160:
                y -= 256
            color256 = bool(attr0 & (1 << 13))
            tile_number = attr2 & 0x3FF
            if bitmap_mode and tile_number < 512:
                continue
            tile_unit = 2 if color256 else 1
            if color256:
                tile_number &= ~1
            width_tiles = width // 8
            matrix = None
            if affine:
                matrix_address = 0x07000006 + ((attr1 >> 9) & 0x1F) * 32
                matrix = tuple(self.bus.read16(matrix_address + offset) for offset in (0, 8, 16, 24))
                matrix = tuple(value - 0x10000 if value & 0x8000 else value for value in matrix)

            for local_y in range(draw_height):
                screen_y = y + local_y
                if not 0 <= screen_y < SCREEN_HEIGHT:
                    continue
                for local_x in range(draw_width):
                    screen_x = x + local_x
                    if not 0 <= screen_x < SCREEN_WIDTH:
                        continue
                    if affine:
                        pa, pb, pc, pd = matrix
                        sample_x = local_x
                        sample_y = local_y
                        if attr0 & (1 << 12):
                            sample_x = max(0, ((screen_x // obj_mosaic_h) * obj_mosaic_h) - x)
                            sample_y = max(0, ((screen_y // obj_mosaic_v) * obj_mosaic_v) - y)
                        dx = sample_x - draw_width // 2
                        dy = sample_y - draw_height // 2
                        source_x = (pa * dx + pb * dy + (width << 7)) >> 8
                        source_y = (pc * dx + pd * dy + (height << 7)) >> 8
                        if not (0 <= source_x < width and 0 <= source_y < height):
                            continue
                    else:
                        sample_x = local_x
                        sample_y = local_y
                        if attr0 & (1 << 12):
                            sample_x = max(0, ((screen_x // obj_mosaic_h) * obj_mosaic_h) - x)
                            sample_y = max(0, ((screen_y // obj_mosaic_v) * obj_mosaic_v) - y)
                        source_x = width - 1 - sample_x if attr1 & (1 << 12) else sample_x
                        source_y = height - 1 - sample_y if attr1 & (1 << 13) else sample_y
                    tile_x, pixel_x = divmod(source_x, 8)
                    tile_y, pixel_y = divmod(source_y, 8)
                    tile_offset_index = (
                        tile_y * width_tiles * tile_unit + tile_x * tile_unit
                        if mapping_1d else tile_y * 32 + tile_x * tile_unit
                    )
                    tile_address = OBJ_VRAM_BASE + (tile_number + tile_offset_index) * 32
                    row_key = (tile_address, pixel_y, color256)
                    tile_row = tile_rows.get(row_key)
                    if tile_row is None:
                        if color256:
                            tile_row = tuple(
                                self._vram_byte(tile_address + pixel_y * 8 + column)
                                for column in range(8)
                            )
                        else:
                            row = []
                            for column in range(8):
                                packed = self._vram_byte(
                                    tile_address + pixel_y * 4 + column // 2
                                )
                                row.append((packed >> (4 if column & 1 else 0)) & 0xF)
                            tile_row = tuple(row)
                        tile_rows[row_key] = tile_row
                    color_index = tile_row[pixel_x]
                    if color_index:
                        mask[screen_y * SCREEN_WIDTH + screen_x] = 1
        return mask

    def _window_masks(
        self, control: int, object_window_pixels: bytearray
    ) -> tuple[bytearray, bytearray]:
        """Build per-pixel layer and color-effect masks for WIN0/WIN1."""
        pixel_count = SCREEN_WIDTH * SCREEN_HEIGHT
        if not (control & 0xE000):
            return bytearray([0x1F]) * pixel_count, bytearray([1]) * pixel_count

        window_out = self.bus.read16(0x0400004A)
        outside = window_out & 0x3F
        layer_masks = bytearray([outside & 0x1F]) * pixel_count
        effect_masks = bytearray([1 if outside & 0x20 else 0]) * pixel_count
        if control & (1 << 15):
            object_window = (window_out >> 8) & 0x3F
            for pixel, covered in enumerate(object_window_pixels):
                if covered:
                    layer_masks[pixel] = object_window & 0x1F
                    effect_masks[pixel] = 1 if object_window & 0x20 else 0
        window_in = self.bus.read16(0x04000048)

        # WIN1 is lower priority than WIN0, so paint its region first.
        for window in (1, 0):
            if not (control & (1 << (13 + window))):
                continue
            horizontal = self.bus.read16(0x04000040 + window * 2)
            vertical = self.bus.read16(0x04000044 + window * 2)
            settings = (window_in >> (8 * window)) & 0x3F
            for y in range(SCREEN_HEIGHT):
                if not self._inside_window_axis(y, vertical):
                    continue
                row = y * SCREEN_WIDTH
                for x in range(SCREEN_WIDTH):
                    if self._inside_window_axis(x, horizontal):
                        pixel = row + x
                        layer_masks[pixel] = settings & 0x1F
                        effect_masks[pixel] = 1 if settings & 0x20 else 0
        return layer_masks, effect_masks

    @staticmethod
    def _plot_pixel(
        pixel_index: int,
        color: tuple[int, int, int],
        layer: int,
        priority: int,
        pixels: bytearray,
        second_pixels: bytearray,
        layers: bytearray,
        second_layers: bytearray,
        priorities: bytearray,
        semi_objects: bytearray,
        window_layers: bytearray,
        semi_object: bool = False,
    ) -> None:
        """Place a visible layer pixel and retain the pixel immediately below it."""
        if layer < 5 and not (window_layers[pixel_index] & (1 << layer)):
            return
        output = pixel_index * 3
        second_pixels[output : output + 3] = pixels[output : output + 3]
        second_layers[pixel_index] = layers[pixel_index]
        pixels[output : output + 3] = bytes(color)
        layers[pixel_index] = layer
        priorities[pixel_index] = priority
        semi_objects[pixel_index] = 1 if semi_object else 0

    def _vram_byte(self, offset: int) -> int:
        if 0 <= offset < len(self.bus.vram):
            return self.bus.vram[offset]
        return self.bus.read8(0x06000000 + offset)

    def _render_mode0(
        self, control: int, pixels: bytearray, priorities: bytearray,
        second_pixels: bytearray, layers: bytearray, second_layers: bytearray,
        semi_objects: bytearray, window_layers: bytearray,
    ) -> None:
        """Composite enabled 4bpp/8bpp tiled text backgrounds."""
        backgrounds = []
        for bg in range(4):
            if control & (1 << (8 + bg)):
                bg_control = self.bus.read16(0x04000008 + bg * 2)
                backgrounds.append(((bg_control & 0x3), bg, bg_control))

        # Draw lower-priority backgrounds first, so higher-priority pixels
        # naturally paint over them. BG0 wins ties with BG1, and so on.
        backgrounds.sort(key=lambda layer: (layer[0], layer[1]), reverse=True)
        if not any(bg_control & (1 << 6) for _priority, _bg, bg_control in backgrounds):
            self._render_mode0_cached_rows(
                backgrounds, pixels, priorities, second_pixels, layers,
                second_layers, semi_objects, window_layers,
            )
            return

        for _priority, bg, bg_control in backgrounds:
            char_base = ((bg_control >> 2) & 0x3) * 0x4000
            color256 = bool(bg_control & (1 << 7))
            screen_base = ((bg_control >> 8) & 0x1F) * 0x800
            size = (bg_control >> 14) & 0x3
            width_tiles = 64 if size in (1, 3) else 32
            height_tiles = 64 if size in (2, 3) else 32
            width_pixels = width_tiles * 8
            height_pixels = height_tiles * 8
            hofs = self.bus.read16(0x04000010 + bg * 4) & 0x1FF
            vofs = self.bus.read16(0x04000012 + bg * 4) & 0x1FF
            tile_bytes = 64 if color256 else 32
            mosaic = self.bus.read16(0x0400004C)
            mosaic_enabled = bool(bg_control & (1 << 6))
            mosaic_h = (mosaic & 0xF) + 1
            mosaic_v = ((mosaic >> 4) & 0xF) + 1

            for screen_y in range(SCREEN_HEIGHT):
                sample_y = (screen_y // mosaic_v) * mosaic_v if mosaic_enabled else screen_y
                bg_y = (sample_y + vofs) % height_pixels
                tile_y, pixel_y = divmod(bg_y, 8)
                for screen_x in range(SCREEN_WIDTH):
                    sample_x = (screen_x // mosaic_h) * mosaic_h if mosaic_enabled else screen_x
                    bg_x = (sample_x + hofs) % width_pixels
                    tile_x, pixel_x = divmod(bg_x, 8)
                    screen_block = (tile_y // 32) * (width_tiles // 32) + (tile_x // 32)
                    map_offset = (
                        screen_base
                        + screen_block * 0x800
                        + ((tile_y & 31) * 32 + (tile_x & 31)) * 2
                    )
                    entry = self._vram_byte(map_offset) | (self._vram_byte(map_offset + 1) << 8)
                    tile_number = entry & 0x3FF
                    if entry & (1 << 10):
                        pixel_x = 7 - pixel_x
                    source_y = 7 - pixel_y if entry & (1 << 11) else pixel_y

                    tile_offset = char_base + tile_number * tile_bytes
                    if color256:
                        palette_index = self._vram_byte(tile_offset + source_y * 8 + pixel_x)
                        transparent = palette_index == 0
                    else:
                        packed = self._vram_byte(tile_offset + source_y * 4 + pixel_x // 2)
                        color_index = (packed >> (4 if pixel_x & 1 else 0)) & 0xF
                        transparent = color_index == 0
                        palette_index = color_index + ((entry >> 12) & 0xF) * 16

                    # Index zero is transparent for tiled backgrounds.
                    if transparent:
                        continue
                    red, green, blue = self._palette_color(palette_index)
                    pixel_index = screen_y * SCREEN_WIDTH + screen_x
                    self._plot_pixel(
                        pixel_index, (red, green, blue), bg, _priority, pixels,
                        second_pixels, layers, second_layers, priorities, semi_objects,
                        window_layers,
                    )

    def _render_mode0_cached_rows(
        self,
        backgrounds: list[tuple[int, int, int]],
        pixels: bytearray,
        priorities: bytearray,
        second_pixels: bytearray,
        layers: bytearray,
        second_layers: bytearray,
        semi_objects: bytearray,
        window_layers: bytearray,
    ) -> None:
        """Composite non-mosaic text BGs by tile segments with cached tile rows."""
        for priority, bg, bg_control in backgrounds:
            char_base = ((bg_control >> 2) & 3) * 0x4000
            color256 = bool(bg_control & (1 << 7))
            screen_base = ((bg_control >> 8) & 0x1F) * 0x800
            size = (bg_control >> 14) & 3
            width_tiles = 64 if size in (1, 3) else 32
            height_tiles = 64 if size in (2, 3) else 32
            width_pixels = width_tiles * 8
            height_pixels = height_tiles * 8
            hofs = self.bus.read16(0x04000010 + bg * 4) & 0x1FF
            vofs = self.bus.read16(0x04000012 + bg * 4) & 0x1FF
            tile_bytes = 64 if color256 else 32
            palette = [self._palette_color(index) for index in range(256)]
            tile_rows: dict[tuple[int, int], tuple[int, ...]] = {}

            for screen_y in range(SCREEN_HEIGHT):
                bg_y = (screen_y + vofs) % height_pixels
                tile_y, pixel_y = divmod(bg_y, 8)
                screen_x = 0
                while screen_x < SCREEN_WIDTH:
                    bg_x = (screen_x + hofs) % width_pixels
                    tile_x, pixel_x = divmod(bg_x, 8)
                    segment = min(8 - pixel_x, SCREEN_WIDTH - screen_x, width_pixels - bg_x)
                    screen_block = (tile_y // 32) * (width_tiles // 32) + (tile_x // 32)
                    map_offset = (
                        screen_base
                        + screen_block * 0x800
                        + ((tile_y & 31) * 32 + (tile_x & 31)) * 2
                    )
                    entry = self._vram_byte(map_offset) | (self._vram_byte(map_offset + 1) << 8)
                    row_key = (entry, pixel_y)
                    tile_row = tile_rows.get(row_key)
                    if tile_row is None:
                        source_y = 7 - pixel_y if entry & (1 << 11) else pixel_y
                        tile_offset = char_base + (entry & 0x3FF) * tile_bytes
                        if color256:
                            tile_row = tuple(
                                self._vram_byte(tile_offset + source_y * 8 + column)
                                for column in range(8)
                            )
                        else:
                            palette_bank = ((entry >> 12) & 0xF) * 16
                            row = []
                            for column in range(8):
                                packed = self._vram_byte(
                                    tile_offset + source_y * 4 + column // 2
                                )
                                color = (packed >> (4 if column & 1 else 0)) & 0xF
                                row.append(palette_bank + color if color else 0)
                            tile_row = tuple(row)
                        tile_rows[row_key] = tile_row

                    for step in range(segment):
                        source_x = 7 - pixel_x - step if entry & (1 << 10) else pixel_x + step
                        palette_index = tile_row[source_x]
                        if palette_index == 0:
                            continue
                        pixel_index = screen_y * SCREEN_WIDTH + screen_x + step
                        self._plot_pixel(
                            pixel_index, palette[palette_index], bg, priority, pixels,
                            second_pixels, layers, second_layers, priorities,
                            semi_objects, window_layers,
                        )
                    screen_x += segment

    def _render_simple_mode0(self, control: int) -> bytes | None:
        """Render ordinary text BG layers without per-pixel layer bookkeeping."""
        enabled = control & 0x0F00
        if (
            not enabled
            or control & (OBJ_ENABLE | 0xE000)
            or self.bus.read16(0x04000050) & 0x00C0
        ):
            return None
        backgrounds = []
        for bg in range(4):
            if enabled & (1 << (8 + bg)):
                bg_control = self.bus.read16(0x04000008 + bg * 2)
                if bg_control & (1 << 6):
                    return None
                backgrounds.append((bg_control & 3, bg, bg_control))
        # Lower-priority layers go down first; BG0 covers BG1 on a tie.
        backgrounds.sort(key=lambda layer: (layer[0], layer[1]), reverse=True)

        backdrop = bytes(self._palette_color(0))
        pixels = bytearray(backdrop * (SCREEN_WIDTH * SCREEN_HEIGHT))
        for _priority, bg, bg_control in backgrounds:
            color256 = bool(bg_control & (1 << 7))
            char_base = ((bg_control >> 2) & 3) * 0x4000
            screen_base = ((bg_control >> 8) & 0x1F) * 0x800
            size = (bg_control >> 14) & 3
            width_tiles = 64 if size in (1, 3) else 32
            height_tiles = 64 if size in (2, 3) else 32
            width_pixels = width_tiles * 8
            height_pixels = height_tiles * 8
            hofs = self.bus.read16(0x04000010 + bg * 4) & 0x1FF
            vofs = self.bus.read16(0x04000012 + bg * 4) & 0x1FF
            tile_bytes = 64 if color256 else 32
            palette = [bytes(self._palette_color(index)) for index in range(256)]
            tile_rows: dict[tuple[int, int], tuple[int, ...]] = {}

            for screen_y in range(SCREEN_HEIGHT):
                bg_y = (screen_y + vofs) % height_pixels
                tile_y, pixel_y = divmod(bg_y, 8)
                screen_x = 0
                while screen_x < SCREEN_WIDTH:
                    bg_x = (screen_x + hofs) % width_pixels
                    tile_x, pixel_x = divmod(bg_x, 8)
                    segment = min(8 - pixel_x, SCREEN_WIDTH - screen_x, width_pixels - bg_x)
                    screen_block = (tile_y // 32) * (width_tiles // 32) + (tile_x // 32)
                    map_offset = (
                        screen_base
                        + screen_block * 0x800
                        + ((tile_y & 31) * 32 + (tile_x & 31)) * 2
                    )
                    entry = self._vram_byte(map_offset) | (self._vram_byte(map_offset + 1) << 8)
                    row_key = (entry, pixel_y)
                    tile_row = tile_rows.get(row_key)
                    if tile_row is None:
                        source_y = 7 - pixel_y if entry & (1 << 11) else pixel_y
                        tile_offset = char_base + (entry & 0x3FF) * tile_bytes
                        if color256:
                            tile_row = tuple(
                                self._vram_byte(tile_offset + source_y * 8 + column)
                                for column in range(8)
                            )
                        else:
                            palette_bank = ((entry >> 12) & 0xF) * 16
                            row_values = []
                            for column in range(8):
                                packed = self._vram_byte(
                                    tile_offset + source_y * 4 + column // 2
                                )
                                color = (packed >> (4 if column & 1 else 0)) & 0xF
                                # Color zero stays zero so it remains transparent,
                                # regardless of the tile's palette bank.
                                row_values.append(palette_bank + color if color else 0)
                            tile_row = tuple(row_values)
                        tile_rows[row_key] = tile_row

                    columns = (
                        (7 - pixel_x - step for step in range(segment))
                        if entry & (1 << 10)
                        else (pixel_x + step for step in range(segment))
                    )
                    indices = tuple(tile_row[column] for column in columns)
                    # Preserve the lower layers beneath transparent tile pixels.
                    run_start = 0
                    while run_start < segment:
                        while run_start < segment and indices[run_start] == 0:
                            run_start += 1
                        run_end = run_start
                        while run_end < segment and indices[run_end] != 0:
                            run_end += 1
                        if run_start < run_end:
                            rgb = b"".join(palette[index] for index in indices[run_start:run_end])
                            pixel_index = screen_y * SCREEN_WIDTH + screen_x + run_start
                            output = pixel_index * 3
                            pixels[output : output + len(rgb)] = rgb
                        run_start = run_end
                    screen_x += segment
        return bytes(pixels)

    @staticmethod
    def _signed28(value: int) -> int:
        value &= 0x0FFFFFFF
        return value - 0x10000000 if value & 0x08000000 else value

    def _render_affine_background(
        self, bg: int, pixels: bytearray, priorities: bytearray,
        second_pixels: bytearray, layers: bytearray, second_layers: bytearray,
        semi_objects: bytearray, window_layers: bytearray,
    ) -> None:
        """Draw one affine tiled background (BG2 or BG3) using 8.8 coordinates."""
        bg_control = self.bus.read16(0x04000008 + bg * 2)
        priority = bg_control & 3
        size = 128 << ((bg_control >> 14) & 3)
        char_base = ((bg_control >> 2) & 3) * 0x4000
        map_base = ((bg_control >> 8) & 0x1F) * 0x800
        wrap = bool(bg_control & (1 << 13))
        matrix = 0x04000020 + (bg - 2) * 0x10
        pa = self.bus.read16(matrix)
        pb = self.bus.read16(matrix + 2)
        pc = self.bus.read16(matrix + 4)
        pd = self.bus.read16(matrix + 6)
        pa = pa - 0x10000 if pa & 0x8000 else pa
        pb = pb - 0x10000 if pb & 0x8000 else pb
        pc = pc - 0x10000 if pc & 0x8000 else pc
        pd = pd - 0x10000 if pd & 0x8000 else pd
        ref_x = self._signed28(self.bus.read32(matrix + 8))
        ref_y = self._signed28(self.bus.read32(matrix + 12))
        tiles_per_side = size // 8
        mosaic = self.bus.read16(0x0400004C)
        mosaic_enabled = bool(bg_control & (1 << 6))
        mosaic_h = (mosaic & 0xF) + 1
        mosaic_v = ((mosaic >> 4) & 0xF) + 1
        palette = [self._palette_color(index) for index in range(256)]
        tile_map: dict[tuple[int, int], int] = {}
        tile_rows: dict[tuple[int, int], tuple[int, ...]] = {}

        for screen_y in range(SCREEN_HEIGHT):
            source_y = ref_y + pd * screen_y
            source_x = ref_x + pb * screen_y
            for screen_x in range(SCREEN_WIDTH):
                if mosaic_enabled:
                    sample_x = (screen_x // mosaic_h) * mosaic_h
                    sample_y = (screen_y // mosaic_v) * mosaic_v
                    x = (ref_x + pa * sample_x + pb * sample_y) >> 8
                    y = (ref_y + pc * sample_x + pd * sample_y) >> 8
                else:
                    x = source_x >> 8
                    y = source_y >> 8
                if wrap:
                    x %= size
                    y %= size
                    inside = True
                else:
                    inside = 0 <= x < size and 0 <= y < size
                if inside:
                    tile_x, pixel_x = divmod(x, 8)
                    tile_y, pixel_y = divmod(y, 8)
                    map_key = (tile_x, tile_y)
                    tile_number = tile_map.get(map_key)
                    if tile_number is None:
                        map_offset = map_base + tile_y * tiles_per_side + tile_x
                        tile_number = self._vram_byte(map_offset)
                        tile_map[map_key] = tile_number
                    row_key = (tile_number, pixel_y)
                    tile_row = tile_rows.get(row_key)
                    if tile_row is None:
                        tile_offset = char_base + tile_number * 64 + pixel_y * 8
                        tile_row = tuple(self._vram_byte(tile_offset + column) for column in range(8))
                        tile_rows[row_key] = tile_row
                    color_index = tile_row[pixel_x]
                    if color_index:
                        pixel_index = screen_y * SCREEN_WIDTH + screen_x
                        # Text BG0/BG1 win ties in mode 1; BG2 wins the BG2/BG3
                        # tie in mode 2.
                        tie_wins = priority == priorities[pixel_index] and bg == 2 and layers[pixel_index] == 3
                        if priority < priorities[pixel_index] or tie_wins:
                            self._plot_pixel(
                                pixel_index, palette[color_index], bg, priority, pixels, second_pixels,
                                layers, second_layers, priorities, semi_objects, window_layers,
                            )
                source_x += pa
                source_y += pc

    def _render_objects(
        self, control: int, pixels: bytearray, bg_priorities: bytearray,
        second_pixels: bytearray, layers: bytearray, second_layers: bytearray,
        semi_objects: bytearray, window_layers: bytearray,
    ) -> None:
        """Draw non-affine OBJ tiles over background pixels they outrank."""
        if not control & OBJ_ENABLE:
            return

        shape_sizes = (
            ((8, 8), (16, 16), (32, 32), (64, 64)),
            ((16, 8), (32, 8), (32, 16), (64, 32)),
            ((8, 16), (8, 32), (16, 32), (32, 64)),
        )
        mapping_1d = bool(control & (1 << 6))
        objects = []
        mosaic = self.bus.read16(0x0400004C)
        obj_mosaic_h = ((mosaic >> 8) & 0xF) + 1
        obj_mosaic_v = ((mosaic >> 12) & 0xF) + 1
        for index in range(128):
            offset = index * 8
            attr0 = self.bus.oam[offset] | (self.bus.oam[offset + 1] << 8)
            attr1 = self.bus.oam[offset + 2] | (self.bus.oam[offset + 3] << 8)
            attr2 = self.bus.oam[offset + 4] | (self.bus.oam[offset + 5] << 8)
            affine = bool(attr0 & (1 << 8))
            double_size = affine and bool(attr0 & (1 << 9))
            if not affine and (attr0 & (1 << 9)):
                continue
            object_mode = (attr0 >> 10) & 0x3
            shape = (attr0 >> 14) & 0x3
            size = (attr1 >> 14) & 0x3
            if object_mode >= 2 or shape == 3:
                continue
            width, height = shape_sizes[shape][size]
            x = attr1 & 0x1FF
            y = attr0 & 0xFF
            if x >= 256:
                x -= 512
            if y >= 160:
                y -= 256
            priority = (attr2 >> 10) & 0x3
            draw_width = width * (2 if double_size else 1)
            draw_height = height * (2 if double_size else 1)
            objects.append((priority, index, x, y, width, height, draw_width, draw_height, affine, attr0, attr1, attr2))

        # Paint far layers first. Lower priority number and lower OAM index win.
        objects.sort(key=lambda item: (item[0], item[1]), reverse=True)
        bitmap_mode = (control & 0x7) >= 3
        object_palette: list[tuple[int, int, int]] | None = None
        tile_rows: dict[tuple[int, int, bool], tuple[int, ...]] = {}
        for priority, _index, x, y, width, height, draw_width, draw_height, affine, attr0, attr1, attr2 in objects:
            color256 = bool(attr0 & (1 << 13))
            hflip = not affine and bool(attr1 & (1 << 12))
            vflip = not affine and bool(attr1 & (1 << 13))
            tile_number = attr2 & 0x3FF
            matrix = None
            if affine:
                matrix_index = (attr1 >> 9) & 0x1F
                matrix_address = 0x07000006 + matrix_index * 32
                matrix = tuple(
                    self.bus.read16(matrix_address + offset) for offset in (0, 8, 16, 24)
                )
                matrix = tuple(value - 0x10000 if value & 0x8000 else value for value in matrix)
            if bitmap_mode and tile_number < 512:
                continue
            tile_unit = 2 if color256 else 1
            if color256:
                tile_number &= ~1
            palette_bank = (attr2 >> 12) & 0xF
            width_tiles = width // 8
            for local_y in range(draw_height):
                screen_y = y + local_y
                if not 0 <= screen_y < SCREEN_HEIGHT:
                    continue
                for local_x in range(draw_width):
                    screen_x = x + local_x
                    if not 0 <= screen_x < SCREEN_WIDTH:
                        continue
                    pixel_index = screen_y * SCREEN_WIDTH + screen_x
                    if priority > bg_priorities[pixel_index]:
                        continue
                    if affine:
                        pa, pb, pc, pd = matrix
                        sample_x = local_x
                        sample_y = local_y
                        if attr0 & (1 << 12):
                            sample_x = max(0, ((screen_x // obj_mosaic_h) * obj_mosaic_h) - x)
                            sample_y = max(0, ((screen_y // obj_mosaic_v) * obj_mosaic_v) - y)
                        dx = sample_x - draw_width // 2
                        dy = sample_y - draw_height // 2
                        source_x = (pa * dx + pb * dy + (width << 7)) >> 8
                        source_y = (pc * dx + pd * dy + (height << 7)) >> 8
                        if not (0 <= source_x < width and 0 <= source_y < height):
                            continue
                    else:
                        sample_x = local_x
                        sample_y = local_y
                        if attr0 & (1 << 12):
                            sample_x = max(0, ((screen_x // obj_mosaic_h) * obj_mosaic_h) - x)
                            sample_y = max(0, ((screen_y // obj_mosaic_v) * obj_mosaic_v) - y)
                        source_x = width - 1 - sample_x if hflip else sample_x
                        source_y = height - 1 - sample_y if vflip else sample_y
                    tile_x, pixel_x = divmod(source_x, 8)
                    tile_y, pixel_y = divmod(source_y, 8)
                    if mapping_1d:
                        tile_offset_index = tile_y * width_tiles * tile_unit + tile_x * tile_unit
                    else:
                        tile_offset_index = tile_y * 32 + tile_x * tile_unit
                    tile_address = OBJ_VRAM_BASE + (tile_number + tile_offset_index) * 32
                    row_key = (tile_address, pixel_y, color256)
                    tile_row = tile_rows.get(row_key)
                    if tile_row is None:
                        if color256:
                            tile_row = tuple(
                                self._vram_byte(tile_address + pixel_y * 8 + column)
                                for column in range(8)
                            )
                        else:
                            row = []
                            for column in range(8):
                                packed = self._vram_byte(
                                    tile_address + pixel_y * 4 + column // 2
                                )
                                row.append((packed >> (4 if column & 1 else 0)) & 0xF)
                            tile_row = tuple(row)
                        tile_rows[row_key] = tile_row
                    color_index = tile_row[pixel_x]
                    if color_index == 0:
                        continue
                    palette_index = color_index if color256 else palette_bank * 16 + color_index
                    if object_palette is None:
                        object_palette = [self._palette_color(i, object_palette=True) for i in range(256)]
                    self._plot_pixel(
                        pixel_index, object_palette[palette_index], 4, priority, pixels,
                        second_pixels, layers, second_layers, bg_priorities,
                        semi_objects, window_layers, semi_object=(object_mode == 1),
                    )

    def _apply_color_effects(
        self,
        pixels: bytearray,
        second_pixels: bytearray,
        layers: bytearray,
        second_layers: bytearray,
        semi_objects: bytearray,
        window_effects: bytearray,
    ) -> None:
        """Apply global GBA alpha blend and brightness effects to visible pixels."""
        effects = self.bus.read16(0x04000050)
        effect_mode = (effects >> 6) & 0x3
        if effect_mode == 0 and not any(semi_objects):
            return
        alpha = self.bus.read16(0x04000052)
        eva = min(alpha & 0x1F, 16)
        evb = min((alpha >> 8) & 0x1F, 16)
        brightness = min(self.bus.read16(0x04000054) & 0x1F, 16)

        for pixel_index, layer in enumerate(layers):
            if not window_effects[pixel_index]:
                continue
            first_target = bool(effects & (1 << layer))
            second_layer = second_layers[pixel_index]
            second_target = bool(effects & (1 << (8 + second_layer)))
            semi_obj = layer == 4 and bool(semi_objects[pixel_index])
            do_alpha = second_target and (semi_obj or (effect_mode == 1 and first_target))
            output = pixel_index * 3
            if do_alpha:
                for channel in range(3):
                    top5 = pixels[output + channel] >> 3
                    bottom5 = second_pixels[output + channel] >> 3
                    value = min(31, (top5 * eva + bottom5 * evb) >> 4)
                    pixels[output + channel] = (value << 3) | (value >> 2)
            elif first_target and effect_mode in (2, 3):
                for channel in range(3):
                    value = pixels[output + channel] >> 3
                    if effect_mode == 2:
                        value += ((31 - value) * brightness) >> 4
                    else:
                        value -= (value * brightness) >> 4
                    pixels[output + channel] = (value << 3) | (value >> 2)

    def frame_rgb(self) -> bytes:
        """Return one 240x160 RGB888 image of the currently selected bitmap mode."""
        generation = self.bus.video_generation
        if generation == self._cached_generation:
            return self._cached_frame
        frame = self._render_frame_rgb()
        self._cached_frame = frame
        self._cached_generation = generation
        return frame

    def _render_frame_rgb(self) -> bytes:
        """Build a new frame after video memory or display state changes."""
        control = self.bus.read16(DISPCNT)
        mode = control & 0x7
        pixel_count = SCREEN_WIDTH * SCREEN_HEIGHT
        if control & (1 << 7):
            # DISPCNT forced blank displays solid white, regardless of layers.
            return bytes([0xFF]) * (pixel_count * 3)
        if mode == 0 and _render_native_mode0 is not None:
            native_frame = _render_native_mode0(self.bus)
            if native_frame is not None:
                return native_frame
        simple_bitmap_scene = (
            bool(control & BG2_ENABLE)
            and not control & (OBJ_ENABLE | 0xE000)
            and not (self.bus.read16(0x0400000C) & (1 << 6))
            # The bitmap fast paths skip layer bookkeeping and color effects.
            # Fall back whenever BLDCNT enables alpha or brightness effects.
            and not (self.bus.read16(0x04000050) & 0x00C0)
        )
        if mode == 0:
            simple_mode0 = self._render_simple_mode0(control)
            if simple_mode0 is not None:
                return simple_mode0
        if (
            mode == 3
            and simple_bitmap_scene
        ):
            # Typical Mode 3 scenes need only direct-color conversion. Skip
            # allocating and sorting per-pixel layer/window/effect buffers.
            color_words = memoryview(self.bus.vram)[: pixel_count * 2].cast("H")
            packed_colors = "".join(chr(color & 0x7FFF) for color in color_words)
            return packed_colors.translate(_RGB555_TRANSLATION).encode("latin1")
        if mode == 4 and simple_bitmap_scene:
            page = 0xA000 if control & FRAME_SELECT else 0
            indices = bytes(self.bus.vram[page : page + pixel_count]).decode("latin1")
            palette = {
                index: "".join(chr(component) for component in self._palette_color(index))
                for index in range(256)
            }
            return indices.translate(palette).encode("latin1")
        if mode == 5 and simple_bitmap_scene:
            page = 0xA000 if control & FRAME_SELECT else 0
            image_pixels = memoryview(self.bus.vram)[page : page + 160 * 128 * 2].cast("H")
            packed_colors = "".join(chr(color & 0x7FFF) for color in image_pixels)
            image = packed_colors.translate(_RGB555_TRANSLATION).encode("latin1")
            backdrop = bytes(self._palette_color(0))
            pixels = bytearray(backdrop * pixel_count)
            for y in range(128):
                source_start = y * 160 * 3
                destination_start = ((y + 16) * SCREEN_WIDTH + 40) * 3
                pixels[destination_start : destination_start + 160 * 3] = image[
                    source_start : source_start + 160 * 3
                ]
            return bytes(pixels)
        backdrop = bytes(self._palette_color(0))
        pixels = bytearray(backdrop * pixel_count)
        second_pixels = bytearray(pixels)
        layers = bytearray([5]) * pixel_count  # Backdrop is layer 5.
        second_layers = bytearray([5]) * pixel_count
        semi_objects = bytearray(pixel_count)
        bg_priorities = bytearray([4]) * pixel_count
        object_window_pixels = self._object_window_pixels(control)
        window_layers, window_effects = self._window_masks(control, object_window_pixels)

        if mode == 0:
            self._render_mode0(
                control, pixels, bg_priorities, second_pixels, layers,
                second_layers, semi_objects, window_layers,
            )

        elif mode in (1, 2):
            # Text and affine layers participate in the same priority sort.
            # Drawing every text BG first and every affine BG afterward made
            # an affine layer cover text even when its numeric priority was
            # lower. GBA draws low priority values on top; lower BG numbers
            # win ties, so render from the back toward the front.
            candidates = range(3) if mode == 1 else range(2, 4)
            backgrounds = []
            for bg in candidates:
                if control & (1 << (8 + bg)):
                    bg_control = self.bus.read16(0x04000008 + bg * 2)
                    backgrounds.append((bg_control & 3, bg, bg_control))
            backgrounds.sort(key=lambda layer: (layer[0], layer[1]), reverse=True)
            for _priority, bg, _bg_control in backgrounds:
                if mode == 1 and bg in (0, 1):
                    text_control = control & ~0x0F00
                    text_control |= 1 << (8 + bg)
                    self._render_mode0(
                        text_control, pixels, bg_priorities, second_pixels,
                        layers, second_layers, semi_objects, window_layers,
                    )
                else:
                    self._render_affine_background(
                        bg, pixels, bg_priorities, second_pixels, layers,
                        second_layers, semi_objects, window_layers,
                    )

        elif mode == 3 and control & BG2_ENABLE:
            # Mode 3 is one 240x160 image of direct 15-bit color pixels.
            source = self.bus.vram
            bg_control = self.bus.read16(0x0400000C)
            bg_priority = bg_control & 0x3
            mosaic = self.bus.read16(0x0400004C)
            mosaic_enabled = bool(bg_control & (1 << 6))
            mosaic_h = (mosaic & 0xF) + 1
            mosaic_v = ((mosaic >> 4) & 0xF) + 1
            for y in range(SCREEN_HEIGHT):
                for x in range(SCREEN_WIDTH):
                    index = y * SCREEN_WIDTH + x
                    if not (window_layers[index] & (1 << 2)):
                        continue
                    sample_x = (x // mosaic_h) * mosaic_h if mosaic_enabled else x
                    sample_y = (y // mosaic_v) * mosaic_v if mosaic_enabled else y
                    sample_index = sample_y * SCREEN_WIDTH + sample_x
                    color = source[sample_index * 2] | (source[sample_index * 2 + 1] << 8)
                    output = index * 3
                    pixels[output : output + 3] = bytes(self._rgb555(color))
                    layers[index] = 2
                    bg_priorities[index] = bg_priority

        elif mode == 4 and control & BG2_ENABLE:
            # Mode 4 stores one 8-bit palette index per pixel and can page-flip.
            page = 0xA000 if control & FRAME_SELECT else 0
            source = self.bus.vram
            bg_control = self.bus.read16(0x0400000C)
            bg_priority = bg_control & 0x3
            mosaic = self.bus.read16(0x0400004C)
            mosaic_enabled = bool(bg_control & (1 << 6))
            mosaic_h = (mosaic & 0xF) + 1
            mosaic_v = ((mosaic >> 4) & 0xF) + 1
            for y in range(SCREEN_HEIGHT):
                for x in range(SCREEN_WIDTH):
                    index = y * SCREEN_WIDTH + x
                    if not (window_layers[index] & (1 << 2)):
                        continue
                    sample_x = (x // mosaic_h) * mosaic_h if mosaic_enabled else x
                    sample_y = (y // mosaic_v) * mosaic_v if mosaic_enabled else y
                    sample_index = sample_y * SCREEN_WIDTH + sample_x
                    output = index * 3
                    pixels[output : output + 3] = bytes(self._palette_color(source[page + sample_index]))
                    layers[index] = 2
                    bg_priorities[index] = bg_priority

        elif mode == 5 and control & BG2_ENABLE:
            # Mode 5 is 160x128 direct color; center it in the 240x160 LCD image.
            page = 0xA000 if control & FRAME_SELECT else 0
            source = self.bus.vram
            image_width = 160
            image_height = 128
            left = (SCREEN_WIDTH - image_width) // 2
            top = (SCREEN_HEIGHT - image_height) // 2
            bg_control = self.bus.read16(0x0400000C)
            bg_priority = bg_control & 0x3
            mosaic = self.bus.read16(0x0400004C)
            mosaic_enabled = bool(bg_control & (1 << 6))
            mosaic_h = (mosaic & 0xF) + 1
            mosaic_v = ((mosaic >> 4) & 0xF) + 1
            for screen_y in range(SCREEN_HEIGHT):
                for screen_x in range(SCREEN_WIDTH):
                    sample_x = (screen_x // mosaic_h) * mosaic_h if mosaic_enabled else screen_x
                    sample_y = (screen_y // mosaic_v) * mosaic_v if mosaic_enabled else screen_y
                    if not (left <= sample_x < left + image_width and top <= sample_y < top + image_height):
                        continue
                    pixel_index = screen_y * SCREEN_WIDTH + screen_x
                    if not (window_layers[pixel_index] & (1 << 2)):
                        continue
                    source_x = sample_x - left
                    source_y = sample_y - top
                    source_index = page + (source_y * image_width + source_x) * 2
                    color = source[source_index] | (source[source_index + 1] << 8)
                    output = pixel_index * 3
                    pixels[output : output + 3] = bytes(self._rgb555(color))
                    layers[pixel_index] = 2
                    bg_priorities[pixel_index] = bg_priority

        if control & OBJ_ENABLE:
            self._render_objects(
                control, pixels, bg_priorities, second_pixels, layers,
                second_layers, semi_objects, window_layers,
            )
        self._apply_color_effects(
            pixels, second_pixels, layers, second_layers, semi_objects, window_effects,
        )
        return bytes(pixels)

    def to_ppm(self) -> bytes:
        """Encode the current frame as a PPM image Tk can display directly."""
        header = f"P6\n{SCREEN_WIDTH} {SCREEN_HEIGHT}\n255\n".encode("ascii")
        return header + self.frame_rgb()
