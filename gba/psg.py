"""First-pass emulation of the GBA's four Game Boy-compatible PSG channels."""

from __future__ import annotations


class GbaPsg:
    """Pulse, wave, and noise generators mixed from the sound IO registers."""

    SAMPLE_RATE = 32768
    DUTY = (0.125, 0.25, 0.5, 0.75)

    def __init__(self, io: bytearray) -> None:
        self.io = io
        self.wave_ram = bytearray(32)
        self.enabled = [False] * 4
        self.length = [0] * 4
        self.envelope_volume = [0] * 4
        self.envelope_timer = [0] * 4
        self.phase = [0.0] * 4
        self.wave_position = 0.0
        self.noise_lfsr = 0x7FFF
        self.frame_step = 0
        self.frame_sample_count = 0
        self.sweep_shadow = 0
        self.sweep_timer = 0
        self.sweep_enabled = False

    def write_register(self, offset: int, value: int) -> None:
        triggers = {0x65: 0, 0x6D: 1, 0x75: 2, 0x7D: 3}
        channel = triggers.get(offset)
        if channel is not None and value & 0x80:
            self.trigger(channel)
            self.io[offset] &= 0x7F  # Trigger is write-only.

    def read_wave_ram(self, offset: int) -> int:
        return self.wave_ram[offset & 0x1F]

    def write_wave_ram(self, offset: int, value: int) -> None:
        self.wave_ram[offset & 0x1F] = value & 0xFF

    def master_disabled(self) -> None:
        self.enabled = [False] * 4
        self.io[0x84] &= 0xF0
        self.frame_step = 0
        self.frame_sample_count = 0

    def trigger(self, channel: int) -> None:
        if not (self.io[0x84] & 0x80):
            return
        envelope_offsets = (0x63, 0x69, None, 0x79)
        if channel == 2:
            dac_enabled = bool(self.io[0x70] & 0x80)
            self.length[channel] = self.length[channel] or 256 - self.io[0x72]
        else:
            envelope = self.io[envelope_offsets[channel]]
            dac_enabled = bool(envelope & 0xF8)
            length_register = self.io[(0x62, 0x68, 0, 0x78)[channel]]
            self.length[channel] = self.length[channel] or 64 - (length_register & 0x3F)
            self.envelope_volume[channel] = envelope >> 4
            period = envelope & 0x07
            self.envelope_timer[channel] = period or 8

        self.enabled[channel] = dac_enabled
        self.phase[channel] = 0.0
        if channel == 2:
            self.wave_position = 0.0
        if channel == 3:
            self.noise_lfsr = 0x40 if self.io[0x7C] & 0x08 else 0x4000
        if channel == 0:
            self.sweep_shadow = self._frequency(0)
            sweep_period = (self.io[0x60] >> 4) & 0x07
            self.sweep_timer = sweep_period or 8
            self.sweep_enabled = bool(sweep_period or (self.io[0x60] & 0x07))
            if self.io[0x60] & 0x07:
                self._calculate_sweep(update=False)

        if self.enabled[channel]:
            self.io[0x84] |= 1 << channel
        else:
            self.io[0x84] &= ~(1 << channel)

    def sample(self) -> tuple[int, int]:
        """Return one signed (left, right) sample at the GBA output rate."""
        if not (self.io[0x84] & 0x80):
            return 0, 0
        self.frame_sample_count += 1
        if self.frame_sample_count >= 64:
            self.frame_sample_count = 0
            self.clock_frame_sequencer()
        channels = [self._pulse_sample(0), self._pulse_sample(1), self._wave_sample(), self._noise_sample()]
        control = self.io[0x80] | (self.io[0x81] << 8)
        ratio_code = self.io[0x82] & 0x03
        ratio = (0.25, 0.5, 1.0, 1.0)[ratio_code]
        right_volume = (control & 0x07) / 7.0
        left_volume = ((control >> 4) & 0x07) / 7.0
        right = sum(sample for i, sample in enumerate(channels) if control & (1 << (8 + i)))
        left = sum(sample for i, sample in enumerate(channels) if control & (1 << (12 + i)))
        return int(left * left_volume * ratio), int(right * right_volume * ratio)

    def clock_frame_sequencer(self) -> None:
        step = self.frame_step
        if step in (0, 2, 4, 6):
            for channel in range(4):
                high_register = (0x65, 0x6D, 0x75, 0x7D)[channel]
                if self.enabled[channel] and self.io[high_register] & 0x40 and self.length[channel] > 0:
                    self.length[channel] -= 1
                    if self.length[channel] == 0:
                        self._disable(channel)
        if step in (2, 6) and self.enabled[0]:
            self._clock_sweep()
        if step == 7:
            for channel in (0, 1, 3):
                self._clock_envelope(channel)
        self.frame_step = (step + 1) & 7

    def _disable(self, channel: int) -> None:
        self.enabled[channel] = False
        self.io[0x84] &= ~(1 << channel)

    def _clock_envelope(self, channel: int) -> None:
        offset = (0x63, 0x69, 0, 0x79)[channel]
        envelope = self.io[offset]
        period = envelope & 0x07
        if not period:
            return
        self.envelope_timer[channel] -= 1
        if self.envelope_timer[channel] > 0:
            return
        self.envelope_timer[channel] = period
        increase = bool(envelope & 0x08)
        volume = self.envelope_volume[channel] + (1 if increase else -1)
        if 0 <= volume <= 15:
            self.envelope_volume[channel] = volume

    def _clock_sweep(self) -> None:
        sweep = self.io[0x60]
        period = (sweep >> 4) & 0x07
        if not period or not self.sweep_enabled:
            return
        self.sweep_timer -= 1
        if self.sweep_timer > 0:
            return
        self.sweep_timer = period
        self._calculate_sweep(update=True)

    def _calculate_sweep(self, update: bool) -> int | None:
        sweep = self.io[0x60]
        shift = sweep & 0x07
        if not shift:
            return None
        change = self.sweep_shadow >> shift
        new_frequency = self.sweep_shadow - change if sweep & 0x08 else self.sweep_shadow + change
        if new_frequency > 2047 or new_frequency < 0:
            self._disable(0)
            return None
        if update:
            self.sweep_shadow = new_frequency
            self.io[0x64] = new_frequency & 0xFF
            self.io[0x65] = (self.io[0x65] & 0xF8) | ((new_frequency >> 8) & 0x07)
            self._calculate_sweep(update=False)
        return new_frequency

    def _frequency(self, channel: int) -> int:
        low_offset, high_offset = ((0x64, 0x65), (0x6C, 0x6D), (0x74, 0x75))[channel]
        return self.io[low_offset] | ((self.io[high_offset] & 0x07) << 8)

    def _pulse_sample(self, channel: int) -> int:
        if not self.enabled[channel]:
            return 0
        duty_register = self.io[(0x62, 0x68)[channel]]
        frequency = self._frequency(channel)
        self.phase[channel] = (self.phase[channel] + 131072 / (2048 - frequency) / self.SAMPLE_RATE) % 1.0
        duty = self.DUTY[duty_register >> 6]
        volume = self.envelope_volume[channel]
        return (volume if self.phase[channel] < duty else -volume) * 128

    def _wave_sample(self) -> int:
        if not self.enabled[2]:
            return 0
        frequency = self._frequency(2)
        self.wave_position = (self.wave_position + 2097152 / (2048 - frequency) / self.SAMPLE_RATE) % 64
        dimension = bool(self.io[0x70] & 0x20)
        length = 64 if dimension else 32
        position = int(self.wave_position) % length
        bank = ((self.io[0x70] >> 6) & 1) * 16 if not dimension else 0
        packed = self.wave_ram[bank + position // 2]
        sample = (packed >> 4) if position % 2 == 0 else (packed & 0x0F)
        volume_code = (self.io[0x73] >> 5) & 0x03
        if self.io[0x73] & 0x80:
            scale = 0.75
        else:
            scale = (0.0, 1.0, 0.5, 0.25)[volume_code]
        return int((sample - 8) * scale * 128)

    def _noise_sample(self) -> int:
        if not self.enabled[3]:
            return 0
        polynomial = self.io[0x7C]
        ratio_code = polynomial & 0x07
        ratio = 0.5 if ratio_code == 0 else float(ratio_code)
        shift = (polynomial >> 4) & 0x0F
        frequency = 524288 / ratio / (1 << (shift + 1))
        self.phase[3] += frequency / self.SAMPLE_RATE
        while self.phase[3] >= 1.0:
            self.phase[3] -= 1.0
            xor = (self.noise_lfsr ^ (self.noise_lfsr >> 1)) & 1
            self.noise_lfsr = (self.noise_lfsr >> 1) | (xor << 14)
            if polynomial & 0x08:
                self.noise_lfsr = (self.noise_lfsr & ~(1 << 6)) | (xor << 6)
        volume = self.envelope_volume[3]
        return volume * 128 if self.noise_lfsr & 1 else -volume * 128
