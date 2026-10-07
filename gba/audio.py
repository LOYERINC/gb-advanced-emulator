"""Small optional Windows PCM output for the emulator's generated sound mix."""

from __future__ import annotations

from io import BytesIO
from queue import Empty, Full, Queue
import struct
from threading import Thread
import wave

try:
    import winsound
except ImportError:  # pragma: no cover - non-Windows desktop fallback
    winsound = None


class AudioOutput:
    """Queue stereo samples for asynchronous playback without blocking Tk."""

    SAMPLE_RATE = 22050
    CHUNK_SAMPLES = 512

    def __init__(self) -> None:
        self._queue: Queue[bytes | None] = Queue(maxsize=8)
        self._pending: list[tuple[int, int]] = []
        self._resample_phase = 0.0
        self.available = winsound is not None
        self._worker: Thread | None = None
        if self.available:
            self._worker = Thread(target=self._play_worker, daemon=True, name="gba-audio")
            self._worker.start()

    def submit(self, samples: list[tuple[tuple[int, int], int]]) -> None:
        if not self.available:
            return
        for (left, right), input_rate in samples:
            self._resample_phase += self.SAMPLE_RATE / max(1, input_rate)
            while self._resample_phase >= 1.0:
                self._pending.append((
                    max(-32768, min(32767, left)),
                    max(-32768, min(32767, right)),
                ))
                self._resample_phase -= 1.0
        while len(self._pending) >= self.CHUNK_SAMPLES:
            self._enqueue(self._pending[: self.CHUNK_SAMPLES])
            del self._pending[: self.CHUNK_SAMPLES]

    def reset(self) -> None:
        self._pending.clear()
        self._resample_phase = 0.0
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break
        if winsound is not None:
            winsound.PlaySound(None, winsound.SND_PURGE)

    def close(self) -> None:
        if not self.available:
            return
        if self._pending:
            self._enqueue(self._pending)
            self._pending.clear()
        try:
            self._queue.put_nowait(None)
        except Full:
            pass

    def _enqueue(self, samples: list[tuple[int, int]]) -> None:
        stream = BytesIO()
        with wave.open(stream, "wb") as output:
            output.setnchannels(2)
            output.setsampwidth(2)
            output.setframerate(self.SAMPLE_RATE)
            interleaved = [value for frame in samples for value in frame]
            output.writeframes(struct.pack(f"<{len(interleaved)}h", *interleaved))
        try:
            self._queue.put_nowait(stream.getvalue())
        except Full:
            # Keep emulation responsive if the host audio device falls behind.
            pass

    def _play_worker(self) -> None:
        assert winsound is not None
        while True:
            chunk = self._queue.get()
            if chunk is None:
                return
            try:
                winsound.PlaySound(chunk, winsound.SND_MEMORY)
            except RuntimeError:
                # An unavailable/disabled audio device should not stop the CPU.
                continue
