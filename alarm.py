"""Audible alarm. The stream stays open; we only switch the tone on and off."""
import numpy as np
import sounddevice as sd


class Alarm:
    """Intermittent beep while active is True.

    The output stream is open at all times and plays silence when idle.
    That way the sound starts instantly: a bluetooth speaker does not have to
    wake up, and the audio system does not have to reopen the device.
    """

    def __init__(self, freq=880, beep=0.25, gap=0.25, volume=0.25, samplerate=44100):
        n = int(beep * samplerate)
        t = np.arange(n) / samplerate

        # smooth edges, otherwise there is a click on each start and stop
        envelope = np.clip(np.minimum(t, beep - t) * 60, 0, 1)
        tone = volume * np.sin(2 * np.pi * freq * t) * envelope

        self.pattern = np.concatenate([tone, np.zeros(int(gap * samplerate))]).astype(np.float32)
        self.pos = 0
        self.active = False

        self.stream = sd.OutputStream(samplerate=samplerate, channels=1, callback=self._fill)
        self.stream.start()

    def _fill(self, outdata, frames, time_info, status):
        """Called by the sound card whenever it needs the next chunk of samples."""
        if not self.active:
            outdata.fill(0)
            self.pos = 0
            return
        idx = (self.pos + np.arange(frames)) % len(self.pattern)
        outdata[:, 0] = self.pattern[idx]
        self.pos = (self.pos + frames) % len(self.pattern)

    def set(self, on):
        self.active = bool(on)

    def close(self):
        self.active = False
        self.stream.stop()
        self.stream.close()
