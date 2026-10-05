import numpy as np
import sounddevice as sd

SAMPLERATE = 44100


class Alarm:

    def __init__(self, freq=880, beep=0.25, gap=0.25, volume=0.25):
        t = np.arange(int(beep * SAMPLERATE)) / SAMPLERATE
        tone = volume * np.sin(2 * np.pi * freq * t)

        ramp = np.linspace(0, 1, int(0.01 * SAMPLERATE))
        tone[:len(ramp)] *= ramp
        tone[-len(ramp):] *= ramp[::-1]

        silence = np.zeros(int(gap * SAMPLERATE))
        self.pattern = np.concatenate([tone, silence])
        self.active = False

    def set(self, on):
        on = bool(on)
        if on == self.active:
            return
        self.active = on
        if on:
            sd.play(self.pattern, SAMPLERATE, loop=True)
        else:
            sd.stop()

    def close(self):
        self.active = False
        sd.stop()
