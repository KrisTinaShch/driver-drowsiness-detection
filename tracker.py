"""Logic on top of the model: smoothing, closure timer, drowsiness measure."""
import time
from collections import deque


class ClosureTracker:
    """Turns a stream of per-frame probabilities into a meaningful state.

    smoothing   weight of the new frame when smoothing (0..1). Lower is calmer.
    threshold   boundary for "closed".
    alarm_after seconds of uninterrupted closure before the alarm.
    window      window for PERCLOS, in seconds.
    """

    def __init__(self, smoothing=0.4, threshold=0.5, alarm_after=2.0, window=60.0):
        self.smoothing = smoothing
        self.threshold = threshold
        self.alarm_after = alarm_after
        self.window = window

        self.value = 0.0          # smoothed probability of "closed"
        self.closed_since = None  # when the current closure started
        self.lost_since = None    # when the face went missing
        self.history = deque()    # (time, is_closed) over the last window

    def update(self, prob, now=None):
        """Add a frame. prob is the probability of "closed" across both eyes."""
        now = time.time() if now is None else now
        self.lost_since = None

        # exponential smoothing: the new value is blended into the old one
        self.value += self.smoothing * (prob - self.value)
        closed = self.value > self.threshold

        if closed:
            if self.closed_since is None:
                self.closed_since = now
        else:
            self.closed_since = None

        self.history.append((now, closed))
        while self.history and now - self.history[0][0] > self.window:
            self.history.popleft()

        return self.state(now)

    def miss(self, now=None):
        """Face not found. The closure timer is NOT reset: when a person falls
        asleep the head drops and the face leaves the frame at the exact moment
        the alarm matters most."""
        now = time.time() if now is None else now
        if self.lost_since is None:
            self.lost_since = now
        return self.state(now)

    def state(self, now=None):
        now = time.time() if now is None else now
        duration = 0.0 if self.closed_since is None else now - self.closed_since
        return {
            "prob": self.value,
            "closed": self.closed_since is not None,
            "duration": duration,
            "alarm": duration >= self.alarm_after,
            "perclos": self.perclos(),
            "lost": 0.0 if self.lost_since is None else now - self.lost_since,
        }

    def perclos(self):
        """Share of time with eyes closed over the window.

        PERCLOS is the standard measure used in driver drowsiness research.
        """
        if not self.history:
            return 0.0
        return sum(c for _, c in self.history) / len(self.history)

    def seen(self):
        """Seconds of observation collected: PERCLOS is unreliable until the
        window fills up."""
        if not self.history:
            return 0.0
        return self.history[-1][0] - self.history[0][0]
