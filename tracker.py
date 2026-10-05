import time


class ClosureTracker:

    def __init__(self, smoothing=0.4, threshold=0.5, alarm_after=2.0):
        self.smoothing = smoothing
        self.threshold = threshold
        self.alarm_after = alarm_after

        self.value = 0.0
        self.closed_since = None
        self.lost_since = None

    def update(self, prob, now=None):
        now = time.time() if now is None else now
        self.lost_since = None

        self.value += self.smoothing * (prob - self.value)
        closed = self.value > self.threshold

        if closed:
            if self.closed_since is None:
                self.closed_since = now
        else:
            self.closed_since = None

        return self.state(now)

    def miss(self, now=None):
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
            "lost": 0.0 if self.lost_since is None else now - self.lost_since,
        }
