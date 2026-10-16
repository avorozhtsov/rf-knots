"""Wall-clock checkpoint cadence, independent from global step divisibility."""
import math


class CheckpointPolicy:
    def __init__(self, now, interval=3600):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError('positive finite checkpoint interval required')
        self.interval = interval
        self.last = now

    def due(self, now, pending_steps, stopping=False):
        return pending_steps > 0 and (stopping or now - self.last >= self.interval)

    def committed(self, now):
        self.last = now
