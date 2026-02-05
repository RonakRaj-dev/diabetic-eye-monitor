import numpy as np
import time


class BlinkMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.blink_timestamps = []
        self.blink_durations = []

    def record_blink(self, duration):
        self.blink_timestamps.append(time.time())
        self.blink_durations.append(duration)

    def get_blink_rate(self):
        elapsed_minutes = (time.time() - self.start_time) / 60.0
        if elapsed_minutes == 0:
            return 0.0
        return len(self.blink_timestamps) / elapsed_minutes

    def average_blink_duration(self):
        if not self.blink_durations:
            return 0.0
        return float(np.mean(self.blink_durations))

    def fatigue_score(self):
        """
        Simple fatigue heuristic:
        longer blinks + higher frequency → higher fatigue
        """
        rate = self.get_blink_rate()
        avg_duration = self.average_blink_duration()

        score = (0.6 * min(rate / 30.0, 1.0)) + (0.4 * min(avg_duration / 0.4, 1.0))
        return round(score, 2)

    def summary(self):
        return {
            "blink_rate_per_min": round(self.get_blink_rate(), 2),
            "avg_blink_duration_sec": round(self.average_blink_duration(), 3),
            "fatigue_score": self.fatigue_score()
        }
