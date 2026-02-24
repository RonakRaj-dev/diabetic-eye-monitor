# src/blink_analysis/blink_metrics.py

import numpy as np
import time


class BlinkMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset metrics for a new blink session"""
        self.blink_durations = []
        self.blink_count = 0
        self.start_time = time.time()

    def record_blink(self, duration):
        self.blink_durations.append(duration)
        self.blink_count += 1

    def summary(self):
        elapsed = max(time.time() - self.start_time, 1e-6)

        blink_rate = (self.blink_count / elapsed) * 60
        avg_duration = (
            np.mean(self.blink_durations)
            if self.blink_durations else 0.0
        )

        fatigue_score = min(
            1.0,
            0.6 * (blink_rate / 20) + 0.4 * (avg_duration / 0.3)
        )

        return {
            "blink_rate_per_min": round(blink_rate, 2),
            "avg_blink_duration_sec": round(avg_duration, 3),
            "fatigue_score": round(float(fatigue_score), 3)
        }
