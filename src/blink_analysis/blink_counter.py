# src/blink_analysis/blink_counter.py

import time


class BlinkCounter:
    def __init__(self, ear_threshold=0.25, min_frames_closed=2):
        self.ear_threshold = ear_threshold
        self.min_frames_closed = min_frames_closed

        self.blink_count = 0
        self.frame_counter = 0
        self.blink_start_time = None
        self.blink_durations = []

    def update(self, ear):
        """
        Update blink state using EAR value
        """

        blink_detected = False

        if ear < self.ear_threshold:
            self.frame_counter += 1
            if self.blink_start_time is None:
                self.blink_start_time = time.time()
        else:
            if self.frame_counter >= self.min_frames_closed:
                self.blink_count += 1
                blink_detected = True

                duration = time.time() - self.blink_start_time
                self.blink_durations.append(duration)

            self.frame_counter = 0
            self.blink_start_time = None

        return blink_detected

    def reset(self):
        self.blink_count = 0
        self.frame_counter = 0
        self.blink_durations.clear()
