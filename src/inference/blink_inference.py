# src/inference/blink_inference.py

import time
from src.preprocessing.eye_cropper import EyeCropper
from src.blink_analysis.ear_calculation import EARCalculator
from src.blink_analysis.blink_counter import BlinkCounter
from src.blink_analysis.blink_metrics import BlinkMetrics
from src.preprocessing.face_alignment import FaceAligner
import numpy as np


class BlinkInference:
    def __init__(self):
        self.eye_cropper = EyeCropper()
        self.ear_calculator = EARCalculator()
        self.blink_counter = BlinkCounter()
        self.blink_metrics = BlinkMetrics()
        self.face_aligner = FaceAligner()

    def infer(self, frame):
        landmarks = self.face_aligner.detect_landmarks(frame)
        if landmarks is None:
            return None

        points = self.face_aligner.landmarks_to_array(landmarks, frame.shape)

        # MediaPipe EAR landmark indices (6 points per eye)
        left_eye_idx = [33, 160, 158, 133, 153, 144]
        right_eye_idx = [362, 385, 387, 263, 373, 380]

        left_eye = np.array([points[i] for i in left_eye_idx])
        right_eye = np.array([points[i] for i in right_eye_idx])

        left_ear = self.ear_calculator.compute_ear(left_eye)
        right_ear = self.ear_calculator.compute_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0

        blinked = self.blink_counter.update(ear)

        if blinked:
            duration = self.blink_counter.blink_durations[-1]
            self.blink_metrics.record_blink(duration)

        return self.blink_metrics.summary()
