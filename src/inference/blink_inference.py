# src/inference/blink_inference.py

import cv2
import time
import numpy as np

from src.preprocessing.eye_cropper import EyeCropper
from src.blink_analysis.ear_calculation import EARCalculator
from src.blink_analysis.blink_counter import BlinkCounter
from src.blink_analysis.blink_metrics import BlinkMetrics
from src.preprocessing.face_alignment import FaceAligner


class BlinkInference:
    def __init__(self):
        self.eye_cropper = EyeCropper()
        self.ear_calculator = EARCalculator()
        self.blink_counter = BlinkCounter()
        self.blink_metrics = BlinkMetrics()
        self.face_aligner = FaceAligner()

    # --------------------------------------------------
    # Frame-level inference (UNCHANGED)
    # --------------------------------------------------
    def infer(self, frame):
        landmarks = self.face_aligner.detect_landmarks(frame)
        if landmarks is None:
            return None

        points = self.face_aligner.landmarks_to_array(landmarks, frame.shape)

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

    # --------------------------------------------------
    # LIVE BLINK INFERENCE (NEW, USED IN FINAL SYSTEM)
    # --------------------------------------------------
    def infer_live(self, duration_sec=10, camera_index=0, show_window=True):
        import cv2
        import time

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("❌ Unable to open webcam")
            return None

        print("📷 Waiting for face detection...")

        # Reset metrics
        self.blink_counter.reset()
        self.blink_metrics.reset()

        face_detected = False
        start_time = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            landmarks = self.face_aligner.detect_landmarks(frame)

            if landmarks is not None and not face_detected:
                face_detected = True
                start_time = time.time()
                print("✅ Face detected. Starting blink capture.")

            if face_detected:
                self.infer(frame)

            if show_window:
                if face_detected:
                    remaining = int(duration_sec - (time.time() - start_time))
                    label = f"Blink capture: {max(remaining, 0)}s"
                else:
                    label = "Align face with camera"

                cv2.putText(
                    frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                cv2.imshow("Blink Capture", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if face_detected and time.time() - start_time >= duration_sec:
                break

        cap.release()
        cv2.destroyAllWindows()

        summary = self.blink_metrics.summary()
        return summary if summary else None
