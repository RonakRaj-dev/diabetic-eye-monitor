# src/preprocessing/eye_cropper.py

import cv2
import numpy as np
from src.preprocessing.face_alignment import FaceAligner


class EyeCropper:
    # MediaPipe eye landmark indices
    LEFT_EYE_IDX = list(range(33, 42))
    RIGHT_EYE_IDX = list(range(263, 272))

    def __init__(self):
        self.face_aligner = FaceAligner()

    def crop_eyes(self, frame):
        landmarks = self.face_aligner.detect_landmarks(frame)
        if landmarks is None:
            return None, None

        points = self.face_aligner.landmarks_to_array(landmarks, frame.shape)

        left_eye = self._crop_region(points[self.LEFT_EYE_IDX], frame)
        right_eye = self._crop_region(points[self.RIGHT_EYE_IDX], frame)

        return left_eye, right_eye

    def _crop_region(self, eye_points, frame, padding=10):
        x_min = np.min(eye_points[:, 0]) - padding
        y_min = np.min(eye_points[:, 1]) - padding
        x_max = np.max(eye_points[:, 0]) + padding
        y_max = np.max(eye_points[:, 1]) + padding

        h, w = frame.shape[:2]
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)

        return frame[y_min:y_max, x_min:x_max]


if __name__ == "__main__":
    from src.camera.webcam_stream import WebcamStream

    stream = WebcamStream().start()
    cropper = EyeCropper()

    try:
        while True:
            frame = stream.read()
            left, right = cropper.crop_eyes(frame)

            if left is not None:
                cv2.imshow("Left Eye", left)
            if right is not None:
                cv2.imshow("Right Eye", right)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()
