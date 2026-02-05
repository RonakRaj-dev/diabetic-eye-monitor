# src/retina_analysis/pupil_detection.py

import cv2
import numpy as np


class PupilDetector:
    def __init__(self, blur_ksize=7):
        self.blur_ksize = blur_ksize

    def detect(self, eye_bgr):
        """
        Returns: (center_x, center_y), radius, mask
        """
        if eye_bgr is None or eye_bgr.size == 0:
            return None, None, None

        gray = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)

        # Adaptive threshold to isolate dark pupil
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None, None, thresh

        # Choose largest dark blob (likely pupil)
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 50:
            return None, None, thresh

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))

        return center, int(radius), thresh
