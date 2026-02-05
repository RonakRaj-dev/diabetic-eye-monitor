# src/retina_analysis/focus_score.py

import cv2
import numpy as np


class FocusScorer:
    def __init__(self):
        pass

    @staticmethod
    def laplacian_variance(image_bgr):
        if image_bgr is None or image_bgr.size == 0:
            return 0.0

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        score = lap.var()
        return float(score)

    @staticmethod
    def normalized_focus(score, low=50.0, high=300.0):
        """
        Map raw focus score to [0,1]
        """
        if score <= low:
            return 0.0
        if score >= high:
            return 1.0
        return (score - low) / (high - low)
