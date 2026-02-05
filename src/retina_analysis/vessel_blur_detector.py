# src/retina_analysis/vessel_blur_detector.py

import cv2
import numpy as np


class VesselBlurDetector:
    def __init__(self):
        pass

    def vessel_clarity_score(self, image_bgr):
        """
        Higher score = clearer vessels / edges
        """
        if image_bgr is None or image_bgr.size == 0:
            return 0.0

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Enhance vessels/edges
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        edges = cv2.Canny(enhanced, 50, 150)
        edge_density = edges.mean() / 255.0  # 0..1

        # Blur sensitivity via Laplacian variance
        lap_var = cv2.Laplacian(enhanced, cv2.CV_64F).var()
        blur_penalty = min(lap_var / 300.0, 1.0)

        # Combine
        score = 0.6 * edge_density + 0.4 * blur_penalty
        return float(round(score, 3))
