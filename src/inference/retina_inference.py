# src/inference/retina_inference.py

import cv2
from src.retina_analysis.focus_score import FocusScorer
from src.retina_analysis.vessel_blur_detector import VesselBlurDetector


class RetinaInference:
    """
    Retina inference using a medical retina (fundus) image.
    This module does NOT use webcam frames.
    """

    def __init__(self):
        self.focus_scorer = FocusScorer()
        self.vessel_detector = VesselBlurDetector()

    def infer_from_image(self, image_path):

        img = cv2.imread(image_path)

        if img is None:
            return None

        # --- Focus analysis ---
        focus_raw = self.focus_scorer.laplacian_variance(img)
        focus_norm = self.focus_scorer.normalized_focus(focus_raw)

        # --- Vessel clarity analysis ---
        vessel_score = self.vessel_detector.vessel_clarity_score(img)

        return {
            "avg_focus_score": round(float(focus_norm), 3),
            "avg_vessel_clarity": round(float(vessel_score), 3)
        }
