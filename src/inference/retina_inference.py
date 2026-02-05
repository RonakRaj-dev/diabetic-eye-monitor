# src/inference/retina_inference.py

from src.preprocessing.eye_cropper import EyeCropper
from src.retina_analysis.pupil_detection import PupilDetector
from src.retina_analysis.focus_score import FocusScorer
from src.retina_analysis.vessel_blur_detector import VesselBlurDetector


class RetinaInference:
    def __init__(self):
        self.eye_cropper = EyeCropper()
        self.pupil_detector = PupilDetector()
        self.focus_scorer = FocusScorer()
        self.vessel_detector = VesselBlurDetector()

    def infer(self, frame):
        left_eye, right_eye = self.eye_cropper.crop_eyes(frame)

        scores = []

        for eye in [left_eye, right_eye]:
            if eye is None:
                continue

            _, radius, _ = self.pupil_detector.detect(eye)
            focus_raw = self.focus_scorer.laplacian_variance(eye)
            focus_norm = self.focus_scorer.normalized_focus(focus_raw)
            vessel_score = self.vessel_detector.vessel_clarity_score(eye)

            scores.append({
                "pupil_radius": radius or 0,
                "focus_score": round(focus_norm, 3),
                "vessel_clarity": vessel_score
            })

        if not scores:
            return None

        # Average both eyes
        avg_focus = sum(s["focus_score"] for s in scores) / len(scores)
        avg_vessel = sum(s["vessel_clarity"] for s in scores) / len(scores)

        return {
            "avg_focus_score": round(avg_focus, 3),
            "avg_vessel_clarity": round(avg_vessel, 3)
        }
