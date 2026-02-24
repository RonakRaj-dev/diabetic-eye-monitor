from src.inference.blink_inference import BlinkInference
from src.inference.retina_inference import RetinaInference
from src.utils.config import Config


class CombinedInference:
    def __init__(self):
        self.blink_infer = BlinkInference()
        self.retina_infer = RetinaInference()

    def infer(self, retina_image_path, blink_duration_sec=10):
        # --- Live blink inference ---
        blink_data = self.blink_infer.infer_live(duration_sec=blink_duration_sec)

        # --- Retina image inference ---
        retina_data = self.retina_infer.infer_from_image(retina_image_path)

        if blink_data is None:
            print("⚠️ Blink data not captured properly. Please retry.")
            return None

        if retina_data is None:
            print("⚠️ Retina image could not be processed.")
            return None

        fatigue = blink_data["fatigue_score"]
        focus = retina_data["avg_focus_score"]
        vessel = retina_data["avg_vessel_clarity"]

        risk_score = (
                Config.WEIGHT_FATIGUE * fatigue +
                Config.WEIGHT_FOCUS * (1 - focus) +
                Config.WEIGHT_VESSEL * (1 - vessel)
        )

        level = self._risk_level(risk_score)

        return {
            "risk_level": level,
            "risk_score": round(risk_score, 3),
            "mode": "live_blink + retina_image",
            "blink_metrics": blink_data,
            "retina_metrics": retina_data,
            "recommendation": self._recommend(level)
        }

    @staticmethod
    def _risk_level(score):
        if score < Config.LOW_RISK_TH:
            return "LOW"
        elif score < Config.HIGH_RISK_TH:
            return "MODERATE"
        return "HIGH"

    @staticmethod
    def _recommend(level):
        if level == "LOW":
            return "No immediate concern. Maintain regular eye checkups."
        if level == "MODERATE":
            return "Early signs detected. Eye screening advised."
        return "High risk indicators detected. Consult ophthalmologist."
