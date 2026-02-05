# src/explainability/metric_explainer.py

class MetricExplainer:
    def __init__(self):
        pass

    @staticmethod
    def explain(blink_metrics, retina_metrics, risk_score):
        explanations = []

        # Blink explanation
        if blink_metrics["fatigue_score"] > 0.6:
            explanations.append(
                "Abnormal blink fatigue detected (frequent or prolonged blinks)."
            )
        elif blink_metrics["fatigue_score"] > 0.4:
            explanations.append(
                "Mild blink irregularity observed."
            )

        # Retina explanation
        if retina_metrics["avg_focus_score"] < 0.4:
            explanations.append(
                "Reduced retinal focus clarity detected."
            )

        if retina_metrics["avg_vessel_clarity"] < 0.4:
            explanations.append(
                "Possible retinal vessel blur observed."
            )

        if not explanations:
            explanations.append(
                "Eye behavior appears within normal range."
            )

        return {
            "risk_score": risk_score,
            "explanations": explanations
        }
