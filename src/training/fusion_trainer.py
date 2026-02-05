# src/training/fusion_trainer.py

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.training.dataset_loader import DatasetLoader


class FusionTrainer:
    def train(self, X, y):
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        os.makedirs("models/fusion_model", exist_ok=True)
        joblib.dump(model, "models/fusion_model/risk_model.pkl")

        return model


# --------------------------------------------------
# RUNNER
# --------------------------------------------------
if __name__ == "__main__":

    print("🔹 Loading fused dataset...")
    loader = DatasetLoader()
    X, y = loader.build_dataset()

    # --------------------------------------------------
    # MVP LABEL STRATEGY (since no true labels)
    # Rule-based pseudo labels
    # --------------------------------------------------
    print("🔹 Generating pseudo-labels...")
    fatigue_idx = 0        # first blink feature
    focus_idx = -2         # retina focus
    vessel_idx = -1        # retina vessel clarity

    risk_score = (
        0.4 * X[:, fatigue_idx] +
        0.3 * (1 - X[:, focus_idx]) +
        0.3 * (1 - X[:, vessel_idx])
    )

    y = (risk_score > np.median(risk_score)).astype(int)

    # --------------------------------------------------
    print("🔹 Training fusion model...")
    trainer = FusionTrainer()
    model = trainer.train(X, y)

    print("✅ Fusion model trained and saved at:")
    print("   models/fusion_model/risk_model.pkl")
