# src/training/dataset_loader.py

import os
import pandas as pd
import numpy as np


class DatasetLoader:
    def __init__(
        self,
        blink_feature_dir="./data/processed/blink_features",
        retina_feature_dir="./data/processed/retina_features",
        label_file=None
    ):
        self.blink_feature_dir = blink_feature_dir
        self.retina_feature_dir = retina_feature_dir
        self.label_file = label_file

    # -----------------------------
    # Load blink features (ALL)
    # -----------------------------
    def load_blink_features(self):
        blink_features = []

        for file in os.listdir(self.blink_feature_dir):
            if not file.endswith(".csv"):
                continue

            path = os.path.join(self.blink_feature_dir, file)
            df = pd.read_csv(path)
            blink_features.append(df.values)

        if not blink_features:
            raise RuntimeError("❌ No blink feature files found")

        return np.vstack(blink_features)

    # -----------------------------
    # Load retina base features
    # -----------------------------
    def load_retina_features(self):
        retina_features = []

        for file in os.listdir(self.retina_feature_dir):
            if not file.endswith(".npy"):
                continue

            path = os.path.join(self.retina_feature_dir, file)
            retina_features.append(np.load(path))

        if not retina_features:
            raise RuntimeError("❌ No retina feature files found")

        return np.array(retina_features)

    # -----------------------------
    # Load labels (optional)
    # -----------------------------
    def load_labels(self):
        if self.label_file is None:
            return None

        df = pd.read_csv(self.label_file)
        return df["risk_label"].values

    # -----------------------------
    # Build fused dataset
    # -----------------------------
    def build_dataset(self):
        blink_X = self.load_blink_features()        # shape: (N, B)
        retina_X = self.load_retina_features()      # shape: (M, R)
        labels = self.load_labels()

        X = []
        y = []

        # Cartesian fusion: each blink sample × each retina profile
        for blink_row in blink_X:
            for retina_row in retina_X:
                X.append(np.hstack([blink_row, retina_row]))
                if labels is not None:
                    y.append(labels.mean())  # MVP-safe placeholder

        X = np.array(X)
        y = np.array(y) if labels is not None else None

        return X, y

loader = DatasetLoader()
X, _ = loader.build_dataset()
print("Final training data shape:", X.shape)