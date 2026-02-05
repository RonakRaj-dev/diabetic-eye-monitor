# src/training/blink_feature_extractor.py

import pandas as pd
import numpy as np
import os


class BlinkFeatureExtractor:
    @staticmethod
    def _clean_series(series_str):
        """
        Converts a comma-separated string to a clean float array.
        Replaces NA / empty values with 0.0
        """
        values = []
        for v in str(series_str).split(","):
            v = v.strip()
            if v in ["NA", "N/A", "", "null", "None"]:
                values.append(0.0)
            else:
                try:
                    values.append(float(v))
                except ValueError:
                    values.append(0.0)
        return np.array(values, dtype=float)

    def extract(self, csv_path):
        df = pd.read_csv(csv_path)

        features = []

        for _, row in df.iterrows():
            left = self._clean_series(row["eye_openness_left_series"])
            right = self._clean_series(row["eye_openness_right_series"])

            # Safety check
            if len(left) < 2 or len(right) < 2:
                continue

            feature = {
                "blink_duration": row["blink_duration"],
                "mean_left": left.mean(),
                "mean_right": right.mean(),
                "min_left": left.min(),
                "min_right": right.min(),
                "closure_speed_left": np.diff(left).min(),
                "closure_speed_right": np.diff(right).min(),
                "asymmetry": abs(left.mean() - right.mean())
            }

            features.append(feature)

        return pd.DataFrame(features)


# -------------------------------
# RUNNER
# -------------------------------
if __name__ == "__main__":
    INPUT_CSV = r"C:\Users\gamer\OneDrive\Desktop\diabetic-eye-monitor\data\raw\blink_dataset\blink_pattern.csv"
    OUTPUT_CSV = r"C:\Users\gamer\OneDrive\Desktop\diabetic-eye-monitor\data\processed\blink_features\blink_pattern_features.csv"

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    extractor = BlinkFeatureExtractor()
    features = extractor.extract(INPUT_CSV)
    features.to_csv(OUTPUT_CSV, index=False)

    print(f"✅ Blink features saved to: {OUTPUT_CSV}")
