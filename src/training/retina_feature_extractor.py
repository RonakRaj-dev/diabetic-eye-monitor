# src/training/retina_feature_extractor.py

import cv2
import os
import numpy as np
from src.retina_analysis.focus_score import FocusScorer
from src.retina_analysis.vessel_blur_detector import VesselBlurDetector


class RetinaFeatureExtractor:
    def __init__(self):
        self.focus = FocusScorer()
        self.vessel = VesselBlurDetector()

    # --------------------------------------------------
    # SINGLE MEDICAL RETINA IMAGE (LIVE / INFERENCE)
    # --------------------------------------------------
    def extract_from_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None

        # Optional resize for stability
        img = cv2.resize(img, (512, 512))

        focus_raw = self.focus.laplacian_variance(img)
        focus_norm = self.focus.normalized_focus(focus_raw)
        vessel_score = self.vessel.vessel_clarity_score(img)

        return np.array(
            [float(focus_norm), float(vessel_score)],
            dtype=float
        )

    # --------------------------------------------------
    # FOLDER OF RETINA IMAGES (TRAINING / DATASET)
    # --------------------------------------------------
    def extract_from_folder(self, folder_path):
        feature_list = []

        for file in os.listdir(folder_path):
            if file.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
                img_path = os.path.join(folder_path, file)
                features = self.extract_from_image(img_path)

                if features is not None:
                    feature_list.append(features)

        if not feature_list:
            return None

        # Aggregate across images (robust mean)
        return np.mean(feature_list, axis=0)


# --------------------------------------------------
# RUNNER (BATCH DATASET EXTRACTION)
# --------------------------------------------------
if __name__ == "__main__":

    INPUT_ROOT = r"C:\Users\gamer\OneDrive\Desktop\diabetic-eye-monitor\data\raw\retina_images"
    OUTPUT_ROOT = r"C:\Users\gamer\OneDrive\Desktop\diabetic-eye-monitor\data\processed\retina_features"

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    extractor = RetinaFeatureExtractor()

    for base_name in os.listdir(INPUT_ROOT):
        base_path = os.path.join(INPUT_ROOT, base_name)

        if not os.path.isdir(base_path):
            continue

        base_features = extractor.extract_from_folder(base_path)

        if base_features is None:
            print(f"⚠️ No valid retina images found in {base_name}")
            continue

        output_file = os.path.join(
            OUTPUT_ROOT, f"{base_name}_features.npy"
        )

        np.save(output_file, base_features)

        print(f"✅ Saved retina features for {base_name} → {output_file}")
