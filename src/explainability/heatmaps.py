# src/explainability/heatmaps.py

import cv2
import numpy as np


class HeatmapGenerator:
    def __init__(self):
        pass

    @staticmethod
    def focus_heatmap(image_bgr):
        """
        Generates a heatmap based on Laplacian response (focus/blur)
        """
        if image_bgr is None or image_bgr.size == 0:
            return None

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Laplacian highlights sharp regions
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.absolute(laplacian)

        # Normalize
        laplacian = cv2.normalize(
            laplacian, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        heatmap = cv2.applyColorMap(laplacian, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image_bgr, 0.6, heatmap, 0.4, 0)

        return overlay

    @staticmethod
    def edge_heatmap(image_bgr):
        """
        Highlights vessel/edge-rich areas
        """
        if image_bgr is None or image_bgr.size == 0:
            return None

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        heatmap = cv2.applyColorMap(edges, cv2.COLORMAP_HOT)
        overlay = cv2.addWeighted(image_bgr, 0.7, heatmap, 0.3, 0)

        return overlay
