# src/preprocessing/image_normalizer.py

import cv2
import numpy as np


class ImageNormalizer:
    def __init__(self, target_size=(64, 64)):
        self.target_size = target_size

    def normalize(self, image):
        if image is None or image.size == 0:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.target_size)

        # Histogram Equalization for lighting normalization
        normalized = cv2.equalizeHist(resized)

        # Scale to [0,1]
        normalized = normalized.astype("float32") / 255.0

        return normalized


if __name__ == "__main__":
    from src.camera.webcam_stream import WebcamStream
    from eye_cropper import EyeCropper

    stream = WebcamStream().start()
    cropper = EyeCropper()
    normalizer = ImageNormalizer()

    try:
        while True:
            frame = stream.read()
            left, right = cropper.crop_eyes(frame)

            if left is not None:
                norm_left = normalizer.normalize(left)
                cv2.imshow("Normalized Left Eye", norm_left)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()
