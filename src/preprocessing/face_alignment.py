# src/preprocessing/face_alignment.py

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat


class FaceAligner:
    def __init__(self):
        base_options = python.BaseOptions(
            model_asset_path="models/face_landmarker.task"
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )

        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = Image(
            image_format=ImageFormat.SRGB,
            data=rgb
        )

        result = self.detector.detect(mp_image)

        if not result.face_landmarks:
            return None

        return result.face_landmarks[0]

    def landmarks_to_array(self, landmarks, frame_shape):
        h, w = frame_shape[:2]
        points = []

        for lm in landmarks:
            points.append((int(lm.x * w), int(lm.y * h)))

        return np.array(points)
