# src/utils/validators.py

import numpy as np


def validate_frame(frame):
    if frame is None:
        return False
    if not isinstance(frame, np.ndarray):
        return False
    if frame.size == 0:
        return False
    return True


def validate_eye_crop(eye):
    if eye is None:
        return False
    if eye.size == 0:
        return False
    return True


def validate_metrics(metrics: dict, required_keys: list):
    if not isinstance(metrics, dict):
        return False
    for key in required_keys:
        if key not in metrics:
            return False
    return True
