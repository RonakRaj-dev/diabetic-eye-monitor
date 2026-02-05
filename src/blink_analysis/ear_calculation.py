import numpy as np
from scipy.spatial import distance as dist


class EARCalculator:
    @staticmethod
    def compute_ear(eye_points):
        """
        eye_points: np.array of shape (6, 2)
        Returns EAR value
        """

        if eye_points is None or len(eye_points) != 6:
            return 0.0

        # Vertical distances
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])

        # Horizontal distance
        C = dist.euclidean(eye_points[0], eye_points[3])

        if C == 0:
            return 0.0

        ear = (A + B) / (2.0 * C)
        return ear
