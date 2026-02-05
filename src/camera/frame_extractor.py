import cv2
import numpy as np


class FrameExtractor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def preprocess(self, frame: np.ndarray) -> dict:
        """
        Takes raw frame and returns processed versions
        """

        if frame is None:
            raise ValueError("Empty frame received")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(frame, self.target_size)
        gray_resized = cv2.resize(gray, self.target_size)

        return {
            "original": frame,
            "gray": gray,
            "resized": resized,
            "gray_resized": gray_resized
        }


if __name__ == "__main__":
    from webcam_stream import WebcamStream

    stream = WebcamStream().start()
    extractor = FrameExtractor()

    try:
        while True:
            frame = stream.read()
            processed = extractor.preprocess(frame)

            cv2.imshow("Original", processed["original"])
            cv2.imshow("Gray", processed["gray"])

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()
