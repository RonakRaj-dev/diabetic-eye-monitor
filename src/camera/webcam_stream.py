import cv2


class WebcamStream:
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("❌ Unable to access webcam")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return self

    def read(self):
        if self.cap is None:
            raise RuntimeError("⚠️ Webcam stream not started")

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("⚠️ Failed to read frame from webcam")

        return frame

    def stop(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def __del__(self):
        self.stop()


if __name__ == "__main__":
    stream = WebcamStream().start()

    try:
        while True:
            frame = stream.read()
            cv2.imshow("Webcam Stream", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()
