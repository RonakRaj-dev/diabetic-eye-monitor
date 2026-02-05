# live_demo.py

import cv2
from src.camera.webcam_stream import WebcamStream
from src.inference.combined_inference import CombinedInference


def main():
    stream = WebcamStream().start()
    infer = CombinedInference()

    print("🎥 Live demo started. Press 'q' to quit.")

    try:
        while True:
            frame = stream.read()
            result = infer.infer(frame)

            if result:
                text = f"RISK: {result['risk_level']} | SCORE: {result['risk_score']} | MODE: {result['mode']}"
                cv2.putText(
                    frame,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                print(result)

            cv2.imshow("Diabetic Eye Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
