# src/utils/config.py

class Config:
    # Camera
    CAMERA_ID = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480

    # Eye preprocessing
    EYE_SIZE = (64, 64)

    # Blink detection
    EAR_THRESHOLD = 0.25
    MIN_FRAMES_CLOSED = 2

    # Blink fatigue normalization
    MAX_BLINK_RATE = 30.0      # blinks/min
    MAX_BLINK_DURATION = 0.4  # seconds

    # Focus scoring
    FOCUS_LOW = 50.0
    FOCUS_HIGH = 300.0

    # Risk fusion weights (must sum to 1)
    WEIGHT_FATIGUE = 0.4
    WEIGHT_FOCUS = 0.3
    WEIGHT_VESSEL = 0.3

    # Risk thresholds
    LOW_RISK_TH = 0.35
    HIGH_RISK_TH = 0.65
