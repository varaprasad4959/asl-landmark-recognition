
import cv2
import json
import time
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp

from pathlib import Path
from collections import deque


# ==================================================
# 1. PATHS AND SETTINGS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "asl_landmark_mlp.pth"
SCALER_PATH = MODEL_DIR / "asl_landmark_scaler.npz"
CLASSES_PATH = MODEL_DIR / "asl_landmark_mlp_classes.json"
HAND_MODEL_PATH = BASE_DIR / "hand_landmarker.task"

MIRROR_WEBCAM = True

NO_HAND_SPACE_TIMEOUT = 2 # seconds
CONFIDENCE_THRESHOLD = 0.70
SMOOTHING_FRAMES = 10
STABLE_FRAMES_REQUIRED = 15

LETTER_CAPTURE_COOLDOWN = 0.8  # seconds
REARM_INVALID_FRAMES = 3

# UI colors (OpenCV uses BGR)
PANEL_COLOR = (30, 30, 30)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)


# ==================================================
# 2. LOAD MODEL AND SCALER
# ==================================================

for path in [
    MODEL_PATH,
    SCALER_PATH,
    CLASSES_PATH,
    HAND_MODEL_PATH
]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


class LandmarkMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


with open(CLASSES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

scaler_data = np.load(SCALER_PATH)

scaler_mean = scaler_data["mean"].astype(np.float32)
scaler_scale = scaler_data["scale"].astype(np.float32)

# Prevent division by zero.
scaler_scale[scaler_scale == 0] = 1.0

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model = LandmarkMLP(
    checkpoint["input_size"],
    checkpoint["num_classes"]
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print("MLP landmark model loaded successfully.")


# ==================================================
# 3. MEDIAPIPE
# ==================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==================================================
# 4. LANDMARK PREPROCESSING
# Must match training
# ==================================================

def extract_features(hand_landmarks):
    points = np.array(
        [[p.x, p.y, p.z] for p in hand_landmarks],
        dtype=np.float32
    )

    # Normalize relative to wrist landmark.
    points -= points[0]

    scale = np.max(np.abs(points))

    if scale < 1e-6:
        return None

    points /= scale

    return points.flatten()


# ==================================================
# 5. TEXT COMPOSER AND AUTOMATIC CAPTURE STATE
# ==================================================

typed_text = ""
word_started = False
no_hand_start_time = None

probability_history = deque(
    maxlen=SMOOTHING_FRAMES
)

current_prediction = "No hand"
current_confidence = 0.0

stable_letter = ""
stable_count = 0

# Automatic capture state.
capture_armed = True
last_captured_letter = ""
last_capture_time = 0.0
invalid_prediction_count = 0

HELP_TEXT = [
    "Letters captured automatically",
    "SPACE: Add space",
    "BACKSPACE: Delete character",
    "X: Clear text",
    "Q: Quit"
]


# ==================================================
# 6. START WEBCAM
# ==================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    camera.release()

    raise RuntimeError(
        "Cannot open webcam. "
        "Try changing VideoCapture(0) to VideoCapture(1)."
    )

last_timestamp_ms = 0
previous_time = time.time()

try:
    with HandLandmarker.create_from_options(options) as landmarker:

        while True:

            success, frame = camera.read()

            if not success:
                print("Unable to read webcam.")
                break

            if MIRROR_WEBCAM:
                frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = max(
                int(time.monotonic() * 1000),
                last_timestamp_ms + 1
            )

            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )
            # ==========================================
            # AUTOMATIC SPACE AFTER HAND DISAPPEARS
            # ==========================================

            if result.hand_landmarks:
                no_hand_start_time = None

            else:
                if word_started:
                    if no_hand_start_time is None:
                        no_hand_start_time = time.time()

                    elif (
                    time.time() - no_hand_start_time
                    >= NO_HAND_SPACE_TIMEOUT
                    ):
                        if typed_text and not typed_text.endswith(" "):
                            typed_text += " "
                            print("Automatic space inserted.")
                            print("Text:", typed_text)

                        word_started = False
                        no_hand_start_time = None
            # ==========================================
            # HAND DETECTION AND MODEL PREDICTION
            # ==========================================

            if result.hand_landmarks:

                # Preserve the original behavior:
                # classify the first detected hand.
                hand = result.hand_landmarks[0]

                # Draw hand landmarks.
                for point in hand:
                    px = int(point.x * width)
                    py = int(point.y * height)

                    if 0 <= px < width and 0 <= py < height:
                        cv2.circle(
                            frame,
                            (px, py),
                            4,
                            GREEN,
                            -1
                        )

                features = extract_features(hand)

                if features is not None:

                    features = (
                        features - scaler_mean
                    ) / scaler_scale

                    input_tensor = torch.from_numpy(
                        features.astype(np.float32)
                    ).unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits = model(input_tensor)

                        probabilities = torch.softmax(
                            logits,
                            dim=1
                        )[0].cpu().numpy()

                    probability_history.append(probabilities)

                    if len(probability_history) >= 5:

                        average_probabilities = np.mean(
                            probability_history,
                            axis=0
                        )

                        index = int(
                            np.argmax(average_probabilities)
                        )

                        current_confidence = float(
                            average_probabilities[index]
                        )

                        if (
                            current_confidence
                            >= CONFIDENCE_THRESHOLD
                        ):
                            current_prediction = class_names[index]
                        else:
                            current_prediction = "Uncertain"

                    else:
                        current_prediction = "Uncertain"
                        current_confidence = 0.0

                else:
                    current_prediction = "Uncertain"
                    current_confidence = 0.0
                    probability_history.clear()

            else:
                current_prediction = "No hand"
                current_confidence = 0.0
                probability_history.clear()

            # ==========================================
            # AUTOMATIC LETTER CAPTURE
            # ==========================================

            is_valid_letter = (
                isinstance(current_prediction, str)
                and len(current_prediction) == 1
                and current_prediction.isalpha()
                and current_prediction.upper() in class_names
                and current_confidence >= CONFIDENCE_THRESHOLD
            )

            if is_valid_letter:

                invalid_prediction_count = 0

                # Track stable consecutive predictions.
                if current_prediction == stable_letter:
                    stable_count += 1
                else:
                    stable_letter = current_prediction
                    stable_count = 1

                cooldown_finished = (
                    time.time() - last_capture_time
                    >= LETTER_CAPTURE_COOLDOWN
                )

                if (
                    stable_count >= STABLE_FRAMES_REQUIRED
                    and cooldown_finished
                ):

                    # Capture when armed, or when a different
                    # stable letter is detected.
                    if (
                        capture_armed
                        or current_prediction != last_captured_letter
                    ):

                        typed_text += current_prediction
                        word_started = True

                        last_captured_letter = current_prediction
                        last_capture_time = time.time()
                        capture_armed = False

                        print("Captured:", current_prediction)
                        print("Text:", typed_text)

            else:
                stable_letter = ""
                stable_count = 0

                invalid_prediction_count += 1

                # Rearm after a short invalid/release interval.
                if (
                    invalid_prediction_count
                    >= REARM_INVALID_FRAMES
                ):
                    capture_armed = True
                    last_captured_letter = ""

            # ==========================================
            # 7. DRAW COMPACT UI
            # ==========================================

            panel_height = 155

            cv2.rectangle(
                frame,
                (0, 0),
                (width, panel_height),
                PANEL_COLOR,
                -1
            )

            # Prediction
            cv2.putText(
                frame,
                f"Prediction: {current_prediction}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                GREEN,
                2,
                cv2.LINE_AA
            )

            # Confidence
            cv2.putText(
                frame,
                f"Confidence: {current_confidence:.2f}",
                (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                WHITE,
                2,
                cv2.LINE_AA
            )

            # Composed text
            cv2.putText(
                frame,
                f"Text: {typed_text[-45:]}",
                (15, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                CYAN,
                2,
                cv2.LINE_AA
            )

            # Stable frames
            cv2.putText(
                frame,
                f"Stable frames: {stable_count}",
                (15, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                WHITE,
                1,
                cv2.LINE_AA
            )

            # FPS
            now = time.time()

            fps = 1.0 / max(
                now - previous_time,
                1e-6
            )

            previous_time = now

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (width - 145, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                WHITE,
                2,
                cv2.LINE_AA
            )

            # Capture status
            capture_status = (
                "Ready for next sign"
                if capture_armed
                else "Captured - release to repeat"
            )

            cv2.putText(
                frame,
                capture_status,
                (width - 350, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                GREEN,
                1,
                cv2.LINE_AA
            )

            # ==========================================
            # 8. KEYBOARD INSTRUCTIONS
            # ==========================================

            help_panel_height = 65

            cv2.rectangle(
                frame,
                (0, height - help_panel_height),
                (width, height),
                PANEL_COLOR,
                -1
            )

            help_positions = [
                (15, height - 40),
                (width // 3, height - 40),
                (2 * width // 3, height - 40),
                (15, height - 12),
                (width // 3, height - 12)
            ]

            for help_line, position in zip(
                HELP_TEXT,
                help_positions
            ):
                cv2.putText(
                    frame,
                    help_line,
                    position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    GREEN,
                    1,
                    cv2.LINE_AA
                )

            cv2.imshow(
                "ASL Landmark Text Composer",
                frame
            )

            # ==========================================
            # 9. KEYBOARD CONTROLS
            # ==========================================

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == 32:  # SPACE

                if typed_text and not typed_text.endswith(" "):
                    typed_text += " "
                    print("Text:", typed_text)

            elif key in (8, 127):  # BACKSPACE

                typed_text = typed_text[:-1]
                print("Text:", typed_text)

            elif key == ord("x"):  # CLEAR TEXT

                typed_text = ""
                print("Text cleared.")

                # Avoid immediate recapture of a held sign.
                capture_armed = False
                last_captured_letter = stable_letter

finally:
    camera.release()
    cv2.destroyAllWindows()


print("Final text:", typed_text)
print("Application closed.")
