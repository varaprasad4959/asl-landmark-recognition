
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
CONFIDENCE_THRESHOLD = 0.60
SMOOTHING_FRAMES = 10

# UI colors (OpenCV uses BGR)
PANEL_COLOR = (30, 30, 30)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)

# ==================================================
# 2. LOAD MODEL AND SCALER
# ==================================================

for path in [MODEL_PATH, SCALER_PATH, CLASSES_PATH, HAND_MODEL_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    points -= points[0]

    scale = np.max(np.abs(points))
    if scale < 1e-6:
        return None

    points /= scale
    return points.flatten()


# ==================================================
# 5. TEXT COMPOSER
# ==================================================

typed_text = ""
probability_history = deque(maxlen=SMOOTHING_FRAMES)

current_prediction = "No hand"
current_confidence = 0.0
stable_letter = ""
stable_count = 0

STABLE_FRAMES_REQUIRED = 5

HELP_TEXT = [
    "C: Add predicted letter",
    "SPACE: Add space",
    "BACKSPACE: Delete last character",
    "X: Clear text",
    "Q: Quit"
]

# ==================================================
# 6. START WEBCAM
# ==================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Cannot open webcam. Try changing VideoCapture(0) to VideoCapture(1)."
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

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                mp_image, timestamp_ms
            )

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]

                # Draw detected hand landmarks
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
                    features = (features - scaler_mean) / scaler_scale

                    input_tensor = torch.from_numpy(
                        features.astype(np.float32)
                    ).unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits = model(input_tensor)
                        probabilities = torch.softmax(
                            logits, dim=1
                        )[0].cpu().numpy()

                    probability_history.append(probabilities)

                    if len(probability_history) >= 5:
                        average_probabilities = np.mean(
                            probability_history, axis=0
                        )

                        index = int(np.argmax(average_probabilities))
                        current_confidence = float(
                            average_probabilities[index]
                        )

                        if current_confidence >= CONFIDENCE_THRESHOLD:
                            current_prediction = class_names[index]
                        else:
                            current_prediction = "Uncertain"

                        # Track prediction stability
                        if current_prediction == stable_letter:
                            stable_count += 1
                        else:
                            stable_letter = current_prediction
                            stable_count = 1
                else:
                    current_prediction = "Uncertain"
                    current_confidence = 0.0
                    probability_history.clear()
            else:
                current_prediction = "No hand"
                current_confidence = 0.0
                probability_history.clear()
                stable_letter = ""
                stable_count = 0

            # ==================================================
            # 7. DRAW COMPACT UI
            # ==================================================

            # Compact top information panel
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
            fps = 1.0 / max(now - previous_time, 1e-6)
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

            # ==================================================
            # 8. HIGH-VISIBILITY KEYBOARD INSTRUCTIONS
            # ==================================================

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

            for help_line, position in zip(HELP_TEXT, help_positions):
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

            cv2.imshow("ASL Landmark Text Composer", frame)

            # ==================================================
            # 9. KEYBOARD CONTROLS
            # ==================================================

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("c"):
                if (
                    current_prediction in class_names
                    and stable_count >= STABLE_FRAMES_REQUIRED
                    and current_confidence >= CONFIDENCE_THRESHOLD
                ):
                    typed_text += current_prediction
                    print("Text:", typed_text)
                else:
                    print("Hold a clear, stable letter before confirming.")

            elif key == 32:  # SPACE
                if typed_text and not typed_text.endswith(" "):
                    typed_text += " "
                    print("Text:", typed_text)

            elif key in (8, 127):  # BACKSPACE
                typed_text = typed_text[:-1]
                print("Text:", typed_text)

            elif key == ord("x"):
                typed_text = ""
                print("Text cleared.")

finally:
    camera.release()
    cv2.destroyAllWindows()

print("Final text:", typed_text)
print("Application closed.")
