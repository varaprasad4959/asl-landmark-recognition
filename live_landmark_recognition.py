
import cv2
import json
import time
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from collections import deque
from pathlib import Path

# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "asl_landmark_mlp.pth"
SCALER_PATH = BASE_DIR / "models" / "asl_landmark_scaler.npz"
CLASSES_PATH = BASE_DIR / "models" / "asl_landmark_mlp_classes.json"
HAND_MODEL_PATH = BASE_DIR / "hand_landmarker.task"

MIRROR_WEBCAM = True
CONFIDENCE_THRESHOLD = 0.70
SMOOTHING_FRAMES = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

for path in [MODEL_PATH, SCALER_PATH, CLASSES_PATH, HAND_MODEL_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

# --------------------------------------------------
# 2. LOAD CLASS NAMES AND SCALER
# --------------------------------------------------

with open(CLASSES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

scaler_data = np.load(SCALER_PATH)
scaler_mean = scaler_data["mean"].astype(np.float32)
scaler_scale = scaler_data["scale"].astype(np.float32)

# --------------------------------------------------
# 3. RECREATE THE EXACT TRAINING MODEL
# --------------------------------------------------

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

print("Landmark model loaded successfully.")
print(f"Classes: {len(class_names)}")

# --------------------------------------------------
# 4. MEDIAPIPE HAND LANDMARKER
# --------------------------------------------------

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

# --------------------------------------------------
# 5. NORMALIZE LANDMARKS
# Must match extract_landmarks.py exactly
# --------------------------------------------------

def extract_features(hand_landmarks):
    points = np.array(
        [[p.x, p.y, p.z] for p in hand_landmarks],
        dtype=np.float32
    )

    # Make wrist landmark the origin
    points = points - points[0]

    # Scale landmarks to a similar size
    scale = np.max(np.abs(points))

    if scale < 1e-6:
        return None

    points = points / scale

    return points.flatten()


# --------------------------------------------------
# 6. WEBCAM AND PREDICTION
# --------------------------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Could not open webcam. Close other camera apps "
        "or try changing VideoCapture(0) to VideoCapture(1)."
    )

probability_history = deque(maxlen=SMOOTHING_FRAMES)
last_timestamp_ms = 0
previous_time = time.time()

try:
    with HandLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
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
                mp_image,
                timestamp_ms
            )

            display_label = "No hand detected"
            display_confidence = 0.0

            if result.hand_landmarks:
                # This baseline classifies the first detected hand.
                hand = result.hand_landmarks[0]

                # Draw hand landmarks
                for point in hand:
                    px = int(point.x * width)
                    py = int(point.y * height)

                    if 0 <= px < width and 0 <= py < height:
                        cv2.circle(
                            frame,
                            (px, py),
                            4,
                            (0, 255, 0),
                            -1
                        )

                features = extract_features(hand)

                if features is not None:
                    # Apply the same StandardScaler used in training
                    features = (
                        features - scaler_mean
                    ) / scaler_scale

                    input_tensor = torch.from_numpy(
                        features.astype(np.float32)
                    ).unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits = model(input_tensor)
                        probabilities = torch.softmax(
                            logits, dim=1
                        )[0].cpu().numpy()

                    probability_history.append(probabilities)

                    # Smooth predictions across recent frames
                    if len(probability_history) >= 5:
                        average_probabilities = np.mean(
                            probability_history, axis=0
                        )

                        predicted_index = int(
                            np.argmax(average_probabilities)
                        )

                        display_confidence = float(
                            average_probabilities[predicted_index]
                        )

                        if display_confidence >= CONFIDENCE_THRESHOLD:
                            display_label = class_names[predicted_index]
                        else:
                            display_label = "Uncertain"
                else:
                    probability_history.clear()
            else:
                probability_history.clear()

            # FPS display
            current_time = time.time()
            fps = 1.0 / max(current_time - previous_time, 1e-6)
            previous_time = current_time

            # Main prediction display
            cv2.rectangle(
                frame, (10, 10), (width - 10, 105),
                (25, 25, 25), -1
            )

            cv2.putText(
                frame,
                f"ASL Sign: {display_label}",
                (25, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0) if display_label not in
                ["No hand detected", "Uncertain"] else (0, 200, 255),
                2
            )

            cv2.putText(
                frame,
                f"Confidence: {display_confidence:.2f}",
                (25, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | Press Q to quit",
                (25, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow("ASL Landmark Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("Webcam recognition stopped.")
