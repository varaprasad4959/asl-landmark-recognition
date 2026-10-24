
import cv2
import json
import numpy as np
import mediapipe as mp

from pathlib import Path
from collections import Counter


# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "signlanguagedataset" / "asl_alphabet_train"
HAND_MODEL_PATH = ROOT / "hand_landmarker.task"

OUTPUT_DIR = ROOT / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DATA = OUTPUT_DIR / "asl_landmarks_dataset.npz"
OUTPUT_LABELS = OUTPUT_DIR / "asl_landmark_class_names.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Set to None to process all images.
# For a quick initial test, set to 100.
MAX_IMAGES_PER_CLASS = None


# =====================================================
# CHECK FILES
# =====================================================

if not DATA_DIR.is_dir():
    raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

if not HAND_MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"MediaPipe model not found: {HAND_MODEL_PATH}"
    )


# =====================================================
# PREPARE DATASET
# =====================================================

class_dirs = sorted(
    path for path in DATA_DIR.iterdir()
    if path.is_dir()
)

class_names = [path.name for path in class_dirs]

if not class_names:
    raise RuntimeError("No class folders found in dataset.")

print("=" * 55)
print("ASL LANDMARK EXTRACTION")
print("=" * 55)
print("Dataset:", DATA_DIR)
print("Classes:", len(class_names))
print("Output:", OUTPUT_DATA)

features = []
labels = []

processed = 0
failed = 0
class_counts = Counter()


# =====================================================
# LANDMARK NORMALIZATION
# =====================================================

def normalize_landmarks(hand_landmarks):
    """
    Convert 21 MediaPipe landmarks into 63 normalized values.

    Translate coordinates so the wrist is the origin,
    then scale them so hand size has less influence.
    """

    points = np.array(
        [[point.x, point.y, point.z]
         for point in hand_landmarks],
        dtype=np.float32
    )

    # Use wrist landmark (index 0) as origin.
    points = points - points[0]

    # Scale using the largest absolute coordinate.
    scale = np.max(np.abs(points))

    if scale < 1e-6:
        return None

    points = points / scale

    return points.flatten()


# =====================================================
# MEDIAPIPE SETUP
# =====================================================

mp_tasks = mp.tasks
BaseOptions = mp_tasks.BaseOptions
HandLandmarker = mp_tasks.vision.HandLandmarker
HandLandmarkerOptions = mp_tasks.vision.HandLandmarkerOptions
RunningMode = mp_tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH)
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)


# =====================================================
# EXTRACT LANDMARKS
# =====================================================

with HandLandmarker.create_from_options(options) as detector:

    for class_index, class_dir in enumerate(class_dirs):

        image_paths = sorted(
            path for path in class_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if MAX_IMAGES_PER_CLASS is not None:
            image_paths = image_paths[:MAX_IMAGES_PER_CLASS]

        print(
            f"\n[{class_index + 1}/{len(class_dirs)}] "
            f"{class_dir.name}: {len(image_paths)} images"
        )

        for image_path in image_paths:

            image_bgr = cv2.imread(str(image_path))

            if image_bgr is None:
                failed += 1
                continue

            image_rgb = cv2.cvtColor(
                image_bgr,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=image_rgb
            )

            try:
                result = detector.detect(mp_image)

                if not result.hand_landmarks:
                    failed += 1
                    continue

                # Use the first detected hand for this sample.
                vector = normalize_landmarks(
                    result.hand_landmarks[0]
                )

                if vector is None:
                    failed += 1
                    continue

                features.append(vector)
                labels.append(class_index)

                class_counts[class_dir.name] += 1

            except Exception as error:
                failed += 1
                print(
                    f"Error processing {image_path.name}: {error}"
                )

            processed += 1

            if processed % 500 == 0:
                print(
                    f"Processed: {processed} | "
                    f"Landmarks saved: {len(features)} | "
                    f"Skipped: {failed}"
                )


# =====================================================
# SAVE DATASET
# =====================================================

if not features:
    raise RuntimeError(
        "No hand landmarks were extracted. "
        "Check the dataset and hand_landmarker.task file."
    )

X = np.asarray(features, dtype=np.float32)
y = np.asarray(labels, dtype=np.int64)

np.savez_compressed(
    OUTPUT_DATA,
    X=X,
    y=y
)

with open(OUTPUT_LABELS, "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=2)

print("\n" + "=" * 55)
print("EXTRACTION COMPLETE")
print("=" * 55)
print("Feature shape:", X.shape)
print("Label shape:", y.shape)
print("Successful samples:", len(X))
print("Skipped images:", failed)
print("\nSamples per class:")

for class_name in class_names:
    print(f"  {class_name}: {class_counts[class_name]}")

print("\nSaved dataset:", OUTPUT_DATA)
print("Saved classes:", OUTPUT_LABELS)
 