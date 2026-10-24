# ASL Landmark Recognition

A real-time American Sign Language (ASL) alphabet recognition project using MediaPipe hand landmarks and a Multi-Layer Perceptron (MLP) classifier.

The application detects hand landmarks through a webcam, classifies hand gestures into alphabet letters, and allows the user to compose text interactively.

## Features

- Real-time hand landmark detection using MediaPipe.
- Landmark-based alphabet classification using a trained MLP.
- Webcam-based recognition.
- Confidence-based prediction and temporal smoothing.
- Interactive text composition with keyboard controls.

## Technology Stack

- Python
- MediaPipe
- OpenCV
- NumPy
- scikit-learn
- PyTorch

## Project Structure

```text
signlanguagedetection/
├── models/
├── signlanguagedataset/
├── extract_landmarks.py
├── hand_landmarker.task
├── live_landmark_recognition.py
├── live_landmark_text.py
├── train_landmark_model.py
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/varaprasad4959/asl-landmark-recognition.git
cd asl-landmark-recognition
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python live_landmark_text.py
```

Allow webcam access when prompted.

## How It Works

1. MediaPipe detects hand landmarks from webcam frames.
2. The landmarks are normalized into a feature representation.
3. A trained MLP classifier predicts the corresponding alphabet class.
4. Prediction smoothing and confidence checks help stabilize live output.
5. The user confirms predictions and composes text using keyboard controls.

## Dataset

The training pipeline uses hand landmark features extracted from an ASL alphabet image dataset. The extraction and training scripts are included in the repository.

Dataset availability, licensing, and download instructions will be documented before the public release.

## Model Evaluation

The current model achieved approximately 99% accuracy on its held-out test split during development. This is an initial result, not a guarantee of real-world accuracy. More systematic evaluation across users, hand orientations, and lighting conditions is planned.

## Limitations

- This project focuses on alphabet-level hand gesture recognition.
- It is not a complete ASL translation system.
- Continuous signing, facial expressions, grammar, and sentence-level translation are outside the current scope.
- Recognition may vary with hand orientation, lighting, camera quality, and individual signing styles.

## Future Improvements

- Evaluate additional landmark datasets.
- Improve robustness across hand orientations and users.
- Add systematic evaluation and confusion-matrix reporting.
- Improve the application interface and usability.
- Explore broader sign recognition beyond static alphabet gestures.

## License

A repository license will be selected after reviewing the licenses of the source dataset and distributed model assets.
