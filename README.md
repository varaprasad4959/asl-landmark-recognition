# 🤟 ASL Landmark Recognition

**Real-time American Sign Language (ASL) alphabet recognition using MediaPipe hand landmarks, PyTorch, and a PySide6 desktop interface.**

ASL Landmark Recognition is a computer vision project that detects hand landmarks from a webcam, classifies hand gestures, and converts recognized signs into editable text. It includes automatic letter capture, automatic spacing, and text-to-speech output.

The project is designed as a lightweight, landmark-based approach to real-time hand gesture recognition.

## ✨ Features

- **Real-time hand tracking** using MediaPipe's Hand Landmarker.
- **Landmark-based classification** using a trained PyTorch Multi-Layer Perceptron (MLP).
- **Live webcam preview** with a polished dark charcoal and teal desktop interface.
- **Automatic letter capture** when a recognized gesture remains stable.
- **Automatic spacing** after a period without a detected hand.
- **Editable text composition** with controls for inserting spaces, deleting characters, and clearing text.
- **Text-to-speech** to read the composed text aloud, including repeated playback.
- **Confidence thresholding and prediction smoothing** to reduce unstable predictions.
- **Keyboard shortcuts** for quick interaction.

## 🖥️ Application Preview

The desktop application includes a live camera panel, gesture prediction and confidence information, and a text composition area.

![alt text](<Screenshot 2026-10-28 011535.png>)

## 🎬 Working Demo

Watch the application perform real-time hand tracking, automatic letter capture, text composition, and text-to-speech output.

<video controls src="asl-demo.mp4" title="Title"></video>

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| MediaPipe | Hand landmark detection |
| PyTorch | Landmark classification model |
| OpenCV | Webcam capture and image processing |
| NumPy | Feature processing and numerical operations |
| PySide6 / Qt | Desktop graphical interface |
| pyttsx3 | Text-to-speech output |

## 🧠 How It Works

The application uses a two-stage recognition pipeline.

1. **Hand detection:** MediaPipe detects the hand and its 21 landmarks.
2. **Feature extraction:** The landmark coordinates are normalized into a 63-feature representation using the project's preprocessing logic.
3. **Classification:** The trained MLP predicts the gesture class from the extracted features.
4. **Prediction stabilization:** Confidence checks and temporal smoothing help reduce inconsistent predictions.
5. **Automatic text composition:** Stable alphabet predictions are captured as characters. A timeout can automatically insert a space between words.
6. **Speech output:** The composed text can be read aloud using the text-to-speech feature.

### Recognition Classes

The trained classifier uses 29 classes:

- Letters: `A`–`Z`
- `space`: space control
- `del`: deletion control class
- `nothing`: no-sign class

The application's text-composition logic determines which predictions are automatically captured.

## 📁 Project Structure

```text
asl-landmark-recognition/
├── models/
│   ├── asl_landmark_mlp.pth
│   ├── asl_landmark_scaler.npz
│   └── asl_landmark_mlp_classes.json
├── hand_landmarker.task
├── extract_landmarks.py
├── train_landmark_model.py
├── live_landmark_recognition.py
├── live_landmark_text.py
├── asl_landmark_pyside6.py
├── requirements.txt
└── README.md
```

The `models/` directory contains the trained classifier, feature-scaling data, and class labels. The MediaPipe task model must be available at the project root.

Dataset files and other generated training artifacts may be stored separately and are not necessarily included in the repository.

## 🚀 Getting Started

### Prerequisites

- Windows 10/11 or another compatible desktop operating system
- Python 3.12 recommended for the development environment
- A working webcam
- A microphone is not required for the application's core recognition functionality

### 1. Clone the repository

```bash
git clone https://github.com/varaprasad4959/asl-landmark-recognition.git
cd asl-landmark-recognition
```

To use the latest PySide6 application from the feature branch:

```bash
git checkout feature/automatic-letter-capture
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install PySide6
```

If PySide6 is already listed in `requirements.txt`, the additional installation command is harmless.

### 4. Verify the required model assets

Before launching the application, make sure these files exist:

```text
hand_landmarker.task
models/asl_landmark_mlp.pth
models/asl_landmark_scaler.npz
models/asl_landmark_mlp_classes.json
```

The application cannot start without its required model assets. If any are missing from the cloned repository, obtain them from the project's development environment before running the application.

### 5. Launch the desktop application

```powershell
python asl_landmark_pyside6.py
```

Allow webcam access if Windows requests permission.

## ⌨️ Controls

| Key | Action |
|---|---|
| `Q` | Quit the application |
| `Esc` | Quit the application |
| `Space` | Insert a space |
| `Backspace` | Delete the last character |
| `V` | Speak the composed text |
| `X` | Clear the composed text |

The on-screen buttons can also be used for the text editing and speech controls.

## 📊 Model Evaluation

During development, the landmark classifier achieved approximately **99% accuracy on its held-out test split**.

This is an initial development result, not a guarantee of real-world performance. The result should not be interpreted as independently validated accuracy across different users, cameras, backgrounds, and signing conditions.

Further evaluation is needed to measure per-class performance, generalization to unseen signers, and robustness under varied lighting and hand orientations.

## ⚠️ Limitations

- The current application focuses on alphabet-level hand gesture recognition and associated text controls.
- It does not provide complete ASL sentence translation.
- ASL grammar, facial expressions, body movement, and continuous dynamic signing are outside the current scope.
- Recognition can vary with hand orientation, lighting, camera quality, and individual signing styles.
- The application requires the trained model files and MediaPipe task asset to run.

## 🔮 Future Improvements

- Add a formal evaluation pipeline with a confusion matrix and per-class metrics.
- Improve recognition robustness across users and environmental conditions.
- Refine text composition and recognition feedback.
- Package the application for easier installation on Windows.
- Explore dynamic gesture recognition and broader sign-language support.

## 📜 License

A license has not yet been specified for this project. Before distributing the source code, trained model, or dataset-derived assets, review the relevant third-party licenses and choose an appropriate project license.

## 👨‍💻 Author

**Vara Prasad G K**

B.Tech — Computer Science and Engineering

GitHub: [@varaprasad4959](https://github.com/varaprasad4959)

---

*Built as a computer vision and machine learning project exploring real-time hand landmark recognition and accessible desktop interaction.*
