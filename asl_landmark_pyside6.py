import sys
import json
import time
import subprocess
import threading
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QTextEdit,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QMessageBox, QSizePolicy
)


# ============================================================
# SETTINGS / PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "asl_landmark_mlp.pth"
SCALER_PATH = MODEL_DIR / "asl_landmark_scaler.npz"
CLASSES_PATH = MODEL_DIR / "asl_landmark_mlp_classes.json"
HAND_MODEL_PATH = BASE_DIR / "hand_landmarker.task"

MIRROR_WEBCAM = True
NO_HAND_SPACE_TIMEOUT = 2.0
CONFIDENCE_THRESHOLD = 0.70
SMOOTHING_FRAMES = 10
STABLE_FRAMES_REQUIRED = 15
LETTER_CAPTURE_COOLDOWN = 0.8
REARM_INVALID_FRAMES = 3

ACCENT = "#39D6B4"
ACCENT_DARK = "#173E39"
BG = "#111719"
PANEL = "#1B2427"
PANEL_ALT = "#202C30"
TEXT = "#E8F1F2"
MUTED = "#93A6AA"
WARNING = "#E9B96E"
RED = "#F07878"


# ============================================================
# MODEL
# ============================================================
for required_path in (MODEL_PATH, SCALER_PATH, CLASSES_PATH, HAND_MODEL_PATH):
    if not required_path.exists():
        raise FileNotFoundError(
            f"Required project file is missing:\n{required_path}\n\n"
            "Keep the models folder and hand_landmarker.task beside this script."
        )

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
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)


with open(CLASSES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

scaler_data = np.load(SCALER_PATH)
scaler_mean = scaler_data["mean"].astype(np.float32)
scaler_scale = scaler_data["scale"].astype(np.float32)
scaler_scale[scaler_scale == 0] = 1.0

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model = LandmarkMLP(checkpoint["input_size"], checkpoint["num_classes"])
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()
print("MLP landmark model loaded successfully.")


def extract_features(hand_landmarks):
    points = np.array([[p.x, p.y, p.z] for p in hand_landmarks], dtype=np.float32)
    points -= points[0]  # wrist-relative normalization
    scale = np.max(np.abs(points))
    if scale < 1e-6:
        return None
    points /= scale
    return points.flatten()


# ============================================================
# CAMERA / RECOGNITION WORKER
# ============================================================
class RecognitionWorker(QThread):
    frame_ready = Signal(QImage)
    state_changed = Signal(dict)
    error_occurred = Signal(str)
    finished_text = Signal(str)

    def __init__(self):
        super().__init__()
        self._stop_requested = False
        self._lock = threading.Lock()
        self.typed_text = ""
        self.word_started = False
        self.no_hand_start_time = None
        self.probability_history = deque(maxlen=SMOOTHING_FRAMES)
        self.current_prediction = "No hand"
        self.current_confidence = 0.0
        self.stable_letter = ""
        self.stable_count = 0
        self.capture_armed = True
        self.last_captured_letter = ""
        self.last_capture_time = 0.0
        self.invalid_prediction_count = 0
        self.fps = 0.0

    def stop(self):
        self._stop_requested = True

    def set_text(self, value):
        with self._lock:
            self.typed_text = value

    def get_text(self):
        with self._lock:
            return self.typed_text

    def run(self):
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            camera.release()
            self.error_occurred.emit(
                "Cannot open webcam. Close other camera apps or try changing camera index 0 to 1."
            )
            return

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        last_timestamp_ms = 0
        previous_time = time.time()

        try:
            with HandLandmarker.create_from_options(options) as landmarker:
                while not self._stop_requested:
                    success, frame = camera.read()
                    if not success:
                        self.error_occurred.emit("Unable to read a frame from the webcam.")
                        break

                    if MIRROR_WEBCAM:
                        frame = cv2.flip(frame, 1)

                    height, width = frame.shape[:2]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    timestamp_ms = max(int(time.monotonic() * 1000), last_timestamp_ms + 1)
                    last_timestamp_ms = timestamp_ms
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)

                    # Automatic space after hand disappears, unchanged from baseline.
                    if result.hand_landmarks:
                        self.no_hand_start_time = None
                    elif self.word_started:
                        if self.no_hand_start_time is None:
                            self.no_hand_start_time = time.time()
                        elif time.time() - self.no_hand_start_time >= NO_HAND_SPACE_TIMEOUT:
                            with self._lock:
                                if self.typed_text and not self.typed_text.endswith(" "):
                                    self.typed_text += " "
                                    print("Automatic space inserted.")
                                    print("Text:", self.typed_text)
                            self.word_started = False
                            self.no_hand_start_time = None

                    if result.hand_landmarks:
                        hand = result.hand_landmarks[0]  # preserve first-hand behavior
                        for point in hand:
                            px, py = int(point.x * width), int(point.y * height)
                            if 0 <= px < width and 0 <= py < height:
                                cv2.circle(frame, (px, py), 4, (75, 225, 190), -1)

                        features = extract_features(hand)
                        if features is not None:
                            features = (features - scaler_mean) / scaler_scale
                            input_tensor = torch.from_numpy(
                                features.astype(np.float32)
                            ).unsqueeze(0).to(device)

                            with torch.no_grad():
                                logits = model(input_tensor)
                                probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

                            self.probability_history.append(probabilities)
                            if len(self.probability_history) >= 5:
                                average_probabilities = np.mean(self.probability_history, axis=0)
                                index = int(np.argmax(average_probabilities))
                                self.current_confidence = float(average_probabilities[index])
                                self.current_prediction = (
                                    class_names[index]
                                    if self.current_confidence >= CONFIDENCE_THRESHOLD
                                    else "Uncertain"
                                )
                            else:
                                self.current_prediction = "Uncertain"
                                self.current_confidence = 0.0
                        else:
                            self.current_prediction = "Uncertain"
                            self.current_confidence = 0.0
                            self.probability_history.clear()
                    else:
                        self.current_prediction = "No hand"
                        self.current_confidence = 0.0
                        self.probability_history.clear()

                    is_valid_letter = (
                        isinstance(self.current_prediction, str)
                        and len(self.current_prediction) == 1
                        and self.current_prediction.isalpha()
                        and self.current_prediction.upper() in class_names
                        and self.current_confidence >= CONFIDENCE_THRESHOLD
                    )

                    if is_valid_letter:
                        self.invalid_prediction_count = 0
                        if self.current_prediction == self.stable_letter:
                            self.stable_count += 1
                        else:
                            self.stable_letter = self.current_prediction
                            self.stable_count = 1

                        cooldown_finished = time.time() - self.last_capture_time >= LETTER_CAPTURE_COOLDOWN
                        if self.stable_count >= STABLE_FRAMES_REQUIRED and cooldown_finished:
                            if self.capture_armed or self.current_prediction != self.last_captured_letter:
                                with self._lock:
                                    self.typed_text += self.current_prediction
                                    composed = self.typed_text
                                self.word_started = True
                                self.last_captured_letter = self.current_prediction
                                self.last_capture_time = time.time()
                                self.capture_armed = False
                                print("Captured:", self.current_prediction)
                                print("Text:", composed)
                    else:
                        self.stable_letter = ""
                        self.stable_count = 0
                        self.invalid_prediction_count += 1
                        if self.invalid_prediction_count >= REARM_INVALID_FRAMES:
                            self.capture_armed = True
                            self.last_captured_letter = ""

                    now = time.time()
                    self.fps = 1.0 / max(now - previous_time, 1e-6)
                    previous_time = now

                    # Camera overlay stays minimal; desktop UI holds the details.
                    cv2.rectangle(frame, (0, 0), (width, 54), (20, 30, 32), -1)
                    cv2.putText(
                        frame, f"Prediction: {self.current_prediction}",
                        (16, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (75, 225, 190), 2, cv2.LINE_AA
                    )
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = QImage(
                        rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0],
                        rgb_frame.strides[0], QImage.Format.Format_RGB888
                    ).copy()
                    self.frame_ready.emit(image)

                    with self._lock:
                        current_text = self.typed_text
                    self.state_changed.emit({
                        "prediction": self.current_prediction,
                        "confidence": self.current_confidence,
                        "text": current_text,
                        "stable_count": self.stable_count,
                        "capture_armed": self.capture_armed,
                        "fps": self.fps,
                        "has_hand": bool(result.hand_landmarks),
                    })
        except Exception as exc:
            self.error_occurred.emit(f"Recognition error: {exc}")
        finally:
            camera.release()
            self.finished_text.emit(self.get_text())


# ============================================================
# MAIN DESKTOP UI
# ============================================================
class MainWindow(QMainWindow):
    voice_status = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASL Landmark Recognition")
        self.resize(1280, 820)
        self.setMinimumSize(960, 650)
        self.worker = None
        self.current_text = ""
        self.voice_process = None

        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; }}
            QLabel {{ color: {TEXT}; }}
            QFrame#panel {{ background: {PANEL}; border: 1px solid #2B3A3E; border-radius: 16px; }}
            QLabel#eyebrow {{ color: {ACCENT}; font-size: 11px; font-weight: 700; letter-spacing: 2px; }}
            QLabel#heading {{ font-size: 25px; font-weight: 700; }}
            QLabel#subheading {{ color: {MUTED}; font-size: 12px; }}
            QLabel#metric {{ font-size: 24px; font-weight: 700; }}
            QLabel#muted {{ color: {MUTED}; }}
            QTextEdit {{ background: #101719; border: 1px solid #34464A; border-radius: 12px;
                         padding: 12px; color: {TEXT}; font-size: 18px; selection-background-color: {ACCENT_DARK}; }}
            QPushButton {{ background: {PANEL_ALT}; color: {TEXT}; border: 1px solid #3A4B4F;
                           border-radius: 10px; padding: 11px 16px; font-weight: 600; }}
            QPushButton:hover {{ border-color: {ACCENT}; background: #26373A; }}
            QPushButton#primary {{ background: {ACCENT}; color: #08211D; border: none; }}
            QPushButton#primary:hover {{ background: #61E6CA; }}
            QPushButton#danger {{ color: #FFDADA; border-color: #734646; }}
        """)

        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(24, 20, 24, 18)
        main.setSpacing(16)

        # Header
        header = QHBoxLayout()
        brand_col = QVBoxLayout()
        eyebrow = QLabel("COMPUTER VISION  /  ACCESSIBILITY")
        eyebrow.setObjectName("eyebrow")
        title = QLabel("ASL Landmark Recognition")
        title.setObjectName("heading")
        subtitle = QLabel("Real-time hand signs · automatic text composition · voice output")
        subtitle.setObjectName("subheading")
        brand_col.addWidget(eyebrow)
        brand_col.addWidget(title)
        brand_col.addWidget(subtitle)
        header.addLayout(brand_col)
        header.addStretch()
        self.camera_badge = QLabel("●  CAMERA STARTING")
        self.camera_badge.setStyleSheet(f"color: {WARNING}; background: {PANEL}; padding: 10px 14px; border-radius: 10px;")
        header.addWidget(self.camera_badge, alignment=Qt.AlignmentFlag.AlignVCenter)
        main.addLayout(header)

        # Main content row
        content = QHBoxLayout()
        content.setSpacing(16)
        main.addLayout(content, stretch=1)

        left_panel = self.panel()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(12)
        cam_header = QHBoxLayout()
        cam_title = QLabel("LIVE CAMERA")
        cam_title.setObjectName("eyebrow")
        self.fps_label = QLabel("— FPS")
        self.fps_label.setObjectName("muted")
        cam_header.addWidget(cam_title)
        cam_header.addStretch()
        cam_header.addWidget(self.fps_label)
        left_layout.addLayout(cam_header)

        self.video_label = QLabel("Starting camera…")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(520, 360)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet("background: #090E10; border: 1px solid #2D4145; border-radius: 12px; color: #7C9297;")
        left_layout.addWidget(self.video_label, stretch=1)

        camera_note = QLabel("Keep one hand visible. Hold a sign steady until it is captured.")
        camera_note.setObjectName("muted")
        camera_note.setWordWrap(True)
        left_layout.addWidget(camera_note)
        content.addWidget(left_panel, stretch=6)

        right_col = QVBoxLayout()
        right_col.setSpacing(16)
        content.addLayout(right_col, stretch=4)

        status_panel = self.panel()
        status_layout = QVBoxLayout(status_panel)
        status_layout.setContentsMargins(18, 16, 18, 16)
        status_layout.setSpacing(12)
        status_title = QLabel("RECOGNITION STATUS")
        status_title.setObjectName("eyebrow")
        status_layout.addWidget(status_title)

        pred_row = QHBoxLayout()
        pred_text_col = QVBoxLayout()
        pred_caption = QLabel("CURRENT PREDICTION")
        pred_caption.setObjectName("muted")
        self.prediction_label = QLabel("No hand")
        self.prediction_label.setObjectName("metric")
        pred_text_col.addWidget(pred_caption)
        pred_text_col.addWidget(self.prediction_label)
        pred_row.addLayout(pred_text_col)
        pred_row.addStretch()
        self.confidence_label = QLabel("0%")
        self.confidence_label.setObjectName("metric")
        self.confidence_label.setStyleSheet(f"color: {ACCENT};")
        pred_row.addWidget(self.confidence_label, alignment=Qt.AlignmentFlag.AlignVCenter)
        status_layout.addLayout(pred_row)

        self.confidence_bar = QFrame()
        self.confidence_bar.setFixedHeight(6)
        self.confidence_bar.setStyleSheet("background: #304044; border-radius: 3px;")
        status_layout.addWidget(self.confidence_bar)

        self.capture_label = QLabel("Waiting for a stable sign")
        self.capture_label.setWordWrap(True)
        self.capture_label.setStyleSheet(f"color: {MUTED}; background: #121B1D; padding: 10px; border-radius: 8px;")
        status_layout.addWidget(self.capture_label)
        right_col.addWidget(status_panel)

        text_panel = self.panel()
        text_layout = QVBoxLayout(text_panel)
        text_layout.setContentsMargins(18, 16, 18, 16)
        text_layout.setSpacing(10)
        text_head = QHBoxLayout()
        text_title = QLabel("COMPOSED TEXT")
        text_title.setObjectName("eyebrow")
        self.char_count = QLabel("0 characters")
        self.char_count.setObjectName("muted")
        text_head.addWidget(text_title)
        text_head.addStretch()
        text_head.addWidget(self.char_count)
        text_layout.addLayout(text_head)

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Your captured letters will appear here…")
        self.text_edit.setMinimumHeight(135)
        # The composed text is generated by sign recognition; users can select/copy it,
        # but cannot type into it manually.
        self.text_edit.setReadOnly(True)
        self.text_edit.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.text_edit.textChanged.connect(self.on_text_edited)
        text_layout.addWidget(self.text_edit, stretch=1)

        actions = QGridLayout()
        actions.setSpacing(8)
        self.speak_button = QPushButton("▶  Speak text")
        self.speak_button.setObjectName("primary")
        self.speak_button.clicked.connect(self.speak_text)
        self.backspace_button = QPushButton("⌫  Delete")
        self.backspace_button.clicked.connect(self.delete_character)
        self.space_button = QPushButton("＋  Space")
        self.space_button.clicked.connect(self.add_space)
        self.clear_button = QPushButton("Clear text")
        self.clear_button.setObjectName("danger")
        self.clear_button.clicked.connect(self.clear_text)
        actions.addWidget(self.speak_button, 0, 0, 1, 2)
        actions.addWidget(self.backspace_button, 1, 0)
        actions.addWidget(self.space_button, 1, 1)
        actions.addWidget(self.clear_button, 2, 0, 1, 2)
        text_layout.addLayout(actions)
        right_col.addWidget(text_panel, stretch=1)

        footer = QHBoxLayout()
        self.system_status = QLabel("Initializing recognition model…")
        self.system_status.setObjectName("muted")
        self.voice_status.connect(self.system_status.setText)
        footer.addWidget(self.system_status)
        footer.addStretch()
        help_label = QLabel("ESC / Q  Quit")
        help_label.setObjectName("muted")
        footer.addWidget(help_label)
        main.addLayout(footer)

        # Application-wide shortcuts work even when a button or the text display has focus.
        # QShortcut is used instead of relying only on QMainWindow.keyPressEvent.
        self._shortcuts = []
        shortcut_actions = [
            ("Esc", self.close),
            ("Q", self.close),
            ("Space", self.add_space),
            ("Backspace", self.delete_character),
            ("V", self.speak_text),
            ("X", self.clear_text),
        ]
        for sequence, callback in shortcut_actions:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)

        self.start_worker()

    def panel(self):
        frame = QFrame()
        frame.setObjectName("panel")
        return frame

    def start_worker(self):
        self.worker = RecognitionWorker()
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.state_changed.connect(self.update_state)
        self.worker.error_occurred.connect(self.show_worker_error)
        self.worker.finished_text.connect(self.on_worker_finished)
        self.worker.start()
        self.system_status.setText("Recognition model loaded · waiting for camera frames")

    def update_frame(self, image):
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        if "RUNNING" not in self.camera_badge.text():
            self.camera_badge.setText("●  CAMERA RUNNING")
            self.camera_badge.setStyleSheet(f"color: {ACCENT}; background: {PANEL}; padding: 10px 14px; border-radius: 10px;")

    def update_state(self, state):
        self.prediction_label.setText(state["prediction"])
        confidence = max(0, min(100, int(state["confidence"] * 100)))
        self.confidence_label.setText(f"{confidence}%")
        self.fps_label.setText(f'{state["fps"]:.1f} FPS')
        if state["capture_armed"]:
            self.capture_label.setText(f'Stable frames: {state["stable_count"]} / {STABLE_FRAMES_REQUIRED} · Ready for next sign')
            self.capture_label.setStyleSheet(f"color: {ACCENT}; background: #12221F; padding: 10px; border-radius: 8px;")
        else:
            self.capture_label.setText(f'Stable frames: {state["stable_count"]} / {STABLE_FRAMES_REQUIRED} · Captured — change or release sign')
            self.capture_label.setStyleSheet(f"color: {WARNING}; background: #272117; padding: 10px; border-radius: 8px;")
        if state["has_hand"]:
            self.camera_badge.setText("●  HAND DETECTED")
            self.camera_badge.setStyleSheet(f"color: {ACCENT}; background: {PANEL}; padding: 10px 14px; border-radius: 10px;")
        else:
            self.camera_badge.setText("●  CAMERA RUNNING")
            self.camera_badge.setStyleSheet(f"color: {MUTED}; background: {PANEL}; padding: 10px 14px; border-radius: 10px;")

        if self.text_edit.toPlainText() != state["text"]:
            cursor = self.text_edit.textCursor()
            old_position = cursor.position()
            self.text_edit.blockSignals(True)
            self.text_edit.setPlainText(state["text"])
            cursor = self.text_edit.textCursor()
            cursor.setPosition(min(old_position, len(state["text"])))
            self.text_edit.setTextCursor(cursor)
            self.text_edit.blockSignals(False)
        self.current_text = state["text"]
        self.char_count.setText(f'{len(state["text"])} characters')
        self.confidence_bar.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {ACCENT}, stop:{confidence/100:.2f} {ACCENT}, stop:{confidence/100:.2f} #304044, stop:1 #304044); border-radius: 3px;"
        )
        self.system_status.setText("Live recognition active · automatic capture enabled")

    def on_text_edited(self):
        if self.worker is None:
            return
        value = self.text_edit.toPlainText()
        self.current_text = value
        self.char_count.setText(f"{len(value)} characters")
        self.worker.set_text(value)

    def add_space(self):
        value = self.text_edit.toPlainText()
        if value and not value.endswith(" "):
            self.text_edit.setPlainText(value + " ")
            self.text_edit.setFocus()

    def delete_character(self):
        value = self.text_edit.toPlainText()
        if value:
            self.text_edit.setPlainText(value[:-1])
            self.text_edit.setFocus()

    def clear_text(self):
        self.text_edit.clear()
        if self.worker:
            # Prevent immediate capture of a held sign after clearing.
            self.worker.capture_armed = False
            self.worker.last_captured_letter = self.worker.stable_letter
        self.text_edit.setFocus()

    def speak_text(self):
        text_to_speak = self.text_edit.toPlainText().strip()
        if not text_to_speak:
            self.system_status.setText("Nothing to speak — capture some signs first")
            return
        if self.voice_process is not None and self.voice_process.poll() is None:
            self.system_status.setText("Voice output is already running")
            return

        def run_speech(text):
            speech_code = (
                "import sys, pyttsx3; "
                "engine = pyttsx3.init(); "
                "engine.setProperty('rate', 145); "
                "engine.setProperty('volume', 1.0); "
                "engine.say(sys.argv[1]); "
                "engine.runAndWait(); "
                "engine.stop()"
            )
            self.voice_process = subprocess.Popen(
                [sys.executable, "-c", speech_code, text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            return_code = self.voice_process.wait()
            if return_code == 0:
                self.voice_status.emit("Voice output completed")
            else:
                error_text = self.voice_process.stderr.read() if self.voice_process.stderr else ""
                self.voice_status.emit("Voice output failed — " + (error_text.strip()[:180] or "check pyttsx3 installation"))

        self.system_status.setText("Speaking composed text…")
        threading.Thread(target=run_speech, args=(text_to_speak,), daemon=True).start()

    def show_worker_error(self, message):
        self.camera_badge.setText("●  CAMERA / MODEL ERROR")
        self.camera_badge.setStyleSheet(f"color: {RED}; background: {PANEL}; padding: 10px 14px; border-radius: 10px;")
        self.system_status.setText(message)
        QMessageBox.warning(self, "ASL Recognition", message)

    def on_worker_finished(self, final_text):
        self.current_text = final_text
        self.system_status.setText("Recognition stopped")

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self.close()
        elif event.key() == Qt.Key.Key_Space:
            self.add_space()
        elif event.key() == Qt.Key.Key_Backspace:
            self.delete_character()
        elif event.key() == Qt.Key.Key_V:
            self.speak_text()
        elif event.key() == Qt.Key.Key_X:
            self.clear_text()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        if self.voice_process is not None and self.voice_process.poll() is None:
            self.voice_process.terminate()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
