
import json
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset


# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "models" / "asl_landmarks_dataset.npz"
LABELS_PATH = ROOT / "models" / "asl_landmark_class_names.json"

MODEL_PATH = ROOT / "models" / "asl_landmark_mlp.pth"
SCALER_PATH = ROOT / "models" / "asl_landmark_scaler.npz"
CLASSES_OUT = ROOT / "models" / "asl_landmark_mlp_classes.json"

BATCH_SIZE = 256
EPOCHS = 40
LEARNING_RATE = 0.001
RANDOM_STATE = 42

# First experiment: alphabet only.
INCLUDE_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# =====================================================
# LOAD DATA
# =====================================================

data = np.load(DATA_PATH)
X_all = data["X"].astype(np.float32)
y_all = data["y"].astype(np.int64)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    original_classes = json.load(f)

if isinstance(original_classes, dict):
    original_classes = [
        name for name, index in sorted(
            original_classes.items(), key=lambda item: item[1]
        )
    ]

# Map the original labels to class names.
sample_names = np.array(
    [original_classes[index] for index in y_all]
)

mask = np.isin(sample_names, INCLUDE_CLASSES)

X = X_all[mask]
names = sample_names[mask]

class_names = INCLUDE_CLASSES.copy()
name_to_index = {name: i for i, name in enumerate(class_names)}

y = np.array([name_to_index[name] for name in names], dtype=np.int64)

if len(X) == 0:
    raise RuntimeError("No alphabet samples found.")

print("Samples:", len(X))
print("Features per sample:", X.shape[1])
print("Classes:", len(class_names))


# =====================================================
# TRAIN / VALIDATION SPLIT
# =====================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

# Fit scaler on training data only to avoid leakage.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)

np.savez(
    SCALER_PATH,
    mean=scaler.mean_,
    scale=scaler.scale_
)


# =====================================================
# TORCH DATA LOADERS
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

train_dataset = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(y_train)
)

val_dataset = TensorDataset(
    torch.from_numpy(X_val),
    torch.from_numpy(y_val)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =====================================================
# MLP CLASSIFIER
# =====================================================

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


model = LandmarkMLP(X.shape[1], len(class_names)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

best_val_accuracy = 0.0


# =====================================================
# TRAIN
# =====================================================

for epoch in range(EPOCHS):

    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(features)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * labels.size(0)

        predictions = logits.argmax(dim=1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    train_loss = train_loss_sum / train_total
    train_accuracy = train_correct / train_total

    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.inference_mode():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            val_loss_sum += loss.item() * labels.size(0)

            predictions = logits.argmax(dim=1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss_sum / val_total
    val_accuracy = val_correct / val_total

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train acc: {train_accuracy * 100:.2f}% | "
        f"Val loss: {val_loss:.4f} | "
        f"Val acc: {val_accuracy * 100:.2f}%"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_size": X.shape[1],
                "num_classes": len(class_names),
                "class_names": class_names
            },
            MODEL_PATH
        )

        with open(CLASSES_OUT, "w", encoding="utf-8") as f:
            json.dump(class_names, f, indent=2)


# =====================================================
# FINAL VALIDATION REPORT
# =====================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_predictions = []
all_targets = []

with torch.inference_mode():
    for features, labels in val_loader:
        logits = model(features.to(device))
        predictions = logits.argmax(dim=1).cpu().numpy()

        all_predictions.extend(predictions)
        all_targets.extend(labels.numpy())

print("\nBest validation accuracy:", f"{best_val_accuracy * 100:.2f}%")
print("\nClassification report:")
print(
    classification_report(
        all_targets,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0
    )
)

print("Saved model:", MODEL_PATH)
print("Saved scaler:", SCALER_PATH)
print("Saved classes:", CLASSES_OUT)
