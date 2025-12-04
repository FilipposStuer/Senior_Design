import os
import glob
import cv2
import numpy as np

import torch
import torch.nn.functional as F
from ultralytics import YOLO

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path

# 1. Dataset settings

DATASET_PATH = r"D:\dataset-resized\dataset-resized"

class_folders = {
    "cardboard": 0,
    "glass": 1,
    "metal": 2,
    "paper": 3,
    "plastic": 4,
    "trash": 5,
}

MAX_PER_CLASS = 200          # Max images per class
YOLO_IMG_SIZE = 640          # YOLO input size

# 2. Load YOLO model (feature extractor)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Use small YOLOv8 model
yolo = YOLO("yolov8n.pt")
model = yolo.model.to(device)
model.eval()                 # Inference mode, no training


@torch.no_grad()
def extract_yolo_features_bgr(img_bgr):
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize to YOLO input
    img_resized = cv2.resize(img_rgb, (YOLO_IMG_SIZE, YOLO_IMG_SIZE))

    # Convert to tensor [1, 3, H, W]
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)

    # USE BACKBONE ONLY
    # YOLOv8 backbone = first 10 layers of model.model
    x = tensor
    feats = []
    for m in model.model[:10]: 
        x = m(x)
        feats.append(x)        # always tensors

    # Global pooling for each feature map
    pooled_vecs = []
    for f in feats:
        v = F.adaptive_avg_pool2d(f, 1)  # [1, C, 1, 1]
        v = v.view(1, -1)                # [1, C]
        pooled_vecs.append(v)

    # Concatenate to a single feature vector
    feat_vec = torch.cat(pooled_vecs, dim=1)
    return feat_vec.cpu().numpy().squeeze()


# 3. Loop dataset and extract features

features = []
labels = []

for folder_name, label in class_folders.items():
    folder_path = os.path.join(DATASET_PATH, folder_name)
    image_files = glob.glob(os.path.join(folder_path, "*.jpg"))
    image_files = image_files[:MAX_PER_CLASS]

    print(f"{folder_name}: {len(image_files)} images")

    for img_path in image_files:
        img = cv2.imread(img_path)  # Read color image
        if img is None:
            print("Warning: cannot read", img_path)
            continue

        # Extract YOLO feature vector
        feat_vec = extract_yolo_features_bgr(img)
        features.append(feat_vec)
        labels.append(label)

features = np.array(features, dtype=np.float32)  
labels = np.array(labels, dtype=np.int32)       

print("Total samples:", features.shape[0])
print("Feature dim:", features.shape[1])

# 4. Train Random Forest

X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels,
)

rf = RandomForestClassifier(
    n_estimators=300,   # Number of trees
    max_depth=20,      # Max depth of each tree
    n_jobs=-1,         # Use all CPU cores
    random_state=42,
)

print("Training Random Forest...")
rf.fit(X_train, y_train)

# 5. Evaluate

y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Test accuracy:", acc)
print("Classification report:")
print(classification_report(y_test, y_pred, digits=4))


print("WORKDIR:", os.getcwd())

save_dir = Path(r"D:\AI detect")
save_dir.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = save_dir / "trash_rf_model.pkl"
print("Saving model to:", MODEL_SAVE_PATH)

try:
    joblib.dump(
        {
            "rf": rf,
            "class_folders": class_folders,
        },
        MODEL_SAVE_PATH,
    )
    print("os.path.exists:", os.path.exists(MODEL_SAVE_PATH))
    print("File size (bytes):", MODEL_SAVE_PATH.stat().st_size)
    print("Model saved OK.")
except Exception as e:
    print("Error when saving model:", repr(e))