import os
import cv2
import numpy as np
import joblib
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox

# ----------------- 1. Model path -----------------
# Load model from the same folder as this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "trash_rf_model.pkl")

YOLO_IMG_SIZE = 640
label_map = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------- 2. Load YOLO once -----------------
# YOLOv8 small model, used only as feature extractor
yolo = YOLO("yolov8n.pt")
yolo_model = yolo.model.to(device)
yolo_model.eval()


@torch.no_grad()
def extract_yolo_features_bgr(img):
    """
    Extract YOLO backbone features from a BGR image.
    Returns a 1D feature vector (numpy array).
    """
    # BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize to YOLO input size
    img_resized = cv2.resize(img_rgb, (YOLO_IMG_SIZE, YOLO_IMG_SIZE))

    # HWC -> CHW, normalize to [0, 1]
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)  # [1, 3, H, W]

    x = tensor
    feats = []
    # Use first 10 layers of YOLO backbone
    for m in yolo_model.model[:10]:
        x = m(x)
        feats.append(x)

    pooled = []
    # Global average pooling for each feature map
    for f in feats:
        v = F.adaptive_avg_pool2d(f, 1)  # [1, C, 1, 1]
        v = v.view(1, -1)                # [1, C]
        pooled.append(v)

    feat_vec = torch.cat(pooled, dim=1)  # [1, total_C]
    return feat_vec.cpu().numpy().squeeze()  # (total_C,)


def load_model(path):
    """
    Load RandomForest model from .pkl file with joblib.
    Supports both:
      joblib.dump(rf, path)
      joblib.dump({"rf": rf, ...}, path)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    data = joblib.load(path)
    if isinstance(data, dict) and "rf" in data:
        return data["rf"]
    return data


# Try to load RF model at start
try:
    print("Loading model from:", MODEL_PATH)
    rf_model = load_model(MODEL_PATH)
    print("RandomForest model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    rf_model = None


def predict_image(img_path):
    """
    Predict label index for a single image file path.
    """
    if rf_model is None:
        raise RuntimeError("Model is not loaded.")

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError("Cannot read image: " + img_path)

    feat = extract_yolo_features_bgr(img)
    feat = feat.reshape(1, -1)
    label_idx = int(rf_model.predict(feat)[0])
    return label_idx


# ----------------- 3. GUI Functions -----------------

def browse_file():
    """
    Open file dialog and put selected path into the entry widget.
    """
    file_path = filedialog.askopenfilename(
        title="Select image",
        initialdir=".",  # or change to your default folder
        filetypes=[
            ("Image files", "*.jpg;*.jpeg;*.png;*.bmp"),
            ("All files", "*.*")
        ]
    )
    if file_path:
        entry_path.delete(0, tk.END)
        entry_path.insert(0, file_path)


def on_predict():
    """
    Handle Predict button click.
    """
    if rf_model is None:
        messagebox.showerror("Error", "Model is not loaded.")
        return

    img_path = entry_path.get().strip()
    if not img_path:
        messagebox.showwarning("Warning", "Please choose an image.")
        return

    try:
        label_idx = predict_image(img_path)
        class_name = label_map[label_idx]
        label_result.config(text=f"Result: {class_name} (label {label_idx})")
    except Exception as e:
        messagebox.showerror("Error", str(e))


# ----------------- 4. Build GUI -----------------

root = tk.Tk()
root.title("Trash Classification (YOLO + RandomForest)")

frame_path = tk.Frame(root)
frame_path.pack(padx=10, pady=10)

tk.Label(frame_path, text="Image path:").pack(side="left")

entry_path = tk.Entry(frame_path, width=60)
entry_path.pack(side="left", padx=5)

tk.Button(frame_path, text="Browse...", command=browse_file).pack(side="left")

tk.Button(root, text="Predict", command=on_predict).pack(pady=5)

label_result = tk.Label(root, text="Result: ", font=("Arial", 14))
label_result.pack(pady=10)

root.mainloop()
