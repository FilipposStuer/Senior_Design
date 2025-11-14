import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

IMG_SIZE = (510, 600)  
label_map = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
MODEL_PATH = "trash_dtree_model.xml" 


def load_model(model_path):
    """Load a trained OpenCV DTrees model from file."""
    model = cv2.ml.DTrees_load(model_path)
    if model is None:
        raise RuntimeError("Failed to load model from: " + model_path)
    return model


model = None
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)


def predict_image(img_path):
    """Predict class index for a single image file."""

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    img = cv2.resize(img, IMG_SIZE)

    feature_dim = IMG_SIZE[0] * IMG_SIZE[1]
    sample = img.reshape(1, feature_dim).astype(np.float32)

    ret, result = model.predict(sample)
    predicted_label = int(result[0, 0])
    return predicted_label


def browse_file():
    """Open file dialog and put selected path into the entry widget."""
    file_path = filedialog.askopenfilename(
        title="Select image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp"), ("All files", "*.*")]
    )
    if file_path:
        entry_path.delete(0, tk.END)
        entry_path.insert(0, file_path)


def on_predict():
    """Handle Predict button click."""
    if model is None:
        messagebox.showerror("Error", "Model is not loaded.")
        return

    img_path = entry_path.get().strip()
    if not img_path:
        messagebox.showwarning("Warning", "Please enter an image path.")
        return

    try:
        label_idx = predict_image(img_path)
        class_name = label_map[label_idx]
        label_result.config(text=f"Result: {class_name}  (label {label_idx})")
    except Exception as e:
        messagebox.showerror("Error", str(e))


root = tk.Tk()
root.title("Trash Classification Demo")

frame_path = tk.Frame(root)
frame_path.pack(padx=10, pady=10, fill="x")

lbl_path = tk.Label(frame_path, text="Image path:")
lbl_path.pack(side="left")

entry_path = tk.Entry(frame_path, width=60)
entry_path.pack(side="left", padx=5)

btn_browse = tk.Button(frame_path, text="Browse...", command=browse_file)
btn_browse.pack(side="left")

btn_predict = tk.Button(root, text="Predict", command=on_predict)
btn_predict.pack(pady=5)

label_result = tk.Label(root, text="Result: ", font=("Arial", 12))
label_result.pack(pady=10)

root.mainloop()
