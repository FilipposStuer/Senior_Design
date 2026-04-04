"""
robot_vision.py - Waste sorter with homography-based coordinate conversion
"""

from __future__ import annotations
import cv2
import serial
import time
import logging
import threading
import numpy as np
from ultralytics import YOLO

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
MODEL_PATH        = "best.pt"
HOMOGRAPHY_PATH   = "homography.npy"
CONFIDENCE_THRESHOLD = 0.25
CAMERA_INDEX      = 0
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
SERIAL_PORT       = "COM5"
SERIAL_BAUDRATE   = 9600
SERIAL_TIMEOUT    = 30

DETECT_CONFIRM_FRAMES = 3
DISAPPEAR_THRESHOLD   = 5
PICK_TIMEOUT          = 20
IDLE_HOME_TIMEOUT     = 15

# Workspace check disabled — arm handles all positions
WS_X_MIN, WS_X_MAX = -999.0, 999.0
WS_Y_MIN, WS_Y_MAX = -999.0, 999.0

# Drop-off positions per class (real-world cm, adjust to your bins)
DROP_POSITIONS = {
    "plastic":   (-5.0, 1.0),
    "paper":     ( 0.0, 1.0),
    "cardboard": ( 5.0, 1.0),
}

# ----------------------------------------------------------------------
# WASTE CLASSES
# ----------------------------------------------------------------------
WASTE_CLASSES = {
    1: {"name": "cardboard", "recycle": True},
    4: {"name": "paper",     "recycle": True},
    5: {"name": "plastic",   "recycle": True},
    0: {"name": "biodegradable", "recycle": False},
    2: {"name": "glass",         "recycle": False},
    3: {"name": "metal",         "recycle": False},
}

CLASS_COLORS = {
    "plastic":   (255, 80,  80),
    "paper":     (200, 200, 255),
    "cardboard": (50,  165, 255),
    "default":   (128, 128, 128),
}

# ----------------------------------------------------------------------
# HOMOGRAPHY-BASED COORDINATE CONVERSION
# ----------------------------------------------------------------------
def load_homography(path):
    try:
        H = np.load(path)
        logging.info(f"Homography loaded from '{path}'")
        return H
    except Exception as e:
        logging.error(f"Could not load homography: {e}")
        return None

def pixel_to_cm(cx, cy, H):
    """Convert pixel (cx, cy) to real-world (x_cm, y_cm) using homography."""
    pt = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H)[0][0]
    x_cm = float(result[0])
    y_cm = float(result[1])
    return x_cm, y_cm

def in_workspace(x_cm, y_cm):
    return True  # no limit — Arduino handles out-of-range

# ----------------------------------------------------------------------
# SERIAL COMMUNICATION
# ----------------------------------------------------------------------
class ArmComm:
    def __init__(self, port, baudrate, timeout):
        self.port      = port
        self.baudrate  = baudrate
        self.timeout   = timeout
        self.conn      = None
        self.connected = False
        self._connect()

    def _connect(self):
        try:
            self.conn = serial.Serial(
                self.port, self.baudrate,
                timeout=1, write_timeout=5
            )
            time.sleep(2)
            self.connected = True
            logging.info(f"Connected to {self.port}")
        except Exception as e:
            logging.error(f"Serial error: {e}")

    def send(self, cmd):
        if not self.connected:
            return False
        try:
            self.conn.write((cmd + "\n").encode())
            return True
        except Exception as e:
            logging.error(f"Send failed: {e}")
            return False

    def pick(self, class_name, x_cm, y_cm, w, h):
        return self.send(f"PICK {class_name.upper()} {x_cm:.2f} {y_cm:.2f} {w} {h}")

    def move(self, x_cm, y_cm):
        return self.send(f"MOVE {x_cm:.2f} {y_cm:.2f}")

    def grip(self):
        return self.send("GRIP")

    def release(self):
        return self.send("RELEASE")

    def home(self):
        return self.send("HOME")

    def wait_done(self, timeout=None):
        deadline = time.time() + (timeout or self.timeout)
        while time.time() < deadline:
            if self.conn and self.conn.in_waiting:
                line = self.conn.readline().decode(errors='replace').strip().upper()
                if "DONE"  in line: return True
                if "ERROR" in line: return False
            time.sleep(0.05)
        return False

    def close(self):
        if self.conn:
            self.conn.close()

# ----------------------------------------------------------------------
# DETECTION
# ----------------------------------------------------------------------
def detect_best(frame, model):
    results = model.predict(
        frame, conf=CONFIDENCE_THRESHOLD,
        iou=0.45, verbose=False, stream=True
    )
    r = next(results, None)
    if r is None or r.boxes is None:
        return None

    best, best_conf = None, 0.0

    for xyxy, conf, c in zip(
        r.boxes.xyxy.cpu().numpy(),
        r.boxes.conf.cpu().numpy(),
        r.boxes.cls.cpu().numpy().astype(int)
    ):
        info = WASTE_CLASSES.get(int(c))
        x1, y1, x2, y2 = map(int, xyxy)
        conf_val = float(conf)

        if not info:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(frame, "UNKNOWN", (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            continue

        name    = info["name"]
        recycle = info["recycle"]
        color   = CLASS_COLORS.get(name, CLASS_COLORS["default"]) if recycle else (128,128,128)
        label   = f"{name} {conf_val:.2f}" if recycle else f"REJECT: {name}"

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, label, (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if recycle and conf_val > best_conf:
            best_conf = conf_val
            best = {
                "class_id":   int(c),
                "class_name": name,
                "center":     ((x1+x2)//2, (y1+y2)//2),
                "bbox":       (x2-x1, y2-y1),
                "confidence": conf_val,
            }
    return best

# ----------------------------------------------------------------------
# FEEDBACK LOOP — waits for object to disappear, then grips
# ----------------------------------------------------------------------
def feedback_loop(camera, model, arm, target_class, stop_event, done_event):
    disappear = 0
    while not stop_event.is_set():
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.1)
            continue
        results = model.predict(
            frame, conf=CONFIDENCE_THRESHOLD,
            verbose=False, stream=True
        )
        r = next(results, None)
        visible = False
        if r and r.boxes:
            for c in r.boxes.cls.cpu().numpy().astype(int):
                if int(c) == target_class:
                    visible = True
                    break
        if visible:
            disappear = 0
        else:
            disappear += 1
            if disappear >= DISAPPEAR_THRESHOLD:
                logging.info("Object gone — closing gripper")
                arm.grip()
                done_event.set()
                break
        time.sleep(0.1)

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger("WasteSorter")

    # Load YOLO model
    try:
        model = YOLO(MODEL_PATH)
        logger.info(f"Model loaded. Classes: {model.names}")
    except Exception as e:
        logger.error(f"Model load error: {e}")
        return

    # Load homography
    H = load_homography(HOMOGRAPHY_PATH)
    if H is None:
        logger.error("Cannot run without homography. Run calibrate_homography.py first.")
        return

    # Camera — try indices 0, 1, 2
    cap = None
    for idx in [0, 1, 2]:
        c = cv2.VideoCapture(idx)
        c.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if c.isOpened():
            ret, _ = c.read()
            if ret:
                cap = c
                logger.info(f"Camera opened at index {idx}")
                break
        c.release()

    if cap is None:
        logger.error("Could not open any camera (tried 0, 1, 2)")
        return

    # Arm
    arm = ArmComm(SERIAL_PORT, SERIAL_BAUDRATE, SERIAL_TIMEOUT)
    if not arm.connected:
        logger.warning("Arm not connected — commands will be simulated")

    # State
    busy            = False
    busy_start      = 0
    confirm         = 0
    last_detection  = None
    last_detect_time = time.time()
    stop_fb         = threading.Event()
    done_fb         = threading.Event()
    fb_thread       = None

    logger.info("System READY. Press Q to quit, H for home, G to grip, R to release.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Arm stuck timeout ---
        if busy and (time.time() - busy_start) > PICK_TIMEOUT:
            logger.error("Pick timeout — resetting arm")
            stop_fb.set()
            if fb_thread: fb_thread.join(timeout=1)
            arm.home()
            arm.wait_done(10)
            busy = False; confirm = 0
            stop_fb.clear(); done_fb.clear(); fb_thread = None

        # --- Feedback loop finished ---
        if busy and done_fb.is_set():
            logger.info("Object picked — moving to drop position")
            stop_fb.set()
            if fb_thread: fb_thread.join(timeout=2)
            arm.wait_done(10)

            # Move to drop-off bin for this class
            """   if last_detection:
                drop = DROP_POSITIONS.get(last_detection["class_name"], (0.0, 1.0))
                logger.info(f"Dropping {last_detection['class_name']} at {drop}")
                arm.move(drop[0], drop[1])
                arm.wait_done(10)
                arm.release()
                arm.wait_done(5) """

            arm.home()
            arm.wait_done(10)
            busy = False; confirm = 0
            stop_fb.clear(); done_fb.clear(); fb_thread = None

        # --- Detection (only when arm is free) ---
        if not busy:
            det = detect_best(frame, model)
            if det:
                confirm += 1
                last_detection = det
                last_detect_time = time.time()
                cv2.putText(frame,
                    f"Confirming {confirm}/{DETECT_CONFIRM_FRAMES}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

                if confirm >= DETECT_CONFIRM_FRAMES:
                    cx, cy = det["center"]
                    w,  h  = det["bbox"]
                    x_cm, y_cm = pixel_to_cm(cx, cy, H)

                    logger.info(
                        f"Picking {det['class_name']} at "
                        f"pixel=({cx},{cy}) → ({x_cm:.2f}, {y_cm:.2f}) cm"
                    )

                    # Workspace check
                    if not in_workspace(x_cm, y_cm):
                        logger.warning(
                            f"({x_cm:.2f}, {y_cm:.2f}) outside workspace — skipping"
                        )
                        confirm = 0
                    else:
                        if arm.pick(det["class_name"], x_cm, y_cm, w, h):
                            if arm.wait_done(15):
                                busy       = True
                                busy_start = time.time()
                                confirm    = 0
                                stop_fb.clear(); done_fb.clear()
                                fb_thread = threading.Thread(
                                    target=feedback_loop,
                                    args=(cap, model, arm,
                                          det["class_id"], stop_fb, done_fb),
                                    daemon=True
                                )
                                fb_thread.start()
                            else:
                                logger.error("Arm did not reach pick position")
                                arm.home(); arm.wait_done(10)
                        else:
                            logger.error("Failed to send PICK command")
            else:
                confirm = 0

        # --- Auto-home on idle ---
        if not busy and (time.time() - last_detect_time) > IDLE_HOME_TIMEOUT:
            logger.info("Idle timeout — sending HOME")
            arm.home(); arm.wait_done(10)
            last_detect_time = time.time()

        # --- HUD ---
        status = "BUSY" if busy else "READY"
        color  = (0,165,255) if busy else (0,255,0)
        cv2.putText(frame, f"Status: {status}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if last_detection and not busy:
            cv2.putText(frame,
                f"Target: {last_detection['class_name']} "
                f"{last_detection['confidence']:.2f}",
                (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow("Waste Sorter", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if not busy:
            if   key == ord('h'): arm.home();    arm.wait_done()
            elif key == ord('g'): arm.grip()
            elif key == ord('r'): arm.release()

    cap.release()
    cv2.destroyAllWindows()
    arm.close()

if __name__ == "__main__":
    main()