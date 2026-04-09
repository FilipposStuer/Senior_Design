"""
robot_vision.py - Waste sorter with homography-based coordinate conversion

This script uses a YOLO model to detect recyclable waste, converts pixel coordinates
to real-world coordinates using a homography matrix, and commands a robot arm (via serial)
to pick and sort the items.
"""

# ----------------------------------------------------------------------
# IMPORT LIBRARIES
# ----------------------------------------------------------------------

from __future__ import annotations    # Enables forward references in type hints

import cv2                     # OpenCV for camera capture, image processing, drawing
import serial                  # PySerial for serial communication with Arduino
import time                    # For delays and timeout handling
import logging                 # For logging messages (info, warnings, errors)
import threading               # For running the feedback loop in a separate thread
import numpy as np             # For numerical operations, especially homography matrices
from ultralytics import YOLO   # YOLO object detection model

# ----------------------------------------------------------------------
# CONFIGURATION PARAMETERS
# ----------------------------------------------------------------------

MODEL_PATH = "best.pt"                 # Path to the trained YOLO model file
HOMOGRAPHY_PATH = "homography.npy"     # Path to the saved homography matrix (from calibration)
CONFIDENCE_THRESHOLD = 0.25            # Minimum confidence score to consider a detection
CAMERA_INDEX = 0                       # Camera device index (0 = built-in, 1 = USB)
FRAME_WIDTH, FRAME_HEIGHT = 640, 480   # Camera resolution (width, height)
SERIAL_PORT = "COM5"                   # Serial port where Arduino is connected
SERIAL_BAUDRATE = 9600                 # Baud rate (must match Arduino firmware)
SERIAL_TIMEOUT = 30                    # Default timeout in seconds for serial responses

DETECT_CONFIRM_FRAMES = 3              # Number of consecutive frames with the same detection to confirm
DISAPPEAR_THRESHOLD = 5                # How many frames the object must be missing before pick is considered complete
PICK_TIMEOUT = 60                      # Seconds after which a stuck pick operation is aborted
IDLE_HOME_TIMEOUT = 15                 # Seconds with no detection before sending HOME command

# Workspace check disabled - arm will attempt any position (Arduino handles out-of-range)
WS_X_MIN, WS_X_MAX = -999.0, 999.0    # Min and max X (cm) - effectively no limit
WS_Y_MIN, WS_Y_MAX = -999.0, 999.0    # Min and max Y (cm) - effectively no limit

# ----------------------------------------------------------------------
# WASTE CLASS MAPPING (from YOLO class IDs)
# ----------------------------------------------------------------------
# The YOLO model was trained with these class IDs:
#   0: biodegradable, 1: cardboard, 2: glass, 3: metal, 4: paper, 5: plastic
#
# Recyclable classes  -> arm picks and drops into recycling bin (normal PICK flow)
# Non-recyclable      -> arm picks at object coords then drops to garbage bin
#                        (sent as PICK GARBAGE x y w h to Arduino)
WASTE_CLASSES = {
    1: {"name": "cardboard",     "recycle": True},
    4: {"name": "paper",         "recycle": True},
    5: {"name": "plastic",       "recycle": True},
    0: {"name": "biodegradable", "recycle": False},   # -> GARBAGE
    2: {"name": "glass",         "recycle": False},   # -> GARBAGE
    3: {"name": "metal",         "recycle": False},   # -> GARBAGE
}

# Colors for drawing bounding boxes (BGR format)
CLASS_COLORS = {
    "plastic":       (255, 80,  80),    # reddish
    "paper":         (200, 200, 255),   # light blue
    "cardboard":     (50,  165, 255),   # orange
    "biodegradable": (50,  200,  50),   # green
    "glass":         (200, 200,  50),   # yellow
    "metal":         (180, 180, 180),   # silver
    "default":       (128, 128, 128),   # grey fallback
}

# ----------------------------------------------------------------------
# CAMERA EXCLUSION ZONES
# ----------------------------------------------------------------------
EXCLUSION_ZONES = [
    (  0,   0, 120, 480),   # Left strip  - plastic + paper bins
    (520,   0, 640, 480),   # Right strip - cardboard bin side
    (  0,   0, 640, 100),   # Top strip   - robot arm base and body
]

# ----------------------------------------------------------------------
# HOMOGRAPHY FUNCTIONS
# ----------------------------------------------------------------------
def load_homography(path):
    """Load a homography matrix from a .npy file."""
    try:
        H = np.load(path)
        logging.info(f"Homography loaded from '{path}'")
        return H
    except Exception as e:
        logging.error(f"Could not load homography: {e}")
        return None

def pixel_to_cm(cx, cy, H):
    """Convert pixel coordinates (cx, cy) to real-world (x_cm, y_cm) via homography."""
    pt = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H)[0][0]
    return float(result[0]), float(result[1])

def in_workspace(x_cm, y_cm):
    """Workspace check disabled - Arduino handles out-of-range gracefully."""
    return True

# ----------------------------------------------------------------------
# EXCLUSION ZONE HELPERS
# ----------------------------------------------------------------------
def point_in_exclusion_zone(cx, cy):
    for (x1, y1, x2, y2) in EXCLUSION_ZONES:
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return True
    return False

def draw_exclusion_zones(frame):
    overlay = frame.copy()
    for (x1, y1, x2, y2) in EXCLUSION_ZONES:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 200), -1)
        cv2.rectangle(frame,   (x1, y1), (x2, y2), (0, 0, 255),  2)
        cv2.putText(frame, "EXCLUSION", (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

# ----------------------------------------------------------------------
# SERIAL COMMUNICATION CLASS
# ----------------------------------------------------------------------
class ArmComm:
    """Handles all serial communication with the Arduino (ItsyBitsy)."""

    def __init__(self, port, baudrate, timeout):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.conn = None
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

    def pick(self, class_name, x_cm, y_cm, w, h, gripper_rot):
        """Send PICK command including gripper rotation servo value."""
        return self.send(f"PICK {class_name.upper()} {x_cm:.2f} {y_cm:.2f} {w} {h} {gripper_rot}")

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
                if "DONE" in line:
                    return True
                if "ERROR" in line:
                    return False
            time.sleep(0.05)
        return False

    def close(self):
        if self.conn:
            self.conn.close()

# ----------------------------------------------------------------------
# DETECTION FUNCTION
# ----------------------------------------------------------------------
def compute_gripper_rotation(w, h):
    """
    Estimate gripper rotation servo value from bounding box aspect ratio.
    If w > h (landscape): rotate 90° to grab across short axis -> servo = 20
    If w <= h (portrait):  no rotation needed                   -> servo = 90
    """
    return 20 if w > h else 90


def detect_best(frame, model):
    """
    Run YOLO detection on a single frame.
    Returns the highest-confidence detection (recyclable OR non-recyclable),
    skipping any detections inside exclusion zones.
    Non-recyclable detections are returned with class_name='garbage' so the
    Arduino knows to pick then drop to the garbage bin.
    """
    results = model.predict(
        frame, conf=CONFIDENCE_THRESHOLD,
        iou=0.45, verbose=False, stream=True
    )
    r = next(results, None)

    if r is None or r.boxes is None:
        return None

    best = None
    best_conf = 0.0

    for xyxy, conf, c in zip(
        r.boxes.xyxy.cpu().numpy(),
        r.boxes.conf.cpu().numpy(),
        r.boxes.cls.cpu().numpy().astype(int)
    ):
        info = WASTE_CLASSES.get(int(c))
        x1, y1, x2, y2 = map(int, xyxy)
        conf_val = float(conf)

        cx_det = (x1 + x2) // 2
        cy_det = (y1 + y2) // 2

        # Skip detections inside exclusion zones
        if point_in_exclusion_zone(cx_det, cy_det):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), 1)
            cv2.putText(frame, "ZONE", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
            continue

        if not info:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "UNKNOWN", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            continue

        name = info["name"]
        recycle = info["recycle"]

        # Non-recyclable: label it GARBAGE for the Arduino, draw in grey
        arm_class = name if recycle else "garbage"
        color = CLASS_COLORS.get(name, CLASS_COLORS["default"])
        label = f"{name} {conf_val:.2f}" if recycle else f"GARBAGE ({name}) {conf_val:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Accept both recyclable and non-recyclable detections as pick candidates
        if conf_val > best_conf:
            best_conf = conf_val
            w_px, h_px = x2 - x1, y2 - y1
            best = {
                "class_id":    int(c),
                "class_name":  arm_class,
                "display_name": name,
                "recycle":     recycle,
                "center":      (cx_det, cy_det),
                "bbox":        (w_px, h_px),
                "gripper_rot": compute_gripper_rotation(w_px, h_px),
                "confidence":  conf_val,
            }

    return best

# ----------------------------------------------------------------------
# FEEDBACK LOOP (runs in separate thread)
# ----------------------------------------------------------------------
def feedback_loop(camera, model, arm, target_class, stop_event, done_event):
    """
    Monitors the camera after the arm has moved to the object.
    Signals done_event when the object disappears for DISAPPEAR_THRESHOLD frames.
    """
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
                logging.info("Object gone — pick complete")
                done_event.set()
                break

        time.sleep(0.1)

# ----------------------------------------------------------------------
# MAIN FUNCTION
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

    # Open camera
    cap = None
    for idx in [0, 1, 2]:
        c = cv2.VideoCapture(idx)
        c.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
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

    # Connect to Arduino
    arm = ArmComm(SERIAL_PORT, SERIAL_BAUDRATE, SERIAL_TIMEOUT)
    if not arm.connected:
        logger.warning("Arm not connected — commands will be simulated")

    # State variables
    busy           = False
    busy_start     = 0
    confirm        = 0
    last_detection = None
    last_detect_time = time.time()
    stop_fb = threading.Event()
    done_fb = threading.Event()
    fb_thread = None

    logger.info("System READY. Press Q to quit, H for home, G to grip, R to release.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        draw_exclusion_zones(frame)

        # --- Feedback loop finished: pick confirmed (non-bin classes only) ---
        if busy and done_fb.is_set():
            logger.info("Object picked — Arduino now handling drop/home internally")
            stop_fb.set()
            if fb_thread:
                fb_thread.join(timeout=2)

            # For GARBAGE: the Arduino PICK GARBAGE command already handles
            # moving to the bin, opening the gripper, and returning home.
            # For recyclables: wait for the Arduino to finish the pick move,
            # then send a HOME (or additional sorting logic goes here).
            arm.wait_done(60)   # Wait for Arduino to finish entire sequence

            busy = False
            confirm = 0
            last_detection = None
            stop_fb.clear()
            done_fb.clear()
            fb_thread = None

        # --- Timeout for stuck pick ---
        if busy and (time.time() - busy_start) > PICK_TIMEOUT:
            logger.error("Pick timeout — resetting arm")
            stop_fb.set()
            if fb_thread:
                fb_thread.join(timeout=1)
            arm.home()
            arm.wait_done(10)
            busy = False
            confirm = 0
            stop_fb.clear()
            done_fb.clear()
            fb_thread = None

        # --- Detection (only when idle) ---
        if not busy:
            det = detect_best(frame, model)
            if det:
                confirm += 1
                last_detection = det
                last_detect_time = time.time()

                cv2.putText(frame,
                            f"Confirming {det['display_name']} {confirm}/{DETECT_CONFIRM_FRAMES}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                if confirm >= DETECT_CONFIRM_FRAMES:
                    cx, cy = det["center"]
                    w, h   = det["bbox"]
                    x_cm, y_cm = pixel_to_cm(cx, cy, H)

                    logger.info(
                        f"Picking {det['display_name']} (->arm class: {det['class_name']}) "
                        f"at pixel=({cx},{cy}) -> ({x_cm:.2f}, {y_cm:.2f}) cm"
                    )

                    if not in_workspace(x_cm, y_cm):
                        logger.warning(f"({x_cm:.2f}, {y_cm:.2f}) outside workspace — skipping")
                        confirm = 0
                    else:
                        # For bin classes the Arduino handles the gripper
                        # internally — no need to open it from Python first.
                        # For non-bin classes open the gripper before moving.
                        bin_classes = {"plastic", "paper", "cardboard"}
                        if det["class_name"].lower() not in bin_classes:
                            arm.grip()
                            if not arm.wait_done(5):
                                logger.warning("Gripper open timed out")
                                confirm = 0
                                continue

                        # Send PICK with the arm class name:
                        #   recyclables  -> "PICK PLASTIC 3.5 2.0 100 80"  (normal pick)
                        #   non-recyclables -> "PICK GARBAGE 3.5 2.0 100 80"
                        #     Arduino will pick at those coords, close gripper,
                        #     then travel to the fixed bin and return HOME automatically.
                        if arm.pick(det["class_name"], x_cm, y_cm, w, h, det["gripper_rot"]):
                            # PLASTIC/PAPER/CARDBOARD: Arduino handles the full sequence
                            if det["class_name"].lower() in bin_classes:
                                busy = True
                                busy_start = time.time()
                                confirm = 0
                                logger.info(
                                    f"{det['class_name']} -> waiting for Arduino to complete "
                                    "full pick+bin+home sequence..."
                                )
                                # Wait for Arduino to finish the entire sequence
                                if not arm.wait_done(60):
                                    logger.warning("Bin sequence timed out")
                                    arm.home()
                                    arm.wait_done(10)
                                else:
                                    logger.info("Bin sequence complete")
                                busy = False

                            else:
                                # Non-bin classes: wait for arm to reach object,
                                # close gripper, then use feedback loop
                                if arm.wait_done(15):
                                    arm.release()
                                    if not arm.wait_done(2):
                                        logger.warning("Gripper close timed out")

                                    busy = True
                                    busy_start = time.time()
                                    confirm = 0
                                    stop_fb.clear()
                                    done_fb.clear()
                                    fb_thread = threading.Thread(
                                        target=feedback_loop,
                                        args=(cap, model, arm,
                                              det["class_id"], stop_fb, done_fb),
                                        daemon=True
                                    )
                                    fb_thread.start()
                                else:
                                    logger.error("Arm did not reach pick position")
                                    arm.home()
                                    arm.wait_done(10)
                        else:
                            logger.error("Failed to send PICK command")
            else:
                confirm = 0

        # --- Auto-home after idle ---
        if not busy and (time.time() - last_detect_time) > IDLE_HOME_TIMEOUT:
            logger.info("Idle timeout — sending HOME")
            arm.home()
            arm.wait_done(10)
            last_detect_time = time.time()

        # --- HUD ---
        status = "BUSY" if busy else "READY"
        color  = (0, 165, 255) if busy else (0, 255, 0)
        cv2.putText(frame, f"Status: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if last_detection and not busy:
            cv2.putText(frame,
                        f"Target: {last_detection['display_name']} "
                        f"({last_detection['class_name']}) "
                        f"{last_detection['confidence']:.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Waste Sorter", frame)

        # --- Keyboard controls ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if not busy:
            if key == ord('h'):
                arm.home()
                arm.wait_done()
            elif key == ord('g'):
                arm.grip()
            elif key == ord('r'):
                arm.release()

    cap.release()
    cv2.destroyAllWindows()
    arm.close()

if __name__ == "__main__":
    main()
