"""
robot_vision.py - Waste sorter with homography-based coordinate conversion

This script uses a YOLO model to detect recyclable waste, converts pixel coordinates
to real-world coordinates using a homography matrix, and commands a robot arm (via serial)
to pick and sort the items.
"""

# ----------------------------------------------------------------------
# IMPORT LIBRARIES
# ----------------------------------------------------------------------

# Enables forward references in type hints (not heavily used here)
from __future__ import annotations

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
CAMERA_INDEX = 0                       # Camera device index (0 = built‑in, 1 = USB)
FRAME_WIDTH, FRAME_HEIGHT = 640, 480   # Camera resolution (width, height)
SERIAL_PORT = "COM5"                   # Serial port where Arduino is connected
SERIAL_BAUDRATE = 9600                 # Baud rate (must match Arduino firmware)
SERIAL_TIMEOUT = 30                    # Default timeout in seconds for serial responses

DETECT_CONFIRM_FRAMES = 3              # Number of consecutive frames with the same detection to confirm
DISAPPEAR_THRESHOLD = 5                # How many frames the object must be missing before pick is considered complete
PICK_TIMEOUT = 20                      # Seconds after which a stuck pick operation is aborted
IDLE_HOME_TIMEOUT = 15                 # Seconds with no detection before sending HOME command

# Workspace check disabled – arm will attempt any position (Arduino handles out‑of‑range)
WS_X_MIN, WS_X_MAX = -999.0, 999.0     # Min and max X (cm) – effectively no limit
WS_Y_MIN, WS_Y_MAX = -999.0, 999.0     # Min and max Y (cm) – effectively no limit

# Drop‑off positions for each waste class (x, y) in real‑world cm (currently unused, kept for future)
DROP_POSITIONS = {
    "plastic":   (0.0, 4.0+1/8),
    "paper":     ( 0.0, 14+3/8),
    "cardboard": ( 16+2/8, 3.5),
}

# ----------------------------------------------------------------------
# WASTE CLASS MAPPING (from YOLO class IDs)
# ----------------------------------------------------------------------
# The YOLO model was trained with these class IDs:
#   0: biodegradable, 1: cardboard, 2: glass, 3: metal, 4: paper, 5: plastic
# Only cardboard (1), paper (4), and plastic (5) are considered recyclable.
WASTE_CLASSES = {
    1: {"name": "cardboard", "recycle": True},   # cardboard – recyclable
    4: {"name": "paper",     "recycle": True},   # paper – recyclable
    5: {"name": "plastic",   "recycle": True},   # plastic – recyclable
    0: {"name": "biodegradable", "recycle": False},  # not recyclable
    2: {"name": "glass",         "recycle": False},  # not recyclable
    3: {"name": "metal",         "recycle": False},  # not recyclable
}

# Colors for drawing bounding boxes (BGR format)
CLASS_COLORS = {
    "plastic":   (255, 80,  80),    # reddish
    "paper":     (200, 200, 255),   # light blue
    "cardboard": (50,  165, 255),   # orange
    "default":   (128, 128, 128),   # grey for rejected / unknown
}

# ----------------------------------------------------------------------
# HOMOGRAPHY FUNCTIONS
# ----------------------------------------------------------------------
def load_homography(path):
    """Load a homography matrix from a .npy file."""
    try:
        H = np.load(path)                         # Load the numpy array
        logging.info(f"Homography loaded from '{path}'")
        return H
    except Exception as e:
        logging.error(f"Could not load homography: {e}")
        return None

def pixel_to_cm(cx, cy, H):
    """
    Convert pixel coordinates (cx, cy) to real‑world (x_cm, y_cm) using the homography matrix H.
    The homography maps pixels to real‑world coordinates in centimeters.
    """
    # Create a 3D point (1x1x2) required by cv2.perspectiveTransform
    pt = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
    # Apply the homography transformation
    result = cv2.perspectiveTransform(pt, H)[0][0]
    x_cm = float(result[0])    # real‑world X coordinate (forward direction)
    y_cm = float(result[1])    # real‑world Y coordinate (lateral direction)
    return x_cm, y_cm

def in_workspace(x_cm, y_cm):
    """Check if the real‑world coordinates are within the robot's workspace (always returns True)."""
    return True   # Workspace check disabled – Arduino will handle out‑of‑range gracefully

# ----------------------------------------------------------------------
# SERIAL COMMUNICATION CLASS
# ----------------------------------------------------------------------
class ArmComm:
    """Handles all serial communication with the Arduino (ItsyBitsy)."""

    def __init__(self, port, baudrate, timeout):
        """Store connection parameters and attempt to open the serial port."""
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.conn = None          # Serial connection object
        self.connected = False    # Connection status flag
        self._connect()           # Try to connect immediately

    def _connect(self):
        """Open the serial port and wait for Arduino to be ready."""
        try:
            self.conn = serial.Serial(
                self.port, self.baudrate,
                timeout=1,          # Read timeout (seconds)
                write_timeout=5     # Write timeout (seconds)
            )
            time.sleep(2)           # Allow Arduino to reset after opening the port
            self.connected = True
            logging.info(f"Connected to {self.port}")
        except Exception as e:
            logging.error(f"Serial error: {e}")

    def send(self, cmd):
        """Send a raw command string over serial (automatically appends a newline)."""
        if not self.connected:
            return False
        try:
            self.conn.write((cmd + "\n").encode())   # Convert to bytes and send
            return True
        except Exception as e:
            logging.error(f"Send failed: {e}")
            return False

    def pick(self, class_name, x_cm, y_cm, w, h):
        """Send a PICK command with class name, position (cm), and bounding box size."""
        return self.send(f"PICK {class_name.upper()} {x_cm:.2f} {y_cm:.2f} {w} {h}")

    def move(self, x_cm, y_cm):
        """Send a MOVE command to go to absolute coordinates (cm)."""
        return self.send(f"MOVE {x_cm:.2f} {y_cm:.2f}")

    def grip(self):
        """Send GRIP command – in the firmware this normally closes the gripper."""
        return self.send("GRIP")

    def release(self):
        """Send RELEASE command – in the firmware this normally opens the gripper."""
        return self.send("RELEASE")

    def home(self):
        """Send HOME command – returns the arm to its home position."""
        return self.send("HOME")

    def wait_done(self, timeout=None):
        """
        Wait for the Arduino to reply with 'DONE' or 'ERROR'.
        Returns True if 'DONE' is received, False on timeout or 'ERROR'.
        """
        deadline = time.time() + (timeout or self.timeout)
        while time.time() < deadline:
            if self.conn and self.conn.in_waiting:
                line = self.conn.readline().decode(errors='replace').strip().upper()
                if "DONE" in line:
                    return True
                if "ERROR" in line:
                    return False
            time.sleep(0.05)      # Small delay to avoid busy‑waiting
        return False

    def close(self):
        """Close the serial connection."""
        if self.conn:
            self.conn.close()

# ----------------------------------------------------------------------
# DETECTION FUNCTION
# ----------------------------------------------------------------------
def detect_best(frame, model):
    """
    Run YOLO detection on a single frame.
    Draw bounding boxes on the frame.
    Return the best (highest confidence) recyclable detection, or None if none.
    """
    # Run inference with stream=True for lower memory usage
    results = model.predict(
        frame, conf=CONFIDENCE_THRESHOLD,
        iou=0.45, verbose=False, stream=True
    )
    r = next(results, None)          # Get the first (and only) result from the generator

    if r is None or r.boxes is None: # No detections at all
        return None

    best = None
    best_conf = 0.0

    # Iterate through all detected bounding boxes
    for xyxy, conf, c in zip(
        r.boxes.xyxy.cpu().numpy(),      # bounding box coordinates (x1,y1,x2,y2)
        r.boxes.conf.cpu().numpy(),      # confidence scores
        r.boxes.cls.cpu().numpy().astype(int)  # class IDs
    ):
        info = WASTE_CLASSES.get(int(c))  # Get class info from our mapping
        x1, y1, x2, y2 = map(int, xyxy)  # Convert to integers for drawing
        conf_val = float(conf)

        if not info:   # Unknown class – draw red box and skip
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "UNKNOWN", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            continue

        name = info["name"]
        recycle = info["recycle"]
        # Choose color: recyclable uses class color, non‑recyclable uses grey
        color = CLASS_COLORS.get(name, CLASS_COLORS["default"]) if recycle else (128, 128, 128)
        label = f"{name} {conf_val:.2f}" if recycle else f"REJECT: {name}"

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # If this is a recyclable item and has higher confidence than previous best, keep it
        if recycle and conf_val > best_conf:
            best_conf = conf_val
            best = {
                "class_id":   int(c),
                "class_name": name,
                "center":     ((x1 + x2) // 2, (y1 + y2) // 2),  # pixel center
                "bbox":       (x2 - x1, y2 - y1),                # width, height
                "confidence": conf_val,
            }
    return best

# ----------------------------------------------------------------------
# FEEDBACK LOOP (runs in separate thread after the arm reaches the object)
# ----------------------------------------------------------------------
def feedback_loop(camera, model, arm, target_class, stop_event, done_event):
    """
    This function runs in a separate thread while the arm is holding the object.
    It monitors the camera: if the object disappears for DISAPPEAR_THRESHOLD frames,
    it signals that the pick is complete (does NOT send GRIP – the gripper was already closed).
    """
    disappear = 0
    while not stop_event.is_set():
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.1)
            continue

        # Run a quick detection on the frame to see if the target object is still visible
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
            disappear = 0                     # Object still there, reset counter
        else:
            disappear += 1
            if disappear >= DISAPPEAR_THRESHOLD:
                logging.info("Object gone — pick complete")
                # Do NOT send GRIP again (gripper already closed after reaching)
                done_event.set()              # Signal the main thread that pick is done
                break
        time.sleep(0.1)                       # ~10 Hz monitoring rate

# ----------------------------------------------------------------------
# MAIN FUNCTION
# ----------------------------------------------------------------------
def main():
    # Set up logging: show timestamp, level, and message
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger("WasteSorter")

    # ------------------- Load YOLO model -------------------
    try:
        model = YOLO(MODEL_PATH)
        logger.info(f"Model loaded. Classes: {model.names}")
    except Exception as e:
        logger.error(f"Model load error: {e}")
        return

    # ------------------- Load homography matrix -------------------
    H = load_homography(HOMOGRAPHY_PATH)
    if H is None:
        logger.error("Cannot run without homography. Run calibrate_homography.py first.")
        return

    # ------------------- Open camera -------------------
    cap = None
    for idx in [0, 1, 2]:        # Try indices 0, 1, 2 until one works
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

    # ------------------- Connect to Arduino -------------------
    arm = ArmComm(SERIAL_PORT, SERIAL_BAUDRATE, SERIAL_TIMEOUT)
    if not arm.connected:
        logger.warning("Arm not connected — commands will be simulated")

    # ------------------- State variables -------------------
    busy = False                # True when arm is moving or picking
    busy_start = 0              # Timestamp when busy state began (for timeout)
    confirm = 0                 # Number of consecutive frames with same detection
    last_detection = None       # Store the last valid detection
    last_detect_time = time.time()   # Last time an object was detected (for idle timeout)
    stop_fb = threading.Event()      # Signal to stop the feedback loop thread
    done_fb = threading.Event()      # Signal that feedback loop has finished
    fb_thread = None                 # Thread object for feedback loop

    logger.info("System READY. Press Q to quit, H for home, G to grip, R to release.")

    # ------------------- Main loop -------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Timeout for stuck pick operation ---
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

        # --- Feedback loop finished: pick is complete, return home ---
        if busy and done_fb.is_set():
            logger.info("Object picked — moving to drop position")
            stop_fb.set()
            if fb_thread:
                fb_thread.join(timeout=2)
            arm.wait_done(10)          # Wait for the final GRIP? (none, because we already closed)
            # Optional drop‑off code (commented out)
            arm.home()
            arm.wait_done(10)
            busy = False
            confirm = 0
            stop_fb.clear()
            done_fb.clear()
            fb_thread = None

        # --- Detection (only when arm is idle) ---
        if not busy:
            det = detect_best(frame, model)
            if det:
                confirm += 1
                last_detection = det
                last_detect_time = time.time()
                # Draw confirmation progress on the frame
                cv2.putText(frame,
                            f"Confirming {confirm}/{DETECT_CONFIRM_FRAMES}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                if confirm >= DETECT_CONFIRM_FRAMES:
                    cx, cy = det["center"]   # Pixel center of the object
                    w, h = det["bbox"]       # Bounding box width and height
                    # Convert pixel to real‑world coordinates (note swapped order: y_cm, x_cm)
                    x_cm, y_cm = pixel_to_cm(cx, cy, H)

                    logger.info(
                        f"Picking {det['class_name']} at "
                        f"pixel=({cx},{cy}) → ({x_cm:.2f}, {y_cm:.2f}) cm"
                    )

                    if not in_workspace(x_cm, y_cm):
                        logger.warning(f"({x_cm:.2f}, {y_cm:.2f}) outside workspace — skipping")
                        confirm = 0
                    else:
                        # --- Open gripper BEFORE moving to the object ---
                        # Note: GRIP is normally "close", but we are using it to "open" due to swapped behavior
                        arm.grip()                     # Send GRIP command
                        if not arm.wait_done(5):
                            logger.warning("Gripper open command timed out")
                            confirm = 0
                            continue

                        # Send PICK command and wait for arm to reach the position
                        if arm.pick(det["class_name"], x_cm, y_cm, w, h):
                            if arm.wait_done(15):      # Wait for Arduino to finish moving
                                # --- Close gripper NOW that the arm is over the object ---
                                # Note: RELEASE is normally "open", but we use it to "close" (swapped)
                                arm.release()
                                if not arm.wait_done(2):
                                    logger.warning("Gripper close command timed out")

                                # Start the feedback loop thread to monitor object disappearance
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

        # --- Auto‑home after prolonged idle (no detections) ---
        if not busy and (time.time() - last_detect_time) > IDLE_HOME_TIMEOUT:
            logger.info("Idle timeout — sending HOME")
            arm.home()
            arm.wait_done(10)
            last_detect_time = time.time()

        # --- Draw HUD (heads‑up display) on the frame ---
        status = "BUSY" if busy else "READY"
        color = (0, 165, 255) if busy else (0, 255, 0)
        cv2.putText(frame, f"Status: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if last_detection and not busy:
            cv2.putText(frame,
                        f"Target: {last_detection['class_name']} {last_detection['confidence']:.2f}",
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

    # ------------------- Cleanup -------------------
    cap.release()
    cv2.destroyAllWindows()
    arm.close()

if __name__ == "__main__":
    main()