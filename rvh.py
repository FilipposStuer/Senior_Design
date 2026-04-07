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
# Only cardboard (1), paper (4), and plastic (5) are considered recyclable.
WASTE_CLASSES = {
    1: {"name": "cardboard", "recycle": True},        # cardboard - recyclable
    4: {"name": "paper",     "recycle": True},        # paper - recyclable
    5: {"name": "plastic",   "recycle": True},        # plastic - recyclable
    0: {"name": "biodegradable", "recycle": False},   # not recyclable
    2: {"name": "glass",         "recycle": False},   # not recyclable
    3: {"name": "metal",         "recycle": False},   # not recyclable
}

# Colors for drawing bounding boxes (BGR format)
CLASS_COLORS = {
    "plastic":   (255, 80,  80),    # reddish
    "paper":     (200, 200, 255),   # light blue
    "cardboard": (50,  165, 255),   # orange
    "default":   (128, 128, 128),   # grey for rejected / unknown
}

# ----------------------------------------------------------------------
# CAMERA EXCLUSION ZONES
# ----------------------------------------------------------------------
# Rectangular regions (in pixels) where the camera will NOT detect objects.
# Any detection whose center falls inside one of these rectangles is ignored.
# This prevents the arm from trying to pick objects out of the bins or
# misidentifying parts of the robot arm as objects.
# Format: (x1, y1, x2, y2) - top-left and bottom-right corners in pixels.
# Zones are drawn on screen in red so the operator can verify coverage.
EXCLUSION_ZONES = [
    (  0,   0, 120, 480),   # Left strip (3/4 of original) - plastic + paper bins / Punto 1 & 3 side
    (520,   0, 640, 480),   # Right strip (3/4 of original) - cardboard bin side
    (  0,   0, 640, 100),   # Top strip - robot arm base and body
]

# ----------------------------------------------------------------------
# HOMOGRAPHY FUNCTIONS
# ----------------------------------------------------------------------
def load_homography(path):
    """Load a homography matrix from a .npy file."""
    try:
        H = np.load(path)                          # Load the numpy array from disk
        logging.info(f"Homography loaded from '{path}'")
        return H
    except Exception as e:
        logging.error(f"Could not load homography: {e}")
        return None

def pixel_to_cm(cx, cy, H):
    """
    Convert pixel coordinates (cx, cy) to real-world (x_cm, y_cm)
    using the homography matrix H.
    """
    pt = np.array([[[float(cx), float(cy)]]], dtype=np.float32)   # Shape required by OpenCV
    result = cv2.perspectiveTransform(pt, H)[0][0]                 # Apply homography
    x_cm = float(result[0])    # Real-world X coordinate (forward direction)
    y_cm = float(result[1])    # Real-world Y coordinate (lateral direction)
    return x_cm, y_cm

def in_workspace(x_cm, y_cm):
    """Check if the real-world coordinates are within the robot workspace (always True)."""
    return True   # Workspace check disabled - Arduino handles out-of-range gracefully

# ----------------------------------------------------------------------
# EXCLUSION ZONE HELPERS
# ----------------------------------------------------------------------
def point_in_exclusion_zone(cx, cy):
    """
    Return True if pixel point (cx, cy) falls inside any exclusion zone.
    Used by detect_best() to skip detections over bin or arm areas.
    """
    for (x1, y1, x2, y2) in EXCLUSION_ZONES:   # Check every defined zone
        if x1 <= cx <= x2 and y1 <= cy <= y2:   # Point is inside this rectangle
            return True
    return False   # Point is outside all zones - detection is valid

def draw_exclusion_zones(frame):
    """
    Draw all exclusion zones on the frame as semi-transparent red rectangles.
    Helps the operator verify which areas of the camera view are masked.
    """
    overlay = frame.copy()                            # Copy frame for blending
    for (x1, y1, x2, y2) in EXCLUSION_ZONES:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 200), -1)    # Filled red on overlay
        cv2.rectangle(frame,   (x1, y1), (x2, y2), (0, 0, 255),  2)    # Red border on frame
        cv2.putText(frame, "EXCLUSION", (x1 + 4, y1 + 18),              # Label on frame
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)   # Blend at 25% opacity

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
        if not self.connected:    # Do nothing if not connected
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
        """Send GRIP command - physically opens the gripper (inverted wiring)."""
        return self.send("GRIP")

    def release(self):
        """Send RELEASE command - physically closes the gripper (inverted wiring)."""
        return self.send("RELEASE")

    def home(self):
        """Send HOME command - returns the arm to its home position."""
        return self.send("HOME")

    def drop(self, class_name):
        """
        Send a DROP command to Arduino with the detected waste class name.
        Arduino is responsible for all movement to the correct bin, releasing
        the object, and returning home. Python only tells it what was picked.
        Example serial output: DROP PLASTIC
        """
        return self.send(f"DROP {class_name.upper()}")   # Send class name in uppercase

    def wait_done(self, timeout=None):
        """
        Wait for the Arduino to reply with 'DONE' or 'ERROR'.
        Returns True if 'DONE' is received, False on timeout or 'ERROR'.
        """
        deadline = time.time() + (timeout or self.timeout)   # Calculate deadline timestamp
        while time.time() < deadline:                         # Keep checking until deadline
            if self.conn and self.conn.in_waiting:            # Data available in serial buffer
                line = self.conn.readline().decode(errors='replace').strip().upper()
                if "DONE" in line:     # Arduino finished successfully
                    return True
                if "ERROR" in line:    # Arduino reported an error
                    return False
            time.sleep(0.05)           # Small delay to avoid busy-waiting (~20 Hz)
        return False                   # Timeout reached without response

    def close(self):
        """Close the serial connection."""
        if self.conn:
            self.conn.close()   # Release the serial port

# ----------------------------------------------------------------------
# DROP-OFF FUNCTION
# ----------------------------------------------------------------------
def drop_object(arm, class_name, logger):
    """
    Tell Arduino to carry the held object to the correct bin and release it.

    Python sends a single DROP command with the waste class name.
    Arduino handles everything: lifting the arm, moving to the bin,
    opening the gripper, and returning home. Python just waits for DONE.

    Parameters
    ----------
    arm        : ArmComm - active serial connection to the Arduino
    class_name : str     - detected waste class ("plastic", "paper", "cardboard")
    logger     : Logger  - logger instance for status messages
    """
    logger.info(f"Sending DROP command for class: '{class_name}'")   # Log what we are dropping

    # Send DROP command to Arduino - it knows where each bin is located
    # and will handle the full movement sequence independently
    if not arm.drop(class_name):                          # Send "DROP PLASTIC" (or paper/cardboard)
        logger.error("Failed to send DROP command")       # Serial write failed
        arm.home()                                        # Fail safe - go home
        arm.wait_done(10)                                 # Wait for home to complete
        return

    # Wait for Arduino to complete the full drop sequence
    # This includes: lift -> move to bin -> open gripper -> return home
    # Timeout is generous because the full sequence takes several seconds
    if not arm.wait_done(60):                                          # Wait up to 60 seconds
        logger.warning("DROP command timed out - Arduino did not respond with DONE")
    else:
        logger.info(f"DROP complete for '{class_name}' - arm returned home")   # Success

# ----------------------------------------------------------------------
# DETECTION FUNCTION
# ----------------------------------------------------------------------
def detect_best(frame, model):
    """
    Run YOLO detection on a single frame.
    Draw bounding boxes on the frame.
    Skip any detection whose center falls inside an exclusion zone.
    Return the best (highest confidence) recyclable detection, or None if none.
    """
    results = model.predict(              # Run YOLO inference on the frame
        frame, conf=CONFIDENCE_THRESHOLD,
        iou=0.45, verbose=False, stream=True
    )
    r = next(results, None)               # Get the first result from the generator

    if r is None or r.boxes is None:      # No detections at all in this frame
        return None

    best = None       # Will hold the best detection found so far
    best_conf = 0.0   # Confidence score of the best detection so far

    for xyxy, conf, c in zip(
        r.boxes.xyxy.cpu().numpy(),             # Bounding box corners (x1,y1,x2,y2)
        r.boxes.conf.cpu().numpy(),             # Confidence scores
        r.boxes.cls.cpu().numpy().astype(int)   # Class IDs as integers
    ):
        info = WASTE_CLASSES.get(int(c))    # Look up class info from our mapping
        x1, y1, x2, y2 = map(int, xyxy)    # Convert coordinates to integers for drawing
        conf_val = float(conf)              # Confidence as a float

        cx_det = (x1 + x2) // 2   # Horizontal center of this bounding box
        cy_det = (y1 + y2) // 2   # Vertical center of this bounding box

        # Skip detection if its center is inside an exclusion zone (bin or arm area)
        if point_in_exclusion_zone(cx_det, cy_det):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), 1)     # Draw dimmed box
            cv2.putText(frame, "ZONE", (x1, y1 - 8),                       # Label it ZONE
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
            continue   # Ignore this detection entirely

        if not info:   # Unknown class ID - draw red box and skip
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "UNKNOWN", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            continue

        name = info["name"]       # Human-readable class name
        recycle = info["recycle"] # Whether this class is recyclable

        # Choose color: recyclable uses class color, non-recyclable uses grey
        color = CLASS_COLORS.get(name, CLASS_COLORS["default"]) if recycle else (128, 128, 128)
        label = f"{name} {conf_val:.2f}" if recycle else f"REJECT: {name}"

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Keep this detection if it is recyclable and has higher confidence than current best
        if recycle and conf_val > best_conf:
            best_conf = conf_val
            best = {
                "class_id":   int(c),
                "class_name": name,
                "center":     (cx_det, cy_det),      # Pixel center of the bounding box
                "bbox":       (x2 - x1, y2 - y1),   # Width and height of the bounding box
                "confidence": conf_val,
            }
    return best   # Return best recyclable detection, or None if nothing found

# ----------------------------------------------------------------------
# FEEDBACK LOOP (runs in separate thread after the arm reaches the object)
# ----------------------------------------------------------------------
def feedback_loop(camera, model, arm, target_class, stop_event, done_event):
    """
    Runs in a separate thread while the arm is holding the object.
    Monitors the camera: if the object disappears for DISAPPEAR_THRESHOLD frames,
    it signals that the pick is complete.
    """
    disappear = 0   # Counter for consecutive frames where object is absent

    while not stop_event.is_set():   # Keep running until main thread signals stop
        ret, frame = camera.read()   # Grab a frame from the camera
        if not ret:                  # Camera read failed - try again
            time.sleep(0.1)
            continue

        # Run detection to check if the target object is still visible
        results = model.predict(
            frame, conf=CONFIDENCE_THRESHOLD,
            verbose=False, stream=True
        )
        r = next(results, None)
        visible = False              # Assume object is not visible until proven otherwise

        if r and r.boxes:
            for c in r.boxes.cls.cpu().numpy().astype(int):   # Check all detected classes
                if int(c) == target_class:                     # Found the target class
                    visible = True
                    break

        if visible:
            disappear = 0                      # Object still visible - reset counter
        else:
            disappear += 1                     # Object not visible - increment counter
            if disappear >= DISAPPEAR_THRESHOLD:
                logging.info("Object gone — pick complete")
                done_event.set()               # Signal main thread that pick is confirmed
                break

        time.sleep(0.1)   # Run at ~10 Hz to avoid flooding the CPU

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
        model = YOLO(MODEL_PATH)                               # Load the trained model
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
    busy = False                          # True when arm is moving or picking
    busy_start = 0                        # Timestamp when busy state began (for timeout)
    confirm = 0                           # Number of consecutive frames with same detection
    last_detection = None                 # Store the last valid detection
    last_detect_time = time.time()        # Last time an object was detected (for idle timeout)
    stop_fb = threading.Event()           # Signal to stop the feedback loop thread
    done_fb = threading.Event()           # Signal that feedback loop has finished
    fb_thread = None                      # Thread object for feedback loop

    logger.info("System READY. Press Q to quit, H for home, G to grip, R to release.")

    # ------------------- Main loop -------------------
    while True:
        ret, frame = cap.read()   # Read one frame from the camera
        if not ret:
            break

        draw_exclusion_zones(frame)   # Draw exclusion zones on every frame

        # --- Feedback loop finished: pick confirmed -> send DROP to Arduino ---
        # This is checked FIRST so it always takes priority over the timeout below
        if busy and done_fb.is_set():
            logger.info("Object picked — sending DROP command to Arduino")
            stop_fb.set()                      # Stop the feedback loop thread
            if fb_thread:
                fb_thread.join(timeout=2)      # Wait for thread to finish cleanly
            arm.wait_done(10)                  # Drain any remaining serial replies

            # Send DROP command - Arduino handles all movement to bin and back home
            if last_detection:
                drop_object(arm, last_detection["class_name"], logger)
            else:
                logger.warning("No detection stored - going home as fallback")
                arm.home()                     # Fail safe if detection was lost
                arm.wait_done(10)

            # Reset all state variables back to idle
            busy = False
            confirm = 0
            last_detection = None              # Clear last detection after drop
            stop_fb.clear()
            done_fb.clear()
            fb_thread = None

        # --- Timeout for stuck pick operation ---
        # Checked AFTER done_fb so a successful pick always goes to drop first
        if busy and (time.time() - busy_start) > PICK_TIMEOUT:
            logger.error("Pick timeout — resetting arm")
            stop_fb.set()                      # Stop the feedback loop thread
            if fb_thread:
                fb_thread.join(timeout=1)
            arm.home()                         # Return arm to safe position
            arm.wait_done(10)
            busy = False
            confirm = 0
            stop_fb.clear()
            done_fb.clear()
            fb_thread = None

        # --- Detection (only when arm is idle) ---
        if not busy:
            det = detect_best(frame, model)    # Run YOLO on current frame
            if det:
                confirm += 1                   # Increment confirmation counter
                last_detection = det           # Store latest valid detection
                last_detect_time = time.time() # Update last detection timestamp

                # Draw confirmation progress on the frame
                cv2.putText(frame,
                            f"Confirming {confirm}/{DETECT_CONFIRM_FRAMES}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                if confirm >= DETECT_CONFIRM_FRAMES:
                    cx, cy = det["center"]     # Pixel center of the detected object
                    w, h = det["bbox"]         # Bounding box width and height in pixels
                    x_cm, y_cm = pixel_to_cm(cx, cy, H)   # Convert to real-world cm

                    logger.info(
                        f"Picking {det['class_name']} at "
                        f"pixel=({cx},{cy}) -> ({x_cm:.2f}, {y_cm:.2f}) cm"
                    )

                    if not in_workspace(x_cm, y_cm):
                        logger.warning(f"({x_cm:.2f}, {y_cm:.2f}) outside workspace — skipping")
                        confirm = 0
                    else:
                        # Open gripper BEFORE moving to the object
                        # Note: GRIP physically opens the gripper (inverted wiring)
                        arm.grip()
                        if not arm.wait_done(5):
                            logger.warning("Gripper open command timed out")
                            confirm = 0
                            continue

                        # Send PICK command - arm moves to object coordinates
                        if arm.pick(det["class_name"], x_cm, y_cm, w, h):
                            if arm.wait_done(15):      # Wait for arm to reach position
                                # Close gripper now that arm is over the object
                                # Note: RELEASE physically closes the gripper (inverted wiring)
                                arm.release()
                                if not arm.wait_done(2):
                                    logger.warning("Gripper close command timed out")

                                # Start feedback loop thread to monitor object disappearance
                                busy = True
                                busy_start = time.time()   # Record when busy state started
                                confirm = 0
                                stop_fb.clear()
                                done_fb.clear()
                                fb_thread = threading.Thread(
                                    target=feedback_loop,
                                    args=(cap, model, arm,
                                          det["class_id"], stop_fb, done_fb),
                                    daemon=True
                                )
                                fb_thread.start()          # Start monitoring thread
                            else:
                                logger.error("Arm did not reach pick position")
                                arm.home()
                                arm.wait_done(10)
                        else:
                            logger.error("Failed to send PICK command")
            else:
                confirm = 0   # No detection this frame - reset confirmation counter

        # --- Auto-home after prolonged idle (no detections) ---
        if not busy and (time.time() - last_detect_time) > IDLE_HOME_TIMEOUT:
            logger.info("Idle timeout — sending HOME")
            arm.home()                         # Return arm to resting position
            arm.wait_done(10)
            last_detect_time = time.time()     # Reset timer to avoid spamming HOME

        # --- Draw HUD (heads-up display) on the frame ---
        status = "BUSY" if busy else "READY"
        color = (0, 165, 255) if busy else (0, 255, 0)   # Orange when busy, green when idle
        cv2.putText(frame, f"Status: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if last_detection and not busy:
            cv2.putText(frame,
                        f"Target: {last_detection['class_name']} {last_detection['confidence']:.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Waste Sorter", frame)

        # --- Keyboard controls ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):    # Q - quit the program
            break
        if not busy:
            if key == ord('h'):        # H - manual home
                arm.home()
                arm.wait_done()
            elif key == ord('g'):      # G - manual grip (physically opens gripper)
                arm.grip()
            elif key == ord('r'):      # R - manual release (physically closes gripper)
                arm.release()

    # ------------------- Cleanup -------------------
    cap.release()              # Release camera resource
    cv2.destroyAllWindows()    # Close all OpenCV windows
    arm.close()                # Close serial connection

if __name__ == "__main__":
    main()