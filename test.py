"""
================================================================================
  robot_vision.py — Waste Sorter with Homography-Based Coordinate Conversion
================================================================================

  This script uses a YOLO model to detect recyclable waste on a work surface,
  converts the detected pixel coordinates to real-world coordinates using a
  homography matrix, and commands a robotic arm (via serial) to pick up and
  sort the items into designated bins.

  SYSTEM OVERVIEW:
      Camera (USB)  -->  YOLO Detection  -->  Homography Transform  -->  Serial
      (640x480 px)      (best.pt model)      (pixel -> cm)              (Arduino)

  COORDINATE PIPELINE:
      pixel (cx, cy)
        --> homography --> real-world position in inches (table frame)
        --> the Arduino receives these values and handles IK internally

  ARDUINO FIRMWARE (v2.3):
      For bin classes (PLASTIC, METAL, CARDBOARD), the Arduino handles the
      entire pick-and-place sequence internally:
        1. Rotates gripper to match object orientation
        2. Opens gripper
        3. Moves arm to object position (IK lookup)
        4. Closes gripper to grab object
        5. Returns to HOME position (lifting the object)
        6. Moves to the designated bin (hardcoded servo angles per class)
        7. Opens gripper to release object
        8. Jiggles gripper to ensure object falls
        9. Returns to HOME
       10. Sends "DONE" over serial

      Python only needs to send PICK and wait for DONE.
      Non-bin classes (biodegradable, glass, paper) are detected and displayed
      on screen but NOT sent to the Arduino — they are ignored.

  DEPENDENCIES:
      pip install ultralytics opencv-python pyserial numpy
================================================================================
"""

# ----------------------------------------------------------------------
# IMPORT LIBRARIES
# ----------------------------------------------------------------------

from __future__ import annotations    # Enables forward references in type hints

import cv2                     # OpenCV: camera capture, image processing, drawing
import serial                  # PySerial: serial communication with Arduino
import time                    # Time: delays, timeouts, timestamps
import logging                 # Logging: structured info/warning/error messages
import threading               # Threading: feedback loop runs in background thread
import numpy as np             # NumPy: numerical operations, homography math
from ultralytics import YOLO   # Ultralytics: YOLO object detection model


# ==============================================================================
# SECTION 1: SYSTEM CONFIGURATION
# ==============================================================================
# Core paths and hardware settings. Adjust SERIAL_PORT to match your OS:
#   Windows: "COM5" (check Device Manager)
#   Linux:   "/dev/ttyUSB0" or "/dev/ttyACM0"
#   macOS:   "/dev/tty.usbmodem*"

MODEL_PATH = "best.pt"                 # Path to the trained YOLO model weights
HOMOGRAPHY_PATH = "homography.npy"     # Path to the calibrated homography matrix
CONFIDENCE_THRESHOLD = 0.30            # Minimum detection confidence (0.0 to 1.0)
CAMERA_INDEX = 0                       # Camera device index (0 = default, 1 = USB)
FRAME_WIDTH, FRAME_HEIGHT = 640, 480   # Camera capture resolution in pixels
SERIAL_PORT = "COM5"                   # Serial port for Arduino communication
SERIAL_BAUDRATE = 9600                 # Baud rate (must match Arduino firmware)
SERIAL_TIMEOUT = 30                    # Default timeout in seconds for serial responses


# ==============================================================================
# SECTION 2: TIMING AND DETECTION PARAMETERS
# ==============================================================================
# These parameters control the detection-to-pick pipeline timing.
#
# DETECT_CONFIRM_FRAMES: How many consecutive frames must show the same object
#     before the system commits to picking it. Prevents false positives from
#     single-frame glitches. Higher = more reliable, but slower response.
#
# DISAPPEAR_THRESHOLD: After the arm moves to the object, how many consecutive
#     frames the object must be missing before the pick is considered successful.
#     Used by the feedback loop thread (currently unused since non-bin classes
#     are ignored, but kept for future use).
#
# PICK_TIMEOUT: Maximum seconds to wait for a pick-and-place cycle to complete.
#     If exceeded, the arm is forcefully reset to HOME to prevent getting stuck.
#
# IDLE_HOME_TIMEOUT: If no object is detected for this many seconds, the arm
#     is sent to HOME position to avoid holding an extended pose.

DETECT_CONFIRM_FRAMES = 3
DISAPPEAR_THRESHOLD = 5
PICK_TIMEOUT = 60
IDLE_HOME_TIMEOUT = 15

# Workspace boundary check is disabled — the Arduino firmware handles
# out-of-range coordinates gracefully by clamping to the nearest IK entry.
WS_X_MIN, WS_X_MAX = -999.0, 999.0
WS_Y_MIN, WS_Y_MAX = -999.0, 999.0


# ==============================================================================
# SECTION 3: WASTE CLASS DEFINITIONS
# ==============================================================================
# Maps YOLO model class IDs to human-readable names and sorting behavior.
#
# The YOLO model was trained with these class IDs:
#   0: biodegradable, 1: cardboard, 2: glass, 3: metal, 4: paper, 5: plastic
#
# "recycle": True  -> The arm will pick this object and carry it to the
#                     class-specific bin. The Arduino handles the full sequence.
#
# "recycle": False -> The object is detected and shown on screen with a
#                     "GARBAGE" label, but the arm does NOT pick it up.
#                     No command is sent to the Arduino.
#
# To change which classes are picked:
#   1. Set "recycle": True for classes you want the arm to pick.
#   2. Make sure the Arduino firmware has a corresponding bin for that class
#      (see BIN_PLASTIC, BIN_METAL, BIN_CARDBOARD arrays in the .ino file).

WASTE_CLASSES = {
    1: {"name": "cardboard",     "recycle": True},
    4: {"name": "paper",         "recycle": False},
    5: {"name": "plastic",       "recycle": True},
    0: {"name": "biodegradable", "recycle": False},
    2: {"name": "glass",         "recycle": False},
    3: {"name": "metal",         "recycle": True},
}

# Bounding box colors for the camera overlay (BGR format for OpenCV)
CLASS_COLORS = {
    "plastic":       (255, 80,  80),    # Blue-red
    "paper":         (200, 200, 255),   # Light pink
    "cardboard":     (50,  165, 255),   # Orange
    "biodegradable": (50,  200,  50),   # Green
    "glass":         (200, 200,  50),   # Yellow-cyan
    "metal":         (180, 180, 180),   # Silver
    "default":       (128, 128, 128),   # Grey fallback
}


# ==============================================================================
# SECTION 4: CAMERA EXCLUSION ZONES
# ==============================================================================
# Rectangular regions of the camera image where detections are IGNORED.
# This prevents the system from trying to pick up the bins themselves,
# the robot arm body, or the camera tripod.
#
# Format: (x1, y1, x2, y2) in pixel coordinates
#   (x1, y1) = top-left corner of the exclusion rectangle
#   (x2, y2) = bottom-right corner of the exclusion rectangle
#
# The camera image is 640x480 pixels:
#   x ranges from 0 (left) to 640 (right)
#   y ranges from 0 (top) to 480 (bottom)
#
# To adjust these zones:
#   1. Run the script and look at the red "EXCLUSION" overlays on the video.
#   2. If a bin or the arm body is outside the red zone, expand the zone.
#   3. If the zone covers part of the work area, shrink it.

EXCLUSION_ZONES = [
    (  0,   0, 120, 480),   # Left strip  — plastic + paper bins area
    (520,   0, 640, 480),   # Right strip — cardboard bin side
    (  0,   0, 640,  40),   # Top strip   — robot arm base and body
    (  0, 400, 640, 480),   # Bottom strip — camera base and tripod
]


# ==============================================================================
# SECTION 5: HOMOGRAPHY COORDINATE CONVERSIONclea
# ==============================================================================
# The homography matrix transforms 2D pixel coordinates from the camera image
# into real-world coordinates (in inches) on the table surface.
#
# This was calibrated using calibrate_homography.py by clicking on 4 known
# points in the camera view and providing their measured positions in inches.
#
# The homography axes align with the arm:
#   Homography X axis = forward from the arm (away from base)
#   Homography Y axis = lateral (right = positive)
#
# The Arduino receives these coordinates directly and handles all internal
# transformations (IK lookup, base angle, etc.) in its firmware.

def load_homography(path):
    """
    Load a 3x3 homography matrix from a .npy file.
    Returns the matrix on success, or None on failure.q
    """
    try:
        H = np.load(path)
        logging.info(f"Homography loaded from '{path}'")
        return H
    except Exception as e:
        logging.error(f"Could not load homography: {e}")
        return None


def pixel_to_cm(cx, cy, H):
    """
    Convert pixel coordinates (cx, cy) to real-world coordinates (x_cm, y_cm)
    using the homography matrix. The output is in the coordinate system that
    the Arduino expects.
    """
    pt = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H)[0][0]
    return float(result[0]), float(result[1])


def in_workspace(x_cm, y_cm):
    """
    Check if the target position is within the arm's reachable workspace.
    Currently disabled (always returns True) because the Arduino firmware
    handles out-of-range coordinates by clamping to the nearest valid IK entry.
    """
    return True


# ==============================================================================
# SECTION 6: EXCLUSION ZONE HELPERS
# ==============================================================================
# These functions check whether a detection falls inside an exclusion zone
# and draw the exclusion zones as semi-transparent red overlays on the frame.

def point_in_exclusion_zone(cx, cy):
    """Return True if the point (cx, cy) falls inside any exclusion zone."""
    for (x1, y1, x2, y2) in EXCLUSION_ZONES:
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return True
    return False


def draw_exclusion_zones(frame):
    """Draw all exclusion zones as semi-transparent red rectangles on the frame."""
    overlay = frame.copy()
    for (x1, y1, x2, y2) in EXCLUSION_ZONES:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 200), -1)
        cv2.rectangle(frame,   (x1, y1), (x2, y2), (0, 0, 255),  2)
        cv2.putText(frame, "EXCLUSION", (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)


# ==============================================================================
# SECTION 7: SERIAL COMMUNICATION WITH ARDUINO
# ==============================================================================
# The ArmComm class manages the serial connection to the Arduino and provides
# high-level methods for each command the firmware supports.
#
# PROTOCOL:
#   - Commands are sent as newline-terminated ASCII strings.
#   - The Arduino replies with "DONE" on success or "ERROR" on failure.
#   - wait_done() blocks until one of these replies is received or timeout.
#
# AVAILABLE COMMANDS:
#   PICK <CLASS> <rx> <ry> <w> <h> <griprot>  — Full pick-and-place sequence
#   MOVE <rx> <ry>                              — Move arm to arbitrary position
#   GRIP                                        — Close the gripper
#   RELEASE                                     — Open the gripper
#   HOME                                        — Return to home position

class ArmComm:
    """Handles all serial communication with the Arduino (ItsyBitsy M4)."""

    def __init__(self, port, baudrate, timeout):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.conn = None
        self.connected = False
        self._connect()

    def _connect(self):
        """Establish serial connection. Waits 2 seconds for Arduino reset."""
        try:
            self.conn = serial.Serial(
                self.port, self.baudrate,
                timeout=1, write_timeout=5
            )
            time.sleep(2)   # Arduino resets when serial connects; wait for boot
            self.connected = True
            logging.info(f"Connected to {self.port}")
        except Exception as e:
            logging.error(f"Serial error: {e}")

    def send(self, cmd):
        """Send a raw command string to the Arduino. Returns True if sent."""
        if not self.connected:
            return False
        try:
            self.conn.write((cmd + "\n").encode())
            return True
        except Exception as e:
            logging.error(f"Send failed: {e}")
            return False

    def pick(self, class_name, x_cm, y_cm, w, h, gripper_rot):
        """
        Send PICK command with object class, position, size, and gripper angle.
        The Arduino executes the full pick-travel-drop-home sequence internally.
        """
        return self.send(f"PICK {class_name.upper()} {x_cm:.2f} {y_cm:.2f} {w} {h} {gripper_rot}")

    def move(self, x_cm, y_cm):
        """Send MOVE command to position the arm at (x_cm, y_cm)."""
        return self.send(f"MOVE {x_cm:.2f} {y_cm:.2f}")

    def grip(self):
        """Send GRIP command to close the gripper."""
        return self.send("GRIP")

    def release(self):
        """Send RELEASE command to open the gripper."""
        return self.send("RELEASE")

    def home(self):
        """Send HOME command to return the arm to its rest position."""
        return self.send("HOME")

    def wait_done(self, timeout=None):
        """
        Block until the Arduino sends "DONE" or "ERROR", or until timeout.
        Returns True if "DONE" was received, False otherwise.
        """
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
        """Close the serial connection."""
        if self.conn:
            self.conn.close()


# ==============================================================================
# SECTION 8: YOLO OBJECT DETECTION
# ==============================================================================
# Runs the YOLO model on each camera frame, draws bounding boxes, and returns
# the highest-confidence detection that is eligible for picking.
#
# Detections inside exclusion zones are skipped (drawn in grey with "ZONE" label).
# Non-recyclable objects are displayed on screen with "GARBAGE" label but are
# still returned as detections — the main loop decides whether to act on them.

def compute_gripper_rotation(w, h):
    """
    Estimate the optimal gripper rotation servo value from the bounding box
    aspect ratio. This helps the gripper align with the object's long axis.

    If the object is wider than tall (landscape): rotate gripper -> servo = 20
    If the object is taller than wide (portrait): no rotation    -> servo = 90
    """
    return 20 if w > h else 90


def detect_best(frame, model):
    """
    Run YOLO detection on a single camera frame.

    Returns:
        A dict with detection info (class_name, center, bbox, confidence, etc.)
        for the highest-confidence detection, or None if nothing was found.

    The returned dict contains:
        class_id:     YOLO class index (int)
        class_name:   Name sent to Arduino (e.g. "plastic", "garbage")
        display_name: Human-readable name for the HUD (e.g. "plastic", "glass")
        recycle:      True if recyclable, False otherwise
        center:       (cx, cy) pixel coordinates of detection center
        bbox:         (width, height) of bounding box in pixels
        gripper_rot:  Suggested gripper rotation servo value (20 or 90)
        confidence:   Detection confidence score (0.0 to 1.0)
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

        # Calculate center of the bounding box
        cx_det = (x1 + x2) // 2
        cy_det = (y1 + y2) // 2

        # Skip detections that fall inside exclusion zones (bins, arm, tripod)
        if point_in_exclusion_zone(cx_det, cy_det):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), 1)
            cv2.putText(frame, "ZONE", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
            continue

        # Unknown class ID (not in WASTE_CLASSES) — draw red box and skip
        if not info:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "UNKNOWN", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            continue

        name = info["name"]
        recycle = info["recycle"]

        # Determine the class name to send to the Arduino:
        #   Recyclable items keep their name (e.g. "plastic", "cardboard")
        #   Non-recyclable items are labeled as "garbage" for display
        arm_class = name if recycle else "garbage"
        color = CLASS_COLORS.get(name, CLASS_COLORS["default"])
        label = f"{name} {conf_val:.2f}" if recycle else f"GARBAGE ({name}) {conf_val:.2f}"

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Track the highest-confidence detection as the pick candidate
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


# ==============================================================================
# SECTION 9: FEEDBACK LOOP (BACKGROUND THREAD) — CURRENTLY UNUSED
# ==============================================================================
# This function was previously used for non-bin classes to monitor when the
# object disappeared from the camera view after the arm moved to pick it.
# Since non-bin classes are now ignored (no PICK sent), this function is kept
# for potential future use but is not called from the main loop.

def feedback_loop(camera, camera_lock, model, arm, target_class, stop_event, done_event):
    """
    Monitor the camera for object disappearance after the arm moves to pick it.

    Args:
        camera:       cv2.VideoCapture instance
        camera_lock:  threading.Lock to synchronize camera access with main loop
        model:        YOLO model instance
        arm:          ArmComm instance (not used directly, but available)
        target_class: YOLO class ID of the object being picked
        stop_event:   threading.Event — set by main loop to abort this thread
        done_event:   threading.Event — set by this thread when object disappears
    """
    disappear = 0

    while not stop_event.is_set():
        with camera_lock:
            ret, frame = camera.read()
        if not ret:
            time.sleep(0.1)
            continue

        # Run detection to check if the target object is still visible
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
            disappear = 0    # Object still there — reset counter
        else:
            disappear += 1   # Object gone — increment counter
            if disappear >= DISAPPEAR_THRESHOLD:
                logging.info("Object gone — pick complete")
                done_event.set()
                break

        time.sleep(0.1)


# ==============================================================================
# SECTION 10: MAIN APPLICATION LOOP
# ==============================================================================
# This is the entry point. It initializes all hardware (camera, model, serial),
# then runs the continuous detect-confirm-pick loop.
#
# STATE MACHINE:
#   IDLE    — No object detected. Arm is at home. Camera is scanning.
#   CONFIRM — Object detected, waiting for N consecutive frame confirmations.
#   BUSY    — Pick-and-place cycle in progress. Detection is paused.
#
# PICK FLOW (bin classes only: plastic, metal, cardboard):
#   1. Python detects and confirms the object over multiple frames.
#   2. Python checks if the class is a bin class (plastic, metal, cardboard).
#   3. If YES: Python sends PICK command with coordinates to Arduino.
#      Arduino handles the ENTIRE sequence (open, descend, grab, lift, bin,
#      drop, home) and sends DONE when complete. Python waits.
#   4. If NO: Python logs "Ignoring non-bin class" and resets the confirmation
#      counter. No command is sent to the Arduino. The arm stays at home.
#   5. Python resumes detection.
#
# KEYBOARD CONTROLS (only active when arm is not busy):
#   Q — Quit the application
#   H — Send arm to home position
#   G — Close gripper (manual test)
#   R — Open gripper (manual test)
#
# SAFETY:
#   The entire main logic is wrapped in try/finally to ensure the camera,
#   serial port, and OpenCV windows are properly closed even if an unexpected
#   error occurs. This prevents resource leaks after a crash.

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger("WasteSorter")

    # Declare hardware references outside try block so finally can access them
    cap = None
    arm = None

    try:
        # -- Load YOLO model --
        try:
            model = YOLO(MODEL_PATH)
            logger.info(f"Model loaded. Classes: {model.names}")
        except Exception as e:
            logger.error(f"Model load error: {e}")
            return

        # -- Load homography matrix --
        H = load_homography(HOMOGRAPHY_PATH)
        if H is None:
            logger.error("Cannot run without homography. Run calibrate_homography.py first.")
            return

        # -- Open camera (try indices 0, 1, 2 in case of multiple cameras) --
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

        # -- Connect to Arduino via serial --
        arm = ArmComm(SERIAL_PORT, SERIAL_BAUDRATE, SERIAL_TIMEOUT)
        if not arm.connected:
            logger.warning("Arm not connected — commands will be simulated")

        # -- State variables --
        busy           = False          # True when a pick-and-place cycle is running
        busy_start     = 0              # Timestamp when the current cycle started
        confirm        = 0              # Consecutive detection confirmation counter
        last_detection = None           # Most recent detection result dict
        last_detect_time = time.time()  # Timestamp of last detection (for idle timeout)

        # These are kept for potential future use with the feedback loop
        stop_fb = threading.Event()
        done_fb = threading.Event()
        fb_thread = None
        camera_lock = threading.Lock()

        logger.info("System READY. Press Q to quit, H for home, G to grip, R to release.")

        # ======================================================================
        # MAIN LOOP — runs until user presses Q or camera fails
        # ======================================================================
        while True:
            # Read a frame from the camera (synchronized with any background threads)
            with camera_lock:
                ret, frame = cap.read()
            if not ret:
                break

            # Draw exclusion zones as red overlays on the frame
            draw_exclusion_zones(frame)

            # ------------------------------------------------------------------
            # CHECK: Pick operation timeout
            # ------------------------------------------------------------------
            # If a pick cycle has been running longer than PICK_TIMEOUT seconds,
            # force-reset the arm to HOME to prevent it from getting stuck.
            if busy and (time.time() - busy_start) > PICK_TIMEOUT:
                logger.error("Pick timeout — resetting arm")
                arm.home()
                arm.wait_done(10)
                busy = False
                confirm = 0

            # ------------------------------------------------------------------
            # DETECTION PHASE (only runs when arm is idle)
            # ------------------------------------------------------------------
            if not busy:
                det = detect_best(frame, model)
                if det:
                    confirm += 1
                    last_detection = det
                    last_detect_time = time.time()

                    # Show confirmation progress on the video feed
                    cv2.putText(frame,
                                f"Confirming {det['display_name']} {confirm}/{DETECT_CONFIRM_FRAMES}",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    # Once enough consecutive frames confirm the detection, act on it
                    if confirm >= DETECT_CONFIRM_FRAMES:
                        cx, cy = det["center"]
                        w, h   = det["bbox"]

                        # Convert pixel coordinates to the coordinate system
                        # that the Arduino expects
                        x_cm, y_cm = pixel_to_cm(cx, cy, H)

                        logger.info(
                            f"Picking {det['display_name']} (->arm class: {det['class_name']}) "
                            f"at pixel=({cx},{cy}) -> ({x_cm:.2f}, {y_cm:.2f}) cm"
                        )

                        # Verify the target is within the workspace
                        if not in_workspace(x_cm, y_cm):
                            logger.warning(f"({x_cm:.2f}, {y_cm:.2f}) outside workspace — skipping")
                            confirm = 0
                        else:
                            # Define which classes the Arduino handles end-to-end
                            # These are the classes that have bin positions in the
                            # Arduino firmware (BIN_PLASTIC, BIN_METAL, BIN_CARDBOARD)
                            bin_classes = {"plastic", "metal", "cardboard"}

                            # --------------------------------------------------
                            # NON-BIN CLASSES: Ignore, do not send to Arduino
                            # --------------------------------------------------
                            # If the detected class is not a bin class (e.g.
                            # biodegradable, glass, paper), we simply log it
                            # and reset the confirmation counter. No PICK
                            # command is sent to the Arduino, so the arm
                            # stays at home and does not move.
                            if det["class_name"].lower() not in bin_classes:
                                logger.info(f"Ignoring non-bin class: {det['display_name']}")
                                confirm = 0

                            # --------------------------------------------------
                            # BIN CLASSES: Send PICK, Arduino handles everything
                            # --------------------------------------------------
                            # For plastic, metal, and cardboard, send the PICK
                            # command. The Arduino firmware runs the full
                            # sequence: open gripper, descend, grab, lift,
                            # travel to bin, drop, jiggle, return home, DONE.
                            # Python blocks on wait_done() until complete.
                            elif arm.pick(det["class_name"], x_cm, y_cm, w, h, det["gripper_rot"]):
                                busy = True
                                busy_start = time.time()
                                confirm = 0
                                logger.info(
                                    f"{det['class_name']} -> waiting for Arduino to complete "
                                    "full pick+bin+home sequence..."
                                )
                                # Block until Arduino sends DONE (up to 60 seconds)
                                if not arm.wait_done(60):
                                    logger.warning("Bin sequence timed out")
                                    arm.home()
                                    arm.wait_done(10)
                                else:
                                    logger.info("Bin sequence complete")
                                busy = False

                            # --------------------------------------------------
                            # PICK COMMAND FAILED TO SEND
                            # --------------------------------------------------
                            else:
                                logger.error("Failed to send PICK command")
                else:
                    # No detection this frame — reset confirmation counter
                    confirm = 0

            # ------------------------------------------------------------------
            # IDLE TIMEOUT — send arm home if nothing detected for a while
            # ------------------------------------------------------------------
            if not busy and (time.time() - last_detect_time) > IDLE_HOME_TIMEOUT:
                logger.info("Idle timeout — sending HOME")
                arm.home()
                arm.wait_done(10)
                last_detect_time = time.time()

            # ------------------------------------------------------------------
            # HUD (Heads-Up Display) — draw status info on the video feed
            # ------------------------------------------------------------------
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

            # ------------------------------------------------------------------
            # KEYBOARD CONTROLS (only active when arm is not busy)
            # ------------------------------------------------------------------
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

    # ==========================================================================
    # CLEANUP — always runs, even if an exception occurred above
    # ==========================================================================
    # This ensures hardware resources are properly released regardless of
    # whether the program exited normally (Q key) or due to an error.
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            if arm is not None:
                arm.close()
        except Exception:
            pass


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    main()