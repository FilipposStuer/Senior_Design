"""
================================================================================
  robot_vision.py — Waste Sorter with Homography-Based Coordinate Conversion
================================================================================

  This script uses a YOLO OBB model to detect recyclable waste on a work
  surface, converts the detected pixel coordinates to real-world coordinates
  using a homography matrix, and commands a robotic arm (via serial) to pick
  up and sort the items into designated bins.

  SYSTEM OVERVIEW:
      Camera (USB)  -->  YOLO OBB Detection  -->  Homography Transform  -->  Serial
      (640x480 px)       (best.pt OBB model)      (pixel -> inches)          (Arduino)

  COORDINATE PIPELINE:
      pixel (cx, cy)
        --> homography --> real-world position in inches (table frame)
        --> Arduino receives these values and handles IK internally

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

from __future__ import annotations

import cv2
import serial
import time
import logging
import threading
import math
import numpy as np
from ultralytics import YOLO


# ==============================================================================
# SECTION 1: SYSTEM CONFIGURATION
# ==============================================================================

MODEL_PATH       = "best.pt"
HOMOGRAPHY_PATH  = "homography.npy"
CONFIDENCE_THRESHOLD = 0.30
CAMERA_INDEX     = 0
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
SERIAL_PORT      = "COM5"
SERIAL_BAUDRATE  = 9600
SERIAL_TIMEOUT   = 30


# ==============================================================================
# SECTION 2: TIMING AND DETECTION PARAMETERS
# ==============================================================================

DETECT_CONFIRM_FRAMES = 3
DISAPPEAR_THRESHOLD   = 5
PICK_TIMEOUT          = 60
IDLE_HOME_TIMEOUT     = 15

WS_X_MIN, WS_X_MAX = -999.0, 999.0
WS_Y_MIN, WS_Y_MAX = -999.0, 999.0


# ==============================================================================
# SECTION 3: WASTE CLASS DEFINITIONS
# ==============================================================================
#   0: biodegradable, 1: cardboard, 2: glass, 3: metal, 4: paper, 5: plastic
#
#   recycle: True  -> arm picks and sorts to class bin
#   recycle: False -> detected and shown on screen, arm does NOT pick

WASTE_CLASSES = {
    1: {"name": "cardboard",     "recycle": True},
    4: {"name": "paper",         "recycle": False},
    5: {"name": "plastic",       "recycle": True},
    0: {"name": "biodegradable", "recycle": False},
    2: {"name": "glass",         "recycle": False},
    3: {"name": "metal",         "recycle": True},
}

CLASS_COLORS = {
    "plastic":       (255, 80,  80),
    "paper":         (200, 200, 255),
    "cardboard":     (50,  165, 255),
    "biodegradable": (50,  200,  50),
    "glass":         (200, 200,  50),
    "metal":         (180, 180, 180),
    "default":       (128, 128, 128),
}


# ==============================================================================
# SECTION 4: CAMERA EXCLUSION ZONES
# ==============================================================================

EXCLUSION_ZONES = [
    (  0,   0, 120, 480),
    (520,   0, 640, 480),
    (  0,   0, 640,  40),
    (  0, 400, 640, 480),
]


# ==============================================================================
# SECTION 5: HOMOGRAPHY COORDINATE CONVERSION
# ==============================================================================

def load_homography(path):
    """Load a 3x3 homography matrix from a .npy file."""
    try:
        H = np.load(path)
        logging.info(f"Homography loaded from '{path}'")
        return H
    except Exception as e:
        logging.error(f"Could not load homography: {e}")
        return None


def pixel_to_world(cx, cy, H):
    """Convert pixel coordinates (cx, cy) to real-world inches via homography."""
    pt = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H)[0][0]
    return float(result[0]), float(result[1])


def in_workspace(x, y):
    """Workspace check — disabled, Arduino handles out-of-range gracefully."""
    return True


# ==============================================================================
# SECTION 6: EXCLUSION ZONE HELPERS
# ==============================================================================

def point_in_exclusion_zone(cx, cy):
    """Return True if (cx, cy) falls inside any exclusion zone."""
    for (x1, y1, x2, y2) in EXCLUSION_ZONES:
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return True
    return False


def draw_exclusion_zones(frame):
    """Draw exclusion zones as semi-transparent red overlays."""
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

class ArmComm:
    """Handles all serial communication with the Arduino (ItsyBitsy M4)."""

    def __init__(self, port, baudrate, timeout):
        self.port      = port
        self.baudrate  = baudrate
        self.timeout   = timeout
        self.conn      = None
        self.connected = False
        self._connect()

    def _connect(self):
        """Establish serial connection. Waits 2 s for Arduino reset."""
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
        """Send a raw command string. Returns True if sent."""
        if not self.connected:
            return False
        try:
            self.conn.write((cmd + "\n").encode())
            return True
        except Exception as e:
            logging.error(f"Send failed: {e}")
            return False

    def pick(self, class_name, x, y, w, h, angle_deg):
        """
        Send PICK command with class, position (inches), bbox size, and
        raw OBB angle in degrees.
        Format: PICK <CLASS> <x> <y> <w> <h> <angle_deg>
        """
        return self.send(
            f"PICK {class_name.upper()} {x:.2f} {y:.2f} {w} {h} {angle_deg:.1f}"
        )

    def move(self, x, y):
        return self.send(f"MOVE {x:.2f} {y:.2f}")

    def grip(self):
        return self.send("GRIP")

    def release(self):
        return self.send("RELEASE")

    def home(self):
        return self.send("HOME")

    def wait_done(self, timeout=None):
        """Block until Arduino replies DONE or ERROR, or timeout expires."""
        deadline = time.time() + (timeout or self.timeout)
        while time.time() < deadline:
            if self.conn and self.conn.in_waiting:
                line = self.conn.readline().decode(errors='replace').strip().upper()
                if "DONE"  in line:
                    return True
                if "ERROR" in line:
                    return False
            time.sleep(0.05)
        return False

    def close(self):
        if self.conn:
            self.conn.close()


# ==============================================================================
# SECTION 8: GRIPPER ROTATION FROM OBB ANGLE
# ==============================================================================

def obb_angle_to_gripper_servo(angle_rad):
    """
    Convert an OBB rotation angle (radians) to a gripper rotation servo value.

    The OBB angle is the rotation of the bounding box's long axis from the
    horizontal. We want the gripper to align perpendicular to the long axis
    (i.e. grab across the short axis).

    Servo range:
        90  -> gripper at 0°   (no rotation, portrait orientation)
        20  -> gripper at 90°  (rotated, landscape orientation)

    The mapping is continuous: angle_deg in [0, 90] maps linearly to [90, 20].
    """
    angle_deg = abs(math.degrees(angle_rad)) % 90.0
    # Linear map: 0° -> servo 90, 90° -> servo 20
    servo = int(round(90.0 - (angle_deg / 90.0) * 70.0))
    return max(20, min(90, servo))


# ==============================================================================
# SECTION 9: OBB OBJECT DETECTION
# ==============================================================================

def draw_obb(frame, points, color, label):
    """Draw an oriented bounding box (4 corner points) and label on the frame."""
    pts = points.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
    x, y = int(points[0][0]), int(points[0][1]) - 8
    cv2.putText(frame, label, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def detect_best(frame, model):
    """
    Run YOLO OBB detection on a single camera frame.

    Uses r.obb (oriented bounding boxes) to get:
      - 4-corner polygon for drawing
      - xywhr for center, size, and rotation angle
      - Continuous gripper rotation from the OBB angle

    Returns:
        Dict with detection info, or None if nothing found.

        Keys:
            class_id     — YOLO class index
            class_name   — name sent to Arduino ("plastic", "metal", etc.)
            display_name — human-readable name for HUD
            recycle      — True if this class is picked
            center       — (cx, cy) pixel center
            bbox         — (w, h) bounding box pixel dimensions
            gripper_rot  — gripper rotation servo value (20–90)
            confidence   — detection confidence (0.0–1.0)
    """
    results = model.predict(
        frame, conf=CONFIDENCE_THRESHOLD,
        iou=0.45, verbose=False, stream=True
    )
    r = next(results, None)

    if r is None or r.obb is None:
        return None

    best      = None
    best_conf = 0.0

    # r.obb.xyxyxyxy  — (N, 4, 2) corner points
    # r.obb.xywhr     — (N, 5)    cx, cy, w, h, angle_rad
    # r.obb.conf      — (N,)      confidence scores
    # r.obb.cls       — (N,)      class IDs

    corners_all = r.obb.xyxyxyxy.cpu().numpy()   # shape (N, 4, 2)
    xywhr_all   = r.obb.xywhr.cpu().numpy()       # shape (N, 5)
    confs       = r.obb.conf.cpu().numpy()
    classes     = r.obb.cls.cpu().numpy().astype(int)

    for idx, (corners, xywhr, conf, c) in enumerate(
        zip(corners_all, xywhr_all, confs, classes)
    ):
        info = WASTE_CLASSES.get(int(c))
        conf_val = float(conf)

        cx_det = int(xywhr[0])
        cy_det = int(xywhr[1])
        w_px   = int(xywhr[2])
        h_px   = int(xywhr[3])
        angle  = float(xywhr[4])   # rotation angle in radians

        # Skip detections inside exclusion zones
        if point_in_exclusion_zone(cx_det, cy_det):
            draw_obb(frame, corners, (60, 60, 60), "ZONE")
            continue

        if not info:
            draw_obb(frame, corners, (0, 0, 255), "UNKNOWN")
            continue

        name    = info["name"]
        recycle = info["recycle"]

        arm_class = name if recycle else "garbage"
        color     = CLASS_COLORS.get(name, CLASS_COLORS["default"])
        label     = f"{name} {conf_val:.2f}" if recycle else f"GARBAGE ({name}) {conf_val:.2f}"

        # Draw OBB polygon and angle indicator
        draw_obb(frame, corners, color, label)

        # Draw a small line showing the object's orientation
        angle_display = math.degrees(angle)
        lx = int(cx_det + 30 * math.cos(angle))
        ly = int(cy_det + 30 * math.sin(angle))
        cv2.arrowedLine(frame, (cx_det, cy_det), (lx, ly), color, 1, tipLength=0.3)
        cv2.putText(frame, f"{angle_display:.1f}deg", (cx_det + 5, cy_det - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        if conf_val > best_conf:
            best_conf = conf_val
            best = {
                "class_id":    int(c),
                "class_name":  arm_class,
                "display_name": name,
                "recycle":     recycle,
                "center":      (cx_det, cy_det),
                "bbox":        (w_px, h_px),
                "angle_deg":   round(math.degrees(angle), 1),
                "confidence":  conf_val,
            }

    return best


# ==============================================================================
# SECTION 10: MAIN APPLICATION LOOP
# ==============================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger("WasteSorter")

    cap = None
    arm = None

    try:
        # -- Load YOLO OBB model --
        try:
            model = YOLO(MODEL_PATH)
            logger.info(f"Model loaded. Classes: {model.names}")
        except Exception as e:
            logger.error(f"Model load error: {e}")
            return

        # -- Load homography --
        H = load_homography(HOMOGRAPHY_PATH)
        if H is None:
            logger.error("Cannot run without homography. Run calibrate_homography.py first.")
            return

        # -- Open camera --
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

        # -- Connect to Arduino --
        arm = ArmComm(SERIAL_PORT, SERIAL_BAUDRATE, SERIAL_TIMEOUT)
        if not arm.connected:
            logger.warning("Arm not connected — commands will be simulated")

        # -- State variables --
        busy             = False
        busy_start       = 0
        confirm          = 0
        last_detection   = None
        last_detect_time = time.time()
        camera_lock      = threading.Lock()

        # Classes the Arduino has bin positions for
        bin_classes = {"plastic", "metal", "cardboard"}

        logger.info("System READY. Press Q to quit, H for home, G to grip, R to release.")

        while True:
            with camera_lock:
                ret, frame = cap.read()
            if not ret:
                break

            draw_exclusion_zones(frame)

            # -- Pick timeout --
            if busy and (time.time() - busy_start) > PICK_TIMEOUT:
                logger.error("Pick timeout — resetting arm")
                arm.home()
                arm.wait_done(10)
                busy    = False
                confirm = 0

            # -- Detection (only when idle) --
            if not busy:
                det = detect_best(frame, model)
                if det:
                    confirm         += 1
                    last_detection   = det
                    last_detect_time = time.time()

                    cv2.putText(frame,
                                f"Confirming {det['display_name']} {confirm}/{DETECT_CONFIRM_FRAMES}",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    if confirm >= DETECT_CONFIRM_FRAMES:
                        cx, cy = det["center"]
                        w, h   = det["bbox"]
                        x, y   = pixel_to_world(cx, cy, H)

                        logger.info(
                            f"Picking {det['display_name']} "
                            f"(arm class: {det['class_name']}, "
                            f"gripper: {det['gripper_rot']}°) "
                            f"pixel=({cx},{cy}) -> ({x:.2f}, {y:.2f}) in"
                        )

                        if not in_workspace(x, y):
                            logger.warning(f"({x:.2f}, {y:.2f}) outside workspace — skipping")
                            confirm = 0

                        elif det["class_name"].lower() not in bin_classes:
                            # Non-bin class — ignore, do not send to Arduino
                            logger.info(f"Ignoring non-bin class: {det['display_name']}")
                            confirm = 0

                        elif arm.pick(det["class_name"], x, y, w, h, det["angle_deg"]):
                            # Bin class — Arduino handles full sequence
                            busy       = True
                            busy_start = time.time()
                            confirm    = 0
                            logger.info(
                                f"{det['class_name']} -> waiting for Arduino "
                                "to complete pick+bin+home sequence..."
                            )
                            if not arm.wait_done(60):
                                logger.warning("Bin sequence timed out")
                                arm.home()
                                arm.wait_done(10)
                            else:
                                logger.info("Bin sequence complete")
                            busy = False

                        else:
                            logger.error("Failed to send PICK command")
                else:
                    confirm = 0

            # -- Idle timeout --
            if not busy and (time.time() - last_detect_time) > IDLE_HOME_TIMEOUT:
                logger.info("Idle timeout — sending HOME")
                arm.home()
                arm.wait_done(10)
                last_detect_time = time.time()

            # -- HUD --
            status = "BUSY" if busy else "READY"
            color  = (0, 165, 255) if busy else (0, 255, 0)
            cv2.putText(frame, f"Status: {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if last_detection and not busy:
                cv2.putText(frame,
                            f"Target: {last_detection['display_name']} "
                            f"({last_detection['class_name']}) "
                            f"angle={last_detection['angle_deg']}° "
                            f"{last_detection['confidence']:.2f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("Waste Sorter", frame)

            # -- Keyboard --
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

    finally:
        for resource, fn in [(cap, lambda r: r.release()),
                             (None, lambda _: cv2.destroyAllWindows()),
                             (arm, lambda r: r.close())]:
            try:
                if resource is not None:
                    fn(resource)
                else:
                    fn(None)
            except Exception:
                pass


if __name__ == "__main__":
    main()
