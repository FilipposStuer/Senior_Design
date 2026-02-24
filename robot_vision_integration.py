"""
Robot Vision - Raspberry Pi Side

PURPOSE:
    This script runs on the Raspberry Pi and handles TWO things only:
        1. Capture video from the camera and detect waste objects using AI (YOLO)
        2. Tell the ItsyBitsy M4 microcontroller WHAT to pick and WHERE it is

    The ItsyBitsy M4 is responsible for actually moving the robot arm.
    This separation keeps each device focused on what it does best:
        - Raspberry Pi  → AI / Vision (needs a powerful processor)
        - ItsyBitsy M4  → Motor control (needs precise real-time timing)

HOW IT COMMUNICATES:
    Raspberry Pi sends ONE line over USB serial:
        "PICK PLASTIC 320 240 80 60"
         ^^^^  ^^^^^^^ ^^^ ^^^  ^^  ^^
         cmd   class   cx  cy  bbox_w  bbox_h

    ItsyBitsy replies when finished:
        "DONE"  → success
        "ERROR" → something went wrong on the arm side
"""

import cv2        # OpenCV: used to capture camera frames and draw boxes on screen
import serial     # PySerial: used to send/receive data over USB to the ItsyBitsy
import time       # Used for delays and timeouts
import logging    # Better than print() — gives timestamps and log levels
import sys        # Used to direct log output to the terminal
from ultralytics import YOLO  # The AI library that runs our waste detection model


# CONFIGURATION
# All settings are here at the top so they are
# easy to find and change without digging into
# the code logic below.

MODEL_PATH           = "best.pt"   # Path to the trained YOLO model file (must be in same folder)
CONFIDENCE_THRESHOLD = 0.5         # Only accept detections above 50% confidence (0.0 - 1.0)

CAMERA_INDEX  = 0      # 0 = first camera connected. Try 1 or 2 if wrong camera opens
FRAME_WIDTH   = 640    # Camera resolution width  (pixels)
FRAME_HEIGHT  = 480    # Camera resolution height (pixels)

SERIAL_PORT     = "/dev/ttyACM0"  # USB port where ItsyBitsy appears on Raspberry Pi OS
SERIAL_BAUDRATE = 9600            # Communication speed (must match ItsyBitsy firmware setting)
SERIAL_TIMEOUT  = 30              # Max seconds to wait for ItsyBitsy to finish a pick sequence

# Waste class definitions — must match the classes your YOLO model was trained on.
# Key   = class ID number that YOLO outputs
# name  = human-readable label
# bin   = which physical bin this waste type goes into
WASTE_CLASSES = {
    0: {"name": "general", "bin": "landfill"},
    1: {"name": "paper",   "bin": "paper"},
    2: {"name": "plastic", "bin": "plastic"},
}

# Colors used to draw bounding boxes on screen for each waste type (BGR format, not RGB)
CLASS_COLORS = {
    "general": (128, 128, 128),  # Gray
    "paper":   (255, 255, 255),  # White
    "plastic": (255,   0,   0),  # Blue
}


# LOGGING SETUP
# Instead of using print(), we use Python's
# logging module. This gives us:
#   - Timestamps on every message
#   - Log levels: INFO, WARNING, ERROR, CRITICAL
#   - Output to BOTH the terminal AND a log file

logging.basicConfig(
    level=logging.INFO,   # Show INFO and above (DEBUG messages will be hidden)
    format="%(asctime)s [%(levelname)s] %(message)s",  # e.g. "2024-01-15 10:30:01 [INFO] ..."
    handlers=[
        logging.StreamHandler(sys.stdout),          # Print to terminal
        logging.FileHandler("robot_vision.log"),    # Also save to robot_vision.log file
    ]
)
logger = logging.getLogger("RobotVision")  # Create a named logger for this module


# SERIAL COMMUNICATION CLASS
# This class handles everything related to
# talking with the ItsyBitsy M4 over USB.

class ItsyBitsyComm:
    """Handles serial communication with the ItsyBitsy M4."""

    def __init__(self, port: str, baudrate: int, timeout: int):
        """
        Initialize the communication object and attempt to connect.

        Parameters:
            port     : USB port name, e.g. "/dev/ttyACM0"
            baudrate : Communication speed, e.g. 9600
            timeout  : Max seconds to wait for a response from ItsyBitsy
        """
        self.port      = port
        self.baudrate  = baudrate
        self.timeout   = timeout
        self.conn      = None      # Will hold the serial connection object
        self.connected = False     # Flag so other parts of the code know if we're connected
        self._connect()            # Try to connect immediately on creation

    def _connect(self):
        """
        Open the USB serial port to the ItsyBitsy.
        If it fails (e.g. ItsyBitsy not plugged in), we log the error
        and set self.connected = False so the rest of the code can handle it gracefully.
        """
        try:
            self.conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1,        # How long to wait when READING a response (per read call)
                write_timeout=1,  # How long to wait when WRITING a command
            )
            time.sleep(2)  # Give the ItsyBitsy time to finish booting after USB connection
            self.connected = True
            logger.info("Connected to ItsyBitsy on %s", self.port)
        except serial.SerialException as exc:
            # Common causes: ItsyBitsy not plugged in, wrong port, no permission
            logger.error("Could not connect to ItsyBitsy: %s", exc)
            self.connected = False

    def send_pick_command(
        self,
        class_name: str,  # e.g. "plastic"
        cx: int,          # X pixel coordinate of object center in the camera frame
        cy: int,          # Y pixel coordinate of object center in the camera frame
        bbox_w: int,      # Width of the detection bounding box (pixels) — used to estimate depth
        bbox_h: int,      # Height of the detection bounding box (pixels) — used to estimate depth
    ) -> bool:
        """
        Send a PICK command to the ItsyBitsy and wait until it finishes.

        The command format sent over serial is:
            "PICK PLASTIC 320 240 80 60\n"

        The ItsyBitsy firmware reads this, calculates servo angles,
        executes the full pick-and-place sequence, then replies:
            "DONE\n"  → everything went fine
            "ERROR\n" → something failed on the arm side

        Returns True if the ItsyBitsy confirmed success, False otherwise.
        """
        # Safety check — don't try to send if we are not connected
        if not self.connected or not self.conn or not self.conn.is_open:
            logger.error("Serial not available.")
            return False

        # Build the command string (class name in uppercase for consistency)
        command = f"PICK {class_name.upper()} {cx} {cy} {bbox_w} {bbox_h}\n"

        try:
            # Send the command bytes over USB serial
            self.conn.write(command.encode())
            logger.info("Sent: %s", command.strip())

            # Now WAIT for the ItsyBitsy to respond.
            # The arm sequence can take 10-20 seconds, so we wait up to SERIAL_TIMEOUT seconds.
            deadline = time.time() + self.timeout
            while time.time() < deadline:
                if self.conn.in_waiting:  # Check if any bytes arrived from ItsyBitsy
                    response = self.conn.readline().decode(errors="replace").strip().upper()
                    logger.info("ItsyBitsy: %s", response)

                    if response == "DONE":
                        return True    # Success! Arm finished the sequence
                    elif response == "ERROR":
                        logger.error("ItsyBitsy reported an error.")
                        return False   # Arm encountered a problem

                    # Any other message (e.g. debug prints from ItsyBitsy) — log and keep waiting

                time.sleep(0.05)  # Small pause to avoid hammering the CPU in this loop

            # If we reach here, ItsyBitsy never responded in time
            logger.error("Timeout waiting for ItsyBitsy response.")
            return False

        except serial.SerialException as exc:
            logger.error("Serial error: %s", exc)
            return False

    def close(self):
        """Close the serial port cleanly when the program exits."""
        if self.conn and self.conn.is_open:
            self.conn.close()
            logger.info("Serial connection closed.")


# MAIN FUNCTION
# This is the entry point of the program.
# It sets everything up, then runs the main
# camera + detection loop until the user quits.

def main():
    logger.info("=" * 60)
    logger.info("ROBOT VISION - RASPBERRY PI")
    logger.info("Vision + Detection only. Arm control → ItsyBitsy M4")
    logger.info("=" * 60)

    # Step 1: Load the AI model
    # YOLO loads the trained weights from best.pt.
    # This file must exist in the same folder as this script.
    logger.info("Loading model from '%s'...", MODEL_PATH)
    try:
        model = YOLO(MODEL_PATH)
        logger.info("Model loaded. Classes: %s", getattr(model, "names", "unknown"))
    except Exception as exc:
        logger.critical("Failed to load model: %s", exc)
        return  # Cannot continue without the model

    # Step 2: Open the camera
    # cv2.VideoCapture(0) opens the first connected camera.
    # We set the resolution to 640x480 for a balance of speed and detail.
    logger.info("Opening camera (index %d)...", CAMERA_INDEX)
    camera = cv2.VideoCapture(CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not camera.isOpened():
        logger.critical("Could not open camera. Check connection or index.")
        return

    logger.info("Camera ready: %dx%d", FRAME_WIDTH, FRAME_HEIGHT)

    # Step 3: Connect to ItsyBitsy
    # If the ItsyBitsy is not connected, the program still runs
    # but pick commands will fail gracefully (logged as errors).
    itsybitsy = ItsyBitsyComm(SERIAL_PORT, SERIAL_BAUDRATE, SERIAL_TIMEOUT)
    if not itsybitsy.connected:
        logger.warning("Running without ItsyBitsy connection.")

    logger.info("System ready!  Controls: [P] pick  [Q] quit")
    logger.info("-" * 60)

    # Variables used in the main loop 
    last_detection = None    # Stores the best detection from the latest frame
    frame_count    = 0       # Total frames processed (used for FPS calculation)
    fps_time       = time.time()
    fps            = 0.0
    busy           = False   # True while ItsyBitsy is running a pick sequence
                             # We skip detection while busy to avoid commanding again mid-sequence

    # Main loop: runs once per camera frame
    try:
        while True:

            # Read one frame from the camera
            ret, frame = camera.read()
            if not ret:
                logger.error("Failed to capture frame.")
                break

            # Count frames and calculate FPS every 30 frames
            frame_count += 1
            if frame_count % 30 == 0:
                now      = time.time()
                fps      = 30.0 / max(now - fps_time, 1e-6)
                fps_time = now

            # Detection (only when arm is not busy)
            # While the ItsyBitsy is executing a pick sequence, we pause
            # detection so we don't try to pick a second object mid-sequence.
            if not busy:
                # Run the YOLO model on the current frame.
                # results[0].boxes.data contains all detections as a list of:
                # [x1, y1, x2, y2, confidence, class_id]
                results = model(frame, verbose=False)[0]
                best_detection = None
                best_conf      = 0.0

                for det in results.boxes.data:
                    x1, y1, x2, y2, conf, class_id = det
                    conf     = float(conf)
                    class_id = int(class_id)

                    # Skip low-confidence detections
                    if conf < CONFIDENCE_THRESHOLD:
                        continue

                    # Skip classes not defined in our waste dictionary
                    if class_id not in WASTE_CLASSES:
                        continue

                    # Convert float coordinates to integers for drawing
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    class_info = WASTE_CLASSES[class_id]
                    class_name = class_info["name"]
                    color      = CLASS_COLORS.get(class_name, (0, 255, 0))

                    # Calculate the center pixel of the detected object.
                    # This is what we send to the ItsyBitsy as the pick target.
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Draw bounding box and labels on frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)    # Box around object
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)        # Red dot at center
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 8),                # Class + confidence
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, f"Bin: {class_info['bin']}", (x1, y2 + 20),  # Bin label
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                    # Keep only the detection with the highest confidence
                    # so we always target the most certain object in the frame
                    if conf > best_conf:
                        best_conf = conf
                        best_detection = {
                            "center":     (cx, cy),
                            "bbox_size":  (x2 - x1, y2 - y1),  # ItsyBitsy uses this to estimate depth
                            "class_id":   class_id,
                            "class_name": class_name,
                            "confidence": conf,
                        }

                last_detection = best_detection  # Save for use when [P] is pressed

            # HUD (Heads-Up Display) 
            # Overlay status info on the video frame before displaying it
            status = "BUSY - Arm moving..." if busy else "READY"
            cv2.putText(frame, f"Status: {status}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 165, 255) if busy else (0, 255, 0), 2)  # Orange if busy, green if ready
            cv2.putText(frame, f"FPS: {fps:.1f}",
                        (FRAME_WIDTH - 90, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Controls: [P] pick  [Q] quit",
                        (10, FRAME_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Show the current pick target if one has been detected
            if last_detection and not busy:
                cv2.putText(
                    frame,
                    f"Target: {last_detection['class_name']} ({last_detection['confidence']:.2f})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )

            # Display the annotated frame in a window on screen
            cv2.imshow("Robot Vision - Waste Sorting", frame)

            # Keyboard input (1ms wait per frame) 
            key = cv2.waitKey(1) & 0xFF

            # [Q] — Quit the program cleanly
            if key == ord("q"):
                logger.info("Quit requested.")
                break

            # [P] — Send pick command to ItsyBitsy
            # Only triggers if: a detection exists AND the arm is not already busy
            elif key == ord("p") and last_detection and not busy:
                cx, cy = last_detection["center"]
                bw, bh = last_detection["bbox_size"]
                name   = last_detection["class_name"]

                logger.info("Sending PICK command → ItsyBitsy: %s at (%d, %d)", name, cx, cy)
                busy = True  # Pause detection while arm is moving

                # This call BLOCKS until ItsyBitsy sends "DONE" or "ERROR"
                # or until SERIAL_TIMEOUT seconds have passed
                success = itsybitsy.send_pick_command(name, cx, cy, bw, bh)

                if success:
                    logger.info("ItsyBitsy completed pick-and-place successfully.")
                else:
                    logger.error("ItsyBitsy pick-and-place failed or timed out.")

                last_detection = None  # Clear so we don't re-pick the same object
                busy = False           # Arm is free again, resume detection

    except KeyboardInterrupt:
        # Ctrl+C pressed in terminal — exit gracefully
        logger.info("Interrupted by user.")

    finally:
        # This block ALWAYS runs, even if there was an error.
        # It ensures we release all hardware resources properly.
        camera.release()           # Stop capturing from camera
        cv2.destroyAllWindows()    # Close the video display window
        itsybitsy.close()          # Close the USB serial connection
        logger.info("Shutdown complete.")


# Entry point
# This ensures main() only runs when you execute this file directly,
# not if another script imports it as a module.
if __name__ == "__main__":
    main()