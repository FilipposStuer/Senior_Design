"""
Robot Vision - Raspberry Pi Side

PURPOSE:
    This script runs on the Raspberry Pi and handles THREE things:
        1. Capture video from the camera and detect waste objects using AI (YOLO)
        2. Tell the ItsyBitsy M4 microcontroller WHAT to pick and WHERE it is
        3. Continuously stream the object's position back to the ItsyBitsy
           so it can run PID corrections in real time

    The ItsyBitsy M4 is responsible for actually moving the robot arm.
    This separation keeps each device focused on what it does best:
        - Raspberry Pi  → AI / Vision (needs a powerful processor)
        - ItsyBitsy M4  → Motor control + PID (needs precise real-time timing)

HOW IT COMMUNICATES:
    Step 1 - Pi detects object, sends initial command:
             "PICK PLASTIC 320 240 80 60\n"
              ^^^^  ^^^^^^^  ^^^  ^^^  ^^  ^^
              cmd   class    cx   cy   bw  bh

    Step 2 - While arm is moving, Pi keeps sending position updates (~10x/sec):
             "POS 318 242\n"   (object still visible, arm not there yet)
             "POS 310 245\n"   (ItsyBitsy uses this for PID corrections)

    Step 3 - Object disappears from frame (arm reached it):
             "TARGET_REACHED\n"

    Step 4 - ItsyBitsy closes gripper, completes sequence, replies:
             "DONE"  → success
             "ERROR" → something went wrong on the arm side

WHY THIS IS CLOSED-LOOP:
    The previous version sent coordinates ONCE and went blind.
    Now the camera acts as a continuous sensor — constantly telling
    the ItsyBitsy where the object is while the arm moves toward it.
    The ItsyBitsy feeds this into its PID controller to correct errors in real time.
"""

import cv2        # OpenCV: used to capture camera frames and draw boxes on screen
import serial     # PySerial: used to send/receive data over USB to the ItsyBitsy
import time       # Used for delays and timeouts
import logging    # Better than print() — gives timestamps and log levels
import sys        # Used to direct log output to the terminal
import threading  # Runs the feedback loop in background while main loop continues
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

# How often the Pi sends position updates to ItsyBitsy while arm is moving.
# 0.1 seconds = ~10 times per second. Fast enough for smooth PID corrections
# without overwhelming the serial port or the Pi's CPU.
FEEDBACK_INTERVAL = 0.1

# If the object disappears from the camera frame for this many consecutive
# frames, we assume the arm has reached it and send TARGET_REACHED.
# At ~10 feedback updates/sec, 5 frames = 0.5 seconds of object being gone.
DISAPPEAR_THRESHOLD = 5

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
# Now includes methods for continuous position
# streaming and TARGET_REACHED signaling.

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

        # This lock prevents two threads from writing to serial at the same time.
        # Without it, the feedback thread and main thread could send commands
        # simultaneously and corrupt the data being sent.
        self._lock = threading.Lock()

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

    def send_raw(self, command: str) -> bool:
        """
        Send a raw command string over serial.
        Uses a thread lock so simultaneous sends from different threads
        don't corrupt each other. This is used by all other send methods.
        """
        if not self.connected or not self.conn or not self.conn.is_open:
            logger.error("Serial not available.")
            return False

        if not command.endswith("\n"):
            command += "\n"

        try:
            with self._lock:  # Only one thread can write at a time
                self.conn.write(command.encode())
            logger.debug("Sent: %s", command.strip())
            return True
        except serial.SerialException as exc:
            logger.error("Serial error: %s", exc)
            return False

    def send_pick_command(
        self,
        class_name: str,  # e.g. "plastic"
        cx: int,          # X pixel coordinate of object center in the camera frame
        cy: int,          # Y pixel coordinate of object center in the camera frame
        bbox_w: int,      # Width of the detection bounding box (pixels) — used to estimate depth
        bbox_h: int,      # Height of the detection bounding box (pixels) — used to estimate depth
    ) -> bool:
        """
        Send the initial PICK command to start the pick sequence.

        The command format sent over serial is:
            "PICK PLASTIC 320 240 80 60\n"

        The ItsyBitsy firmware uses this to:
          - Know which bin to go to
          - Get the first position estimate to start moving toward the object
          - Get bbox size to estimate how far away the object is (depth)

        After this, the feedback loop takes over and keeps sending POS updates.
        """
        command = f"PICK {class_name.upper()} {cx} {cy} {bbox_w} {bbox_h}"
        logger.info("Sending initial PICK: %s", command)
        return self.send_raw(command)

    def send_position_update(self, cx: int, cy: int) -> bool:
        """
        Send a position update while the arm is moving toward the object.
        This is the feedback that makes it a closed-loop control system.

        Format: "POS 318 242\n"
                 ^^^  ^^^  ^^^
                 cmd  cx   cy

        The ItsyBitsy receives this ~10 times per second and feeds it into
        its PID controller to calculate corrections and keep the arm on target.
        """
        return self.send_raw(f"POS {cx} {cy}")

    def send_target_reached(self) -> bool:
        """
        Tell the ItsyBitsy the arm has reached the object.

        This is triggered when the object disappears from the camera frame,
        meaning the arm or gripper is now covering it.

        The ItsyBitsy responds by closing the gripper and completing
        the rest of the sequence: lift, move to bin, release, return home.
        """
        logger.info("Object no longer visible — sending TARGET_REACHED")
        return self.send_raw("TARGET_REACHED")

    def wait_for_done(self) -> bool:
        """
        Block and wait for the ItsyBitsy to send "DONE" or "ERROR".
        Called after TARGET_REACHED while the arm finishes the sequence.

        Returns True if DONE received, False if ERROR or timeout.
        """
        logger.info("Waiting for ItsyBitsy to complete sequence...")
        deadline = time.time() + self.timeout

        while time.time() < deadline:
            try:
                if self.conn and self.conn.in_waiting:  # Check if bytes arrived from ItsyBitsy
                    response = self.conn.readline().decode(errors="replace").strip().upper()
                    logger.info("ItsyBitsy: %s", response)

                    if response == "DONE":
                        return True    # Success! Arm finished the full sequence
                    elif response == "ERROR":
                        logger.error("ItsyBitsy reported an error.")
                        return False   # Arm encountered a problem

                    # Any other message (e.g. debug prints from ItsyBitsy) — log and keep waiting

                time.sleep(0.05)  # Small pause to avoid hammering the CPU in this loop

            except serial.SerialException as exc:
                logger.error("Serial error while waiting: %s", exc)
                return False

        # If we reach here, ItsyBitsy never responded in time
        logger.error("Timeout waiting for ItsyBitsy response.")
        return False

    def close(self):
        """Close the serial port cleanly when the program exits."""
        if self.conn and self.conn.is_open:
            self.conn.close()
            logger.info("Serial connection closed.")


# FEEDBACK LOOP FUNCTION
# This runs in a background thread while the arm is moving.
# It continuously detects the target object and streams
# its position to the ItsyBitsy for PID correction.
# When the object disappears from the frame, it sends
# TARGET_REACHED to tell the ItsyBitsy to close the gripper.

def run_feedback_loop(
    camera: cv2.VideoCapture,
    model: YOLO,
    itsybitsy: ItsyBitsyComm,
    target_class_id: int,           # The class ID of the object we are tracking
    stop_event: threading.Event,    # Set from outside to stop this loop
    reached_event: threading.Event, # This function sets it when TARGET_REACHED is sent
):
    """
    Continuously track the target object and stream its position to ItsyBitsy.

    Runs in a background thread so the main loop can keep displaying
    the video feed while the arm is moving toward the object.
    """
    # Count how many consecutive frames the object has been missing.
    # When this reaches DISAPPEAR_THRESHOLD, we send TARGET_REACHED.
    disappear_count = 0

    logger.info("Feedback loop started — tracking class_id %d", target_class_id)

    while not stop_event.is_set():

        ret, frame = camera.read()
        if not ret:
            logger.warning("Feedback loop: failed to read frame.")
            time.sleep(FEEDBACK_INTERVAL)
            continue

        # Run YOLO detection on this frame — only looking for our target class
        results = model(frame, verbose=False)[0]
        best_detection = None
        best_conf = 0.0

        for det in results.boxes.data:
            x1, y1, x2, y2, conf, class_id = det
            conf     = float(conf)
            class_id = int(class_id)

            # Only track the same class as the object we are picking
            if class_id != target_class_id:
                continue
            if conf < CONFIDENCE_THRESHOLD:
                continue

            if conf > best_conf:
                best_conf = conf
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                best_detection = {
                    "cx": (x1 + x2) // 2,
                    "cy": (y1 + y2) // 2,
                }

        if best_detection:
            # Object is still visible — reset counter and send position update
            disappear_count = 0
            itsybitsy.send_position_update(
                best_detection["cx"],
                best_detection["cy"],
            )
        else:
            # Object not detected in this frame — increment disappear counter
            disappear_count += 1
            logger.debug("Object not visible, disappear count: %d", disappear_count)

            if disappear_count >= DISAPPEAR_THRESHOLD:
                # Object has been gone long enough — arm must have reached it
                itsybitsy.send_target_reached()
                reached_event.set()  # Signal the main thread that TARGET_REACHED was sent
                break                # Stop the feedback loop

        # Wait before next update (~10 position updates per second)
        time.sleep(FEEDBACK_INTERVAL)

    logger.info("Feedback loop stopped.")


# MAIN FUNCTION
# This is the entry point of the program.
# It sets everything up, then runs the main
# camera + detection loop until the user quits.

def main():
    logger.info("=" * 60)
    logger.info("ROBOT VISION - RASPBERRY PI (Closed-Loop Version)")
    logger.info("Vision + Detection + Position Feedback → ItsyBitsy M4 PID")
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
                             # We skip detection while busy to avoid commanding mid-sequence again

    # Threading objects for the feedback loop.
    # stop_event  : main loop sets this to stop the feedback thread
    # reached_event: feedback thread sets this when TARGET_REACHED is sent
    stop_event      = threading.Event()
    reached_event   = threading.Event()
    feedback_thread = None

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

            # Check if feedback thread has signaled TARGET_REACHED.
            # If it did, stop the feedback loop and wait for ItsyBitsy to finish.
            if busy and reached_event.is_set():
                logger.info("TARGET_REACHED received — waiting for arm to finish sequence...")

                # Stop the feedback thread cleanly before waiting for DONE
                stop_event.set()
                if feedback_thread:
                    feedback_thread.join()

                # Wait for ItsyBitsy to finish the rest of the sequence
                # (lift, move to bin, release, return home)
                success = itsybitsy.wait_for_done()

                if success:
                    logger.info("ItsyBitsy completed pick-and-place successfully.")
                else:
                    logger.error("ItsyBitsy pick-and-place failed or timed out.")

                # Reset everything so we are ready for the next object
                busy            = False
                stop_event.clear()
                reached_event.clear()
                feedback_thread = None
                last_detection  = None

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
            status = "BUSY - Streaming position to arm..." if busy else "READY"
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
                cid    = last_detection["class_id"]

                logger.info("Sending PICK command → ItsyBitsy: %s at (%d, %d)", name, cx, cy)
                busy = True  # Pause detection while arm is moving

                # Send the initial PICK command to start the sequence
                if itsybitsy.send_pick_command(name, cx, cy, bw, bh):

                    # Start the feedback thread in the background.
                    # It will keep sending POS updates to ItsyBitsy
                    # until the object disappears from the frame.
                    stop_event.clear()
                    reached_event.clear()
                    feedback_thread = threading.Thread(
                        target=run_feedback_loop,
                        args=(camera, model, itsybitsy, cid, stop_event, reached_event),
                        daemon=True,  # Thread stops automatically if main program exits
                    )
                    feedback_thread.start()
                    logger.info("Feedback loop started — streaming position to ItsyBitsy.")
                else:
                    logger.error("Failed to send PICK command.")
                    busy = False  # Reset busy if initial send failed

    except KeyboardInterrupt:
        # Ctrl+C pressed in terminal — exit gracefully
        logger.info("Interrupted by user.")

    finally:
        # This block ALWAYS runs, even if there was an error.
        # It ensures we release all hardware resources properly.

        # Stop the feedback thread if it is still running
        if feedback_thread and feedback_thread.is_alive():
            stop_event.set()
            feedback_thread.join()

        camera.release()           # Stop capturing from camera
        cv2.destroyAllWindows()    # Close the video display window
        itsybitsy.close()          # Close the USB serial connection
        logger.info("Shutdown complete.")


# Entry point
# This ensures main() only runs when you execute this file directly,
# not if another script imports it as a module.
if __name__ == "__main__":
    main()
