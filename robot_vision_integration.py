"""
Robot Vision - Raspberry Pi Side
"""

import cv2
import serial
import time
import logging
import sys
from ultralytics import YOLO

# CONFIGURATION

MODEL_PATH          = "best.pt"
CONFIDENCE_THRESHOLD = 0.5

CAMERA_INDEX  = 0
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480

SERIAL_PORT     = "/dev/ttyACM0"
SERIAL_BAUDRATE = 9600
SERIAL_TIMEOUT  = 30       # seconds to wait for ItsyBitsy to finish the sequence

WASTE_CLASSES = {
    0: {"name": "general", "bin": "landfill"},
    1: {"name": "paper",   "bin": "paper"},
    2: {"name": "plastic", "bin": "plastic"},
}

CLASS_COLORS = {
    "general": (128, 128, 128),
    "paper":   (255, 255, 255),
    "plastic": (255,   0,   0),
}

# LOGGING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("robot_vision.log"),
    ]
)
logger = logging.getLogger("RobotVision")

# SERIAL COMMUNICATION

class ItsyBitsyComm:
    
    """Handles serial communication with the ItsyBitsy M4."""

    def __init__(self, port: str, baudrate: int, timeout: int):
        self.port     = port
        self.baudrate = baudrate
        self.timeout  = timeout
        self.conn: serial.Serial | None = None
        self.connected = False
        self._connect()

    def _connect(self):
        try:
            self.conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1,
                write_timeout=1,
            )
            time.sleep(2)  # Wait for ItsyBitsy to boot
            self.connected = True
            logger.info("Connected to ItsyBitsy on %s", self.port)
        except serial.SerialException as exc:
            logger.error("Could not connect to ItsyBitsy: %s", exc)
            self.connected = False

    def send_pick_command(
        self,
        class_name: str,
        cx: int,
        cy: int,
        bbox_w: int,
        bbox_h: int,
    ) -> bool:
        
        """
        Send a PICK command and wait for the ItsyBitsy to finish.
        """
        if not self.connected or not self.conn or not self.conn.is_open:
            logger.error("Serial not available.")
            return False

        command = f"PICK {class_name.upper()} {cx} {cy} {bbox_w} {bbox_h}\n"

        try:
            self.conn.write(command.encode())
            logger.info("Sent: %s", command.strip())

            # Wait for ItsyBitsy to complete the full pick-and-place
            deadline = time.time() + self.timeout
            while time.time() < deadline:
                if self.conn.in_waiting:
                    response = self.conn.readline().decode(errors="replace").strip().upper()
                    logger.info("ItsyBitsy: %s", response)
                    if response == "DONE":
                        return True
                    elif response == "ERROR":
                        logger.error("ItsyBitsy reported an error.")
                        return False
                time.sleep(0.05)

            logger.error("Timeout waiting for ItsyBitsy response.")
            return False

        except serial.SerialException as exc:
            logger.error("Serial error: %s", exc)
            return False

    def close(self):
        if self.conn and self.conn.is_open:
            self.conn.close()
            logger.info("Serial connection closed.")


# MAIN LOOP

def main():
    logger.info("=" * 60)
    logger.info("ROBOT VISION - RASPBERRY PI")
    logger.info("Vision + Detection only. Arm control → ItsyBitsy M4")
    logger.info("=" * 60)

    # -- Load YOLO model --
    logger.info("Loading model from '%s'...", MODEL_PATH)
    try:
        model = YOLO(MODEL_PATH)
        logger.info("Model loaded. Classes: %s", getattr(model, "names", "unknown"))
    except Exception as exc:
        logger.critical("Failed to load model: %s", exc)
        return

    # -- Open camera --
    logger.info("Opening camera (index %d)...", CAMERA_INDEX)
    camera = cv2.VideoCapture(CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not camera.isOpened():
        logger.critical("Could not open camera. Check connection or index.")
        return
    logger.info("Camera ready: %dx%d", FRAME_WIDTH, FRAME_HEIGHT)

    # -- Connect to ItsyBitsy --
    itsybitsy = ItsyBitsyComm(SERIAL_PORT, SERIAL_BAUDRATE, SERIAL_TIMEOUT)
    if not itsybitsy.connected:
        logger.warning("Running without ItsyBitsy connection.")

    logger.info("System ready!  Controls: [P] pick  [Q] quit")
    logger.info("-" * 60)

    last_detection = None
    frame_count    = 0
    fps_time       = time.time()
    fps            = 0.0
    busy           = False   # True while ItsyBitsy is executing a sequence

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                logger.error("Failed to capture frame.")
                break

            frame_count += 1
            if frame_count % 30 == 0:
                now  = time.time()
                fps  = 30.0 / max(now - fps_time, 1e-6)
                fps_time = now

            # -- Run detection (skip if arm is busy) --
            if not busy:
                results = model(frame, verbose=False)[0]
                best_detection = None
                best_conf      = 0.0

                for det in results.boxes.data:
                    x1, y1, x2, y2, conf, class_id = det
                    conf     = float(conf)
                    class_id = int(class_id)

                    if conf < CONFIDENCE_THRESHOLD:
                        continue
                    if class_id not in WASTE_CLASSES:
                        continue

                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    class_info = WASTE_CLASSES[class_id]
                    class_name = class_info["name"]
                    color      = CLASS_COLORS.get(class_name, (0, 255, 0))
                    cx         = (x1 + x2) // 2
                    cy         = (y1 + y2) // 2

                    # Draw
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, f"Bin: {class_info['bin']}", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                    if conf > best_conf:
                        best_conf = conf
                        best_detection = {
                            "center":     (cx, cy),
                            "bbox_size":  (x2 - x1, y2 - y1),
                            "class_id":   class_id,
                            "class_name": class_name,
                            "confidence": conf,
                        }

                last_detection = best_detection

            # -- HUD --
            status = "BUSY - Arm moving..." if busy else "READY"
            cv2.putText(frame, f"Status: {status}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 165, 255) if busy else (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}",
                        (FRAME_WIDTH - 90, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Controls: [P] pick  [Q] quit",
                        (10, FRAME_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if last_detection and not busy:
                cv2.putText(
                    frame,
                    f"Target: {last_detection['class_name']} ({last_detection['confidence']:.2f})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )

            cv2.imshow("Robot Vision - Waste Sorting", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                logger.info("Quit requested.")
                break

            elif key == ord("p") and last_detection and not busy:
                cx, cy   = last_detection["center"]
                bw, bh   = last_detection["bbox_size"]
                name     = last_detection["class_name"]

                logger.info("Sending PICK command → ItsyBitsy: %s at (%d, %d)", name, cx, cy)
                busy = True

                success = itsybitsy.send_pick_command(name, cx, cy, bw, bh)

                if success:
                    logger.info("ItsyBitsy completed pick-and-place successfully.")
                else:
                    logger.error("ItsyBitsy pick-and-place failed or timed out.")

                last_detection = None
                busy = False

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        itsybitsy.close()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()