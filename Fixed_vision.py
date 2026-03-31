"""
Robot Vision - Raspberry Pi Side (Auto-Pick Version)
Detection logic updated to use stream=True predict with secondary classifier support.
"""

from __future__ import annotations
import cv2
import serial
import time
import logging
import sys
import threading
from ultralytics import YOLO
from secondary_cls import maybe_refine_with_cls


# CONFIGURATION

MODEL_PATH           = "best.pt"
CLS_WEIGHTS          = None       # e.g. "path/to/cls_best.pt" — set to enable secondary classifier
TH_ACCEPT            = 0.70       # Confidence threshold: >= passes directly, < goes to secondary
CONFIDENCE_THRESHOLD = 0.25       # Minimum confidence for YOLO to consider a detection at all

CAMERA_INDEX  = 0
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480

SERIAL_PORT     = "/dev/cu.usbmodem1101"
SERIAL_BAUDRATE = 9600
SERIAL_TIMEOUT  = 30

FEEDBACK_INTERVAL     = 0.1
DETECT_CONFIRM_FRAMES = 3
DISAPPEAR_THRESHOLD   = 5

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


# LOGGING SETUP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("robot_vision.log"),
    ]
)
logger = logging.getLogger("RobotVision")


# DETECTION HELPER

def draw_box(img, xyxy, label: str):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def detect_best(frame, model_det, model_cls):
    """
    Run YOLO detection on a single frame.
    Optionally refines low-confidence detections with a secondary classifier.
    Returns the highest-confidence detection dict, or None if nothing found.
    """
    results = model_det.predict(
        frame, conf=CONFIDENCE_THRESHOLD, iou=0.45, verbose=False, stream=True
    )
    r = next(results, None)

    best_detection = None
    best_conf      = 0.0

    if r is None or r.boxes is None or len(r.boxes) == 0:
        return None

    xyxys = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clss  = r.boxes.cls.cpu().numpy().astype(int)

    for xyxy, conf, c in zip(xyxys, confs, clss):
        class_id   = int(c)
        name       = model_det.names.get(class_id, str(class_id))
        final_name = name
        final_conf = float(conf)

        # Optional secondary classifier for low-confidence detections
        if model_cls and final_conf < TH_ACCEPT:
            final_name, final_conf = maybe_refine_with_cls(
                frame, xyxy, model_cls,
                det_name=name, det_conf=final_conf
            )

        # Draw box on frame regardless
        draw_box(frame, xyxy, f"{final_name} {final_conf:.2f}")

        if final_conf > best_conf:
            best_conf = final_conf
            x1, y1, x2, y2 = map(int, xyxy)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            best_detection = {
                "center":     (cx, cy),
                "bbox_size":  (x2 - x1, y2 - y1),
                "class_id":   class_id,
                "class_name": final_name,
                "confidence": final_conf,
            }

    return best_detection


# SERIAL COMMUNICATION CLASS

class ItsyBitsyComm:
    """Handles serial communication with the ItsyBitsy M4."""

    def __init__(self, port: str, baudrate: int, timeout: int):
        self.port      = port
        self.baudrate  = baudrate
        self.timeout   = timeout
        self.conn      = None
        self.connected = False
        self._lock     = threading.Lock()
        self._connect()

    def _connect(self):
        try:
            self.conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1,
                write_timeout=1,
            )
            time.sleep(2)
            self.connected = True
            logger.info("Connected to ItsyBitsy on %s", self.port)
        except serial.SerialException as exc:
            logger.error("Could not connect to ItsyBitsy: %s", exc)
            self.connected = False

    def send_raw(self, command: str) -> bool:
        if not self.connected or not self.conn or not self.conn.is_open:
            logger.error("Serial not available.")
            return False
        if not command.endswith("\n"):
            command += "\n"
        try:
            with self._lock:
                self.conn.write(command.encode())
            logger.debug("Sent: %s", command.strip())
            return True
        except serial.SerialException as exc:
            logger.error("Serial error: %s", exc)
            return False

    def send_pick_command(self, class_name: str, cx: int, cy: int, bbox_w: int, bbox_h: int) -> bool:
        command = f"PICK {class_name.upper()} {cx} {cy} {bbox_w} {bbox_h}"
        logger.info("Sending initial PICK: %s", command)
        return self.send_raw(command)

    def send_position_update(self, cx: int, cy: int) -> bool:
        return self.send_raw(f"POS {cx} {cy}")

    def send_target_reached(self) -> bool:
        logger.info("Object no longer visible — sending TARGET_REACHED")
        return self.send_raw("TARGET_REACHED")

    def wait_for_done(self) -> bool:
        logger.info("Waiting for ItsyBitsy to complete sequence...")
        deadline = time.time() + self.timeout

        while time.time() < deadline:
            try:
                if self.conn and self.conn.in_waiting:
                    response = self.conn.readline().decode(errors="replace").strip().upper()
                    logger.info("ItsyBitsy: %s", response)
                    if response == "DONE":
                        return True
                    elif response == "ERROR":
                        logger.error("ItsyBitsy reported an error.")
                        return False
                time.sleep(0.05)
            except serial.SerialException as exc:
                logger.error("Serial error while waiting: %s", exc)
                return False

        logger.error("Timeout waiting for ItsyBitsy response.")
        return False

    def close(self):
        if self.conn and self.conn.is_open:
            self.conn.close()
            logger.info("Serial connection closed.")


# FEEDBACK LOOP

def run_feedback_loop(
    camera: cv2.VideoCapture,
    model_det: YOLO,
    model_cls,
    itsybitsy: ItsyBitsyComm,
    target_class_id: int,
    stop_event: threading.Event,
    reached_event: threading.Event,
):
    disappear_count = 0
    logger.info("Feedback loop started — tracking class_id %d", target_class_id)

    while not stop_event.is_set():
        ret, frame = camera.read()
        if not ret:
            logger.warning("Feedback loop: failed to read frame.")
            time.sleep(FEEDBACK_INTERVAL)
            continue

        results = model_det.predict(
            frame, conf=CONFIDENCE_THRESHOLD, iou=0.45, verbose=False, stream=True
        )
        r = next(results, None)

        best_detection = None
        best_conf      = 0.0

        if r is not None and r.boxes is not None and len(r.boxes) > 0:
            xyxys = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss  = r.boxes.cls.cpu().numpy().astype(int)

            for xyxy, conf, c in zip(xyxys, confs, clss):
                if int(c) != target_class_id:
                    continue
                final_conf = float(conf)
                final_name = model_det.names.get(int(c), str(c))

                if model_cls and final_conf < TH_ACCEPT:
                    final_name, final_conf = maybe_refine_with_cls(
                        frame, xyxy, model_cls,
                        det_name=final_name, det_conf=final_conf
                    )

                if final_conf > best_conf:
                    best_conf = final_conf
                    x1, y1, x2, y2 = map(int, xyxy)
                    best_detection = {"cx": (x1 + x2) // 2, "cy": (y1 + y2) // 2}

        if best_detection:
            disappear_count = 0
            itsybitsy.send_position_update(best_detection["cx"], best_detection["cy"])
        else:
            disappear_count += 1
            logger.debug("Object not visible, disappear count: %d", disappear_count)
            if disappear_count >= DISAPPEAR_THRESHOLD:
                itsybitsy.send_target_reached()
                reached_event.set()
                break

        time.sleep(FEEDBACK_INTERVAL)

    logger.info("Feedback loop stopped.")


# MAIN FUNCTION

def main():
    logger.info("=" * 60)
    logger.info("ROBOT VISION - RASPBERRY PI (Auto-Pick Version)")
    logger.info("=" * 60)

    logger.info("Loading detection model from '%s'...", MODEL_PATH)
    try:
        model_det = YOLO(MODEL_PATH)
        logger.info("Detection model loaded.")
    except Exception as exc:
        logger.critical("Failed to load detection model: %s", exc)
        return

    model_cls = None
    if CLS_WEIGHTS:
        try:
            model_cls = YOLO(CLS_WEIGHTS)
            logger.info("Secondary classifier loaded from '%s'.", CLS_WEIGHTS)
        except Exception as exc:
            logger.warning("Failed to load secondary classifier: %s", exc)

    logger.info("Opening camera (index %d)...", CAMERA_INDEX)
    camera = cv2.VideoCapture(CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not camera.isOpened():
        logger.critical("Could not open camera.")
        return

    logger.info("Camera ready: %dx%d", FRAME_WIDTH, FRAME_HEIGHT)

    itsybitsy = ItsyBitsyComm(SERIAL_PORT, SERIAL_BAUDRATE, SERIAL_TIMEOUT)
    if not itsybitsy.connected:
        logger.warning("Running without ItsyBitsy connection.")

    logger.info("System ready! Auto-pick enabled. Controls: [Q] quit")
    logger.info("-" * 60)

    last_detection  = None
    frame_count     = 0
    fps_time        = time.time()
    fps             = 0.0
    busy            = False
    confirm_count   = 0

    stop_event      = threading.Event()
    reached_event   = threading.Event()
    feedback_thread = None

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                logger.error("Failed to capture frame.")
                break

            frame_count += 1
            if frame_count % 30 == 0:
                now      = time.time()
                fps      = 30.0 / max(now - fps_time, 1e-6)
                fps_time = now

            # Check if feedback thread sent TARGET_REACHED
            if busy and reached_event.is_set():
                logger.info("TARGET_REACHED — waiting for arm to finish sequence...")
                stop_event.set()
                if feedback_thread:
                    feedback_thread.join()

                success = itsybitsy.wait_for_done()
                if success:
                    logger.info("Pick-and-place completed successfully.")
                else:
                    logger.error("Pick-and-place failed or timed out.")

                busy            = False
                confirm_count   = 0
                stop_event.clear()
                reached_event.clear()
                feedback_thread = None
                last_detection  = None

            # Detection (only when arm is not busy)
            if not busy:
                best_detection = detect_best(frame, model_det, model_cls)

                if best_detection:
                    confirm_count += 1
                    last_detection = best_detection

                    cv2.putText(
                        frame,
                        f"Confirming: {confirm_count}/{DETECT_CONFIRM_FRAMES}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                    )

                    if confirm_count >= DETECT_CONFIRM_FRAMES:
                        cx, cy = last_detection["center"]
                        bw, bh = last_detection["bbox_size"]
                        name   = last_detection["class_name"]
                        cid    = last_detection["class_id"]

                        logger.info("Auto-picking %s at (%d, %d)", name, cx, cy)
                        busy          = True
                        confirm_count = 0

                        if itsybitsy.send_pick_command(name, cx, cy, bw, bh):
                            stop_event.clear()
                            reached_event.clear()
                            feedback_thread = threading.Thread(
                                target=run_feedback_loop,
                                args=(camera, model_det, model_cls, itsybitsy,
                                      cid, stop_event, reached_event),
                                daemon=True,
                            )
                            feedback_thread.start()
                            logger.info("Feedback loop started.")
                        else:
                            logger.error("Failed to send PICK command.")
                            busy = False
                else:
                    confirm_count = 0
                    cv2.putText(frame, "No objects", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # HUD
            status = "BUSY - Arm moving..." if busy else "READY - Watching for objects..."
            cv2.putText(frame, f"Status: {status}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 165, 255) if busy else (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}",
                        (FRAME_WIDTH - 90, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Controls: [Q] quit",
                        (10, FRAME_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if last_detection and not busy:
                cv2.putText(
                    frame,
                    f"Target: {last_detection['class_name']} ({last_detection['confidence']:.2f})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )

            cv2.imshow("Robot Vision - Waste Sorting", frame)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                logger.info("Quit requested.")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    finally:
        if feedback_thread and feedback_thread.is_alive():
            stop_event.set()
            feedback_thread.join()

        camera.release()
        cv2.destroyAllWindows()
        itsybitsy.close()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()