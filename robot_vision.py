"""
waste_sorter.py - Conversion corregida para que el brazo baje
"""

from __future__ import annotations
import cv2
import serial
import time
import logging
import sys
import threading
from ultralytics import YOLO

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
MODEL_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.25
CAMERA_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
SERIAL_PORT = "COM5"
SERIAL_BAUDRATE = 9600
SERIAL_TIMEOUT = 30

DETECT_CONFIRM_FRAMES = 3
DISAPPEAR_THRESHOLD = 5
PICK_TIMEOUT = 20
IDLE_HOME_TIMEOUT = 30

# Workspace limits (solo para referencia, la verificación está desactivada)
ENABLE_WORKSPACE_CHECK = False
WS_X_MIN, WS_X_MAX = 8.5, 13.5   # No se usan ahora
WS_Y_MIN, WS_Y_MAX = 0.6, 4.4

# ----------------------------------------------------------------------
# CLASES CORRECTAS (best.pt: 1=cartón,4=papel,5=plástico)
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
    "plastic":   (255, 80, 80),
    "paper":     (200, 200, 255),
    "cardboard": (50, 165, 255),
    "default":   (128, 128, 128),
}

# ----------------------------------------------------------------------
# CONVERSIÓN PÍXEL → CM (corregida para que y_cm no sea tan grande)
# ----------------------------------------------------------------------
def pixel_to_cm(cx, cy):
    # Para x: mantén tu conversión
    OFFSET_X = 6.0
    SCALE_X = 0.020
    x_cm = OFFSET_X + (cx - 320) * SCALE_X
    
    # Fuerza y_cm a un valor bajo (ej. 1.0 cm) para que el brazo baje
    y_cm = 0.8   # ← valor fijo de prueba
    
    return x_cm, y_cm

# ----------------------------------------------------------------------
# COMUNICACIÓN SERIE
# ----------------------------------------------------------------------
class ArmComm:
    def __init__(self, port, baudrate, timeout):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.conn = None
        self.connected = False
        self._connect()

    def _connect(self):
        try:
            self.conn = serial.Serial(self.port, self.baudrate, timeout=1, write_timeout=1)
            time.sleep(2)
            self.connected = True
            logging.info(f"Conectado a {self.port}")
        except Exception as e:
            logging.error(f"Error serie: {e}")

    def send(self, cmd):
        if not self.connected:
            return False
        try:
            self.conn.write((cmd + "\n").encode())
            return True
        except:
            return False

    def pick(self, class_name, x_cm, y_cm, w, h):
        return self.send(f"PICK {class_name.upper()} {x_cm:.2f} {y_cm:.2f} {w} {h}")

    def grip(self):
        return self.send("GRIP")

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
# DETECCIÓN
# ----------------------------------------------------------------------
def detect_best(frame, model):
    results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, iou=0.45, verbose=False, stream=True)
    r = next(results, None)
    if r is None or r.boxes is None:
        return None

    best = None
    best_conf = 0.0
    for xyxy, conf, c in zip(r.boxes.xyxy.cpu().numpy(),
                             r.boxes.conf.cpu().numpy(),
                             r.boxes.cls.cpu().numpy().astype(int)):
        info = WASTE_CLASSES.get(int(c))
        if not info:
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "UNKNOWN", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            continue

        name = info["name"]
        recycle = info["recycle"]
        conf_val = float(conf)

        color = CLASS_COLORS.get(name, CLASS_COLORS["default"])
        if not recycle:
            color = (128, 128, 128)
        x1, y1, x2, y2 = map(int, xyxy)
        label = f"{name} {conf_val:.2f}" if recycle else f"REJECT: {name}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if recycle and conf_val > best_conf:
            best_conf = conf_val
            best = {
                "class_id": int(c),
                "class_name": name,
                "center": ((x1+x2)//2, (y1+y2)//2),
                "bbox": (x2-x1, y2-y1),
                "confidence": conf_val,
            }
    return best

# ----------------------------------------------------------------------
# BUCLE DE FEEDBACK (espera a que el objeto desaparezca)
# ----------------------------------------------------------------------
def feedback_loop(camera, model, arm, target_class, stop_event, done_event):
    disappear = 0
    while not stop_event.is_set():
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.1)
            continue
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False, stream=True)
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
                logging.info("Objeto desaparecido, enviando GRIP")
                arm.grip()
                done_event.set()
                break
        time.sleep(0.1)

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("WasteSorter")

    # Cargar modelo
    try:
        model = YOLO(MODEL_PATH)
        logger.info(f"Modelo cargado. Clases: {model.names}")
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        return

    # Cámara
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        logger.error("No se pudo abrir la cámara")
        return

    # Brazo
    arm = ArmComm(SERIAL_PORT, SERIAL_BAUDRATE, SERIAL_TIMEOUT)
    if not arm.connected:
        logger.warning("Brazo no conectado – solo se simularán comandos")

    # Estado
    busy = False
    busy_start = 0
    confirm = 0
    last_detection = None
    last_detect_time = time.time()
    stop_fb = threading.Event()
    done_fb = threading.Event()
    fb_thread = None

    logger.info("Sistema LISTO. Presiona Q para salir.")
    logger.info("Verificación de workspace DESACTIVADA. Conversión de coordenadas ajustada.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Timeout si el brazo se queda atascado
        if busy and (time.time() - busy_start) > PICK_TIMEOUT:
            logger.error("Tiempo de espera agotado – reiniciando brazo")
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

        # Cuando el feedback loop termina (objeto desapareció)
        if busy and done_fb.is_set():
            logger.info("Feedback loop finalizado, esperando que termine GRIP...")
            stop_fb.set()
            if fb_thread:
                fb_thread.join(timeout=2)
            arm.wait_done(10)
            arm.home()
            arm.wait_done(10)
            busy = False
            confirm = 0
            stop_fb.clear()
            done_fb.clear()
            fb_thread = None

        # Detección (solo si el brazo está libre)
        if not busy:
            det = detect_best(frame, model)
            if det:
                confirm += 1
                last_detection = det
                last_detect_time = time.time()
                cv2.putText(frame, f"Confirmando {confirm}/{DETECT_CONFIRM_FRAMES}", (10,90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                if confirm >= DETECT_CONFIRM_FRAMES:
                    cx, cy = det["center"]
                    w, h = det["bbox"]
                    x_cm, y_cm = pixel_to_cm(cx, cy)

                    # Siempre intenta recoger (workspace check desactivado)
                    logger.info(f"Recogiendo {det['class_name']} en ({x_cm:.1f}, {y_cm:.1f}) cm")
                    if arm.pick(det["class_name"], x_cm, y_cm, w, h):
                        if arm.wait_done(15):
                            busy = True
                            busy_start = time.time()
                            confirm = 0
                            stop_fb.clear()
                            done_fb.clear()
                            fb_thread = threading.Thread(target=feedback_loop,
                                args=(cap, model, arm, det["class_id"], stop_fb, done_fb),
                                daemon=True)
                            fb_thread.start()
                        else:
                            logger.error("El brazo no alcanzó la posición de recogida")
                            arm.home()
                            arm.wait_done(10)
                    else:
                        logger.error("Fallo al enviar comando PICK")
            else:
                confirm = 0

        # Auto-home por inactividad
        if not busy and (time.time() - last_detect_time) > IDLE_HOME_TIMEOUT:
            logger.info("Inactividad prolongada, enviando HOME")
            arm.home()
            arm.wait_done(10)
            last_detect_time = time.time()

        # HUD
        status = "OCUPADO" if busy else "LISTO"
        color = (0,165,255) if busy else (0,255,0)
        cv2.putText(frame, f"Estado: {status}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if last_detection and not busy:
            cv2.putText(frame, f"Objetivo: {last_detection['class_name']} {last_detection['confidence']:.2f}",
                        (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow("Waste Sorter", frame)

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
                arm.send("RELEASE")

    cap.release()
    cv2.destroyAllWindows()
    arm.close()

if __name__ == "__main__":
    main()