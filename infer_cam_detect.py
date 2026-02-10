from __future__ import annotations
from pathlib import Path
import cv2
from ultralytics import YOLO

# 后面低置信复核会用到（现在可先不启用）
from secondary_cls import maybe_refine_with_cls


def draw_box(img, xyxy, label: str):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main():
    DETECT_WEIGHTS = r"C:\Users\szc\Desktop\detector\runs\detect\runs\yolo_detect_train\weights\best.pt"
    
    CLS_WEIGHTS = None  # 先不启用：例如 "path/to/cls_best.pt"

    TH_ACCEPT = 0.70    # 置信度阈值：>= 直接过；< 进入复核（未来）

    model_det = YOLO(DETECT_WEIGHTS)
    model_cls = YOLO(CLS_WEIGHTS) if CLS_WEIGHTS else None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # detect 推理（stream=True 更省内存）
        results = model_det.predict(
        frame, conf=0.25, iou=0.45, verbose=False, stream=True
    )

        r = next(results, None)   # 只取这一帧的结果

        if r is not None and r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes
            xyxys = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss  = boxes.cls.cpu().numpy().astype(int)

            for xyxy, conf, c in zip(xyxys, confs, clss):
                name = model_det.names.get(int(c), str(c))
                final_name, final_conf = name, float(conf)

                if model_cls and final_conf < TH_ACCEPT:
                    final_name, final_conf = maybe_refine_with_cls(
                        frame, xyxy, model_cls,
                        det_name=name, det_conf=final_conf
                    )

                draw_box(frame, xyxy, f"{final_name} {final_conf:.2f}")
        else:
            cv2.putText(frame, "No objects", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)


        cv2.imshow("YOLO Detect", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
