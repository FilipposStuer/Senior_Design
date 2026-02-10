from __future__ import annotations
import numpy as np

def safe_crop(img, xyxy, pad=8):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def maybe_refine_with_cls(img, xyxy, model_cls, det_name: str, det_conf: float):
    """
    低置信时，用 cls 模型对框内 crop 复核。
    现在先做最简单版本：直接取 cls top1。
    后面你要加门控（crop脏/重叠大/cls不确定就别改判）也放这里。
    """
    crop = safe_crop(img, xyxy, pad=8)
    if crop is None:
        return det_name, det_conf

    res = model_cls.predict(crop, verbose=False)[0]
    if res.probs is None:
        # 说明你传进来的不是 classify 模型
        return det_name, det_conf

    top1 = int(res.probs.top1)
    conf = float(res.probs.top1conf)
    cls_name = model_cls.names.get(top1, str(top1))

    # 简单策略：cls更自信就用cls，否则保留detect
    if conf > det_conf:
        return cls_name, conf
    return det_name, det_conf
