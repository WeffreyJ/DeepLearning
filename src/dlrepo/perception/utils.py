# src/dlrepo/perception/utils.py
from __future__ import annotations
import numpy as np
import cv2


def colorize_depth(d: np.ndarray) -> np.ndarray:
    """
    Convert a depth map (float32, arbitrary range) to a color image (uint8, BGR).
    Robust to NumPy 2.0 (uses np.ptp) and degenerate ranges.
    """
    if d is None or np.size(d) == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    d = np.asarray(d, dtype=np.float32)
    dmin = float(np.min(d))
    drng = float(np.ptp(d))  # NumPy 2.0 compatible

    if not np.isfinite(drng) or drng <= 1e-12:
        dnorm = np.zeros_like(d, dtype=np.float32)
    else:
        dnorm = (d - dmin) / (drng + 1e-8)

    d8 = np.clip(dnorm * 255.0, 0, 255).astype(np.uint8)
    cm = cv2.applyColorMap(d8, cv2.COLORMAP_MAGMA)  # returns BGR
    return cm

def draw_boxes(img: np.ndarray, dets):
    out = img.copy()
    for x1,y1,x2,y2,score,cls in dets or []:
        cv2.rectangle(out, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        cv2.putText(out, f"{int(cls)}:{score:.2f}", (int(x1), int(y1)-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    return out
