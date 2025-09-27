import numpy as np, cv2

def colorize_depth(d: np.ndarray) -> np.ndarray:
    d = (d - d.min()) / (d.ptp() + 1e-8)
    cm = cv2.applyColorMap((d*255).astype("uint8"), cv2.COLORMAP_MAGMA)
    return cm  # BGR

def draw_boxes(img: np.ndarray, dets):
    out = img.copy()
    for x1,y1,x2,y2,score,cls in dets or []:
        cv2.rectangle(out, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        cv2.putText(out, f"{int(cls)}:{score:.2f}", (int(x1), int(y1)-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    return out
