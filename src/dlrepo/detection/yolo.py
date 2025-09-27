# src/dlrepo/detection/yolo.py
from __future__ import annotations
import torch, numpy as np
from typing import List, Tuple, Optional, Dict

class YOLODetector:
    """Thin wrapper around Ultralytics YOLO for consistent outputs.

    Outputs: list of dicts with keys:
      - 'xyxy': np.ndarray [4] (x1,y1,x2,y2)
      - 'conf': float
      - 'cls':  int
      - 'name': str
    """
    def __init__(
        self,
        model_name: str = "yolov8n",
        device: str = "auto",
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
    ):
        if device == "auto":
            device = ("cuda" if torch.cuda.is_available() else
                      ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                              else "cpu"))
        self.device = device

        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError("Ultralytics not installed. Try: pip install ultralytics") from e

        self.model = YOLO(model_name)
        # Send to device if possible (YOLO handles it, but set attribute for clarity)
        self.model.to(self.device)

        self.conf = float(conf)
        self.iou  = float(iou)
        self.max_det = int(max_det)

        # Class names (dict: id->name)
        self.names = self.model.model.names if hasattr(self.model, "model") else getattr(self.model, "names", {})
        if isinstance(self.names, list):
            self.names = {i:n for i,n in enumerate(self.names)}

    @torch.no_grad()
    def infer(self, bgr: np.ndarray) -> List[Dict]:
        """Run detection on a single BGR frame (H,W,3 uint8)."""
        # Ultralytics expects RGB; it converts internally too, but we keep explicit & fast
        import cv2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        res = self.model.predict(
            source=rgb,
            verbose=False,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            device=self.device,
            imgsz=None,   # infer from input shape
        )
        out = []
        if not res:
            return out
        r0 = res[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None:  # no detections
            return out

        # boxes.xyxy (N,4), boxes.conf (N,1), boxes.cls (N,1)
        xyxy = boxes.xyxy.detach().cpu().numpy() if hasattr(boxes, "xyxy") else np.zeros((0,4))
        conf = boxes.conf.detach().cpu().numpy().reshape(-1) if hasattr(boxes, "conf") else np.zeros((0,), dtype=np.float32)
        cls  = boxes.cls.detach().cpu().numpy().astype(int).reshape(-1) if hasattr(boxes, "cls") else np.zeros((0,), dtype=np.int32)

        for i in range(xyxy.shape[0]):
            cid = int(cls[i])
            out.append({
                "xyxy": xyxy[i].astype(np.float32),
                "conf": float(conf[i]),
                "cls":  cid,
                "name": str(self.names.get(cid, str(cid))),
            })
        return out
