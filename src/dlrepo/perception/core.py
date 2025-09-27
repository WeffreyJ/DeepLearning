from typing import Dict, Any
import numpy as np, torch

class PerceptionGraph:
    def __init__(self, *, depth=None, detector=None, tracker=None, navigator=None, device: str = "auto"):
        self.depth = depth; self.detector = detector; self.tracker = tracker; self.navigator = navigator
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)

    @torch.no_grad()
    def process_frame(self, rgb: np.ndarray, t: float) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.depth is not None: out["depth"] = self.depth.infer(rgb)
        if self.detector is not None:
            det = self.detector.detect(rgb); out["detections"] = det
            if self.tracker is not None: out["tracks"] = self.tracker.update(det, t)
        if self.navigator is not None and "depth" in out: out["nav"] = self.navigator.step(out["depth"])
        return out
