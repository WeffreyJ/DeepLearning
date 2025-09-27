# src/dlrepo/models/depth/midas_small.py
import cv2
import numpy as np
import torch

class MiDaSSmall:
    """
    Robust MiDaS small wrapper:
    - Tries several TorchHub entries: MiDaS_small, midas_v21_small, DPT_Small
    - Picks a suitable transform: small_transform, dpt_transform, default_transform
    - Returns a depth map normalized to [0,1], resized to the input frame
    """
    def __init__(self, device="cpu", repo="intel-isl/MiDaS"):
        if device == "auto":
            device = (
                "cuda" if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        self.device = torch.device(device)

        # Try loading a "small" MiDaS model from hub with a few fallback names
        entries = ["MiDaS_small", "midas_v21_small", "DPT_Small"]
        last_err = None
        self.model = None
        for name in entries:
            try:
                self.model = torch.hub.load(repo, name, trust_repo=True).to(self.device).eval()
                self._entry = name
                break
            except Exception as e:
                last_err = e
        if self.model is None:
            raise RuntimeError(f"Could not load MiDaS small model from {repo}. Last error: {last_err}")

        # Pick a reasonable transform from the hub transforms module
        tmod = torch.hub.load(repo, "transforms", trust_repo=True)
        self.transforms = None
        for tname in ("small_transform", "dpt_transform", "default_transform"):
            if hasattr(tmod, tname):
                self.transforms = getattr(tmod, tname)
                break

    @torch.no_grad()
    def infer(self, bgr: np.ndarray) -> np.ndarray:
        """Run depth inference on a single BGR (OpenCV) image; return HxW float32 in [0,1]."""
        # BGR -> RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            t = self.transforms(rgb)  # MiDaS transforms typically return (1,3,H,W) or a dict
            if isinstance(t, dict) and "image" in t:   # some versions return {"image": tensor, ...}
                t = t["image"]
            t = t.to(self.device)
        else:
            # Minimal fallback if transforms are missing
            import torchvision.transforms as T
            t = T.ToTensor()(rgb).unsqueeze(0).to(self.device)  # (1,3,H,W)

        # Ensure 4D tensor (B,C,H,W)
        if t.dim() == 3:
            t = t.unsqueeze(0)
        elif t.dim() != 4:
            raise RuntimeError(f"Unexpected input dim from transform: {t.shape}")

        # Forward
        pred = self.model(t)  # [1,H',W'] or [1,1,H',W']
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)  # -> [1,1,H',W']

        # Resize to original size and squeeze to HxW
        H, W = rgb.shape[:2]
        depth = torch.nn.functional.interpolate(
            pred, size=(H, W), mode="bicubic", align_corners=False
        ).squeeze().float().cpu().numpy()

        # Normalize to [0,1] for visualization
        dmin, dptp = float(np.min(depth)), float(np.ptp(depth))
        if dptp <= 1e-8:
            return np.zeros_like(depth, dtype=np.float32)
        return ((depth - dmin) / dptp).astype(np.float32)
