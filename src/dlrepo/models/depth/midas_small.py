import torch, numpy as np, cv2
class MiDaSSmall:
    def __init__(self, device="cpu"):
        if device=="auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)
        self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Small").to(self.device).eval()
        self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
    @torch.no_grad()
    def infer(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = self.transforms(rgb).to(self.device)
        pred = self.model(t.unsqueeze(0))
        depth = torch.nn.functional.interpolate(pred.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False).squeeze().cpu().numpy()
        d = (depth - depth.min()) / (depth.ptp() + 1e-8)
        return d
