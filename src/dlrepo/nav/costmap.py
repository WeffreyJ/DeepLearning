import numpy as np
class DepthCostmap:
    def __init__(self, near_th=0.3): self.near_th = near_th
    def step(self, depth_norm: np.ndarray):
        mask = (depth_norm < self.near_th).astype(np.uint8)
        h, w = mask.shape
        left = mask[:, : w//2].mean(); right = mask[:, w//2 :].mean()
        steer = float(np.clip(right - left, -1.0, 1.0))
        return {"mask": mask, "steer": steer}
