import torch
import copy

class ModelEMA:
    def __init__(self, model, decay=0.9999):
        # Deep-copy the full module, keep same device, eval mode, and freeze params
        self.ema = copy.deepcopy(model).to(next(model.parameters()).device)
        self.ema.eval()
        self.decay = float(decay)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for ema_p, p in zip(self.ema.state_dict().values(), model.state_dict().values()):
            if isinstance(ema_p, torch.Tensor):
                ema_p.copy_(ema_p * d + p.detach() * (1. - d))

    @torch.no_grad()
    def copy_to(self, model):
        model.load_state_dict(self.ema.state_dict(), strict=True)
