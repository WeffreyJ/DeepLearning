import torch
def mixup_data(x, y, alpha: float):
    if alpha <= 0.0:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[idx], y, y[idx], lam
def mixup_criterion(criterion, pred, y_a, y_b, lam: float):
    if lam == 1.0: return criterion(pred, y_a)
    return lam*criterion(pred, y_a) + (1-lam)*criterion(pred, y_b)
