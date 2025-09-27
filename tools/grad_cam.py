import argparse, yaml, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from dlrepo.models.cnn import resnet18

# Simple Grad-CAM for the last conv block
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target = target_layer
        self.activations = None
        self.gradients = None
        self.h1 = target_layer.register_forward_hook(self._forward_hook)
        self.h2 = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, m, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, m, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, scores, class_idx=None):
        # scores: (B,C)
        if class_idx is None:
            class_idx = scores.argmax(dim=1)
        self.model.zero_grad()
        one_hot = torch.zeros_like(scores)
        one_hot[torch.arange(scores.size(0)), class_idx] = 1.0
        (scores * one_hot).sum().backward(retain_graph=True)

        # weights: GAP over gradients
        weights = self.gradients.mean(dim=(2,3), keepdim=True)  # (B, K,1,1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (B,1,H,W)
        cam = F.relu(cam)
        # normalize to [0,1]
        cam = (cam - cam.amin(dim=(2,3), keepdim=True)) / (cam.amax(dim=(2,3), keepdim=True) + 1e-8)
        return cam

def denorm(x):
    mean = torch.tensor([0.4914,0.4822,0.4465], device=x.device).view(1,3,1,1)
    std  = torch.tensor([0.2023,0.1994,0.2010], device=x.device).view(1,3,1,1)
    return (x*std+mean).clamp(0,1)

def overlay(img, heatmap, alpha=0.45):
    import numpy as np
    import matplotlib.cm as cm
    hm = heatmap.squeeze().cpu().numpy()
    hm_color = cm.jet(hm)[...,:3]  # (H,W,3)
    base = img.permute(1,2,0).cpu().numpy()
    return (alpha*hm_color + (1-alpha)*base).clip(0,1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--count", type=int, default=6)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    data_dir = cfg.get("data",{}).get("root","data/raw")
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])
    test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tfm)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.count, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = resnet18(num_classes=10, pretrained=False).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    cam_layer = model.layer4[-1].conv2  # last conv
    cam = GradCAM(model, cam_layer)

    x, y = next(iter(test_loader))
    
    x = x.to(device)
    logits = model(x)          # <-- no torch.no_grad here
    pred = logits.argmax(1)
    heat = cam(logits)


    x_disp = denorm(x)
    cols = min(args.count, 6)
    rows = (args.count + cols - 1)//cols
    plt.figure(figsize=(3*cols, 3*rows))
    for i in range(args.count):
        plt.subplot(rows, cols, i+1)
        plt.axis("off")
        ov = overlay(x_disp[i], F.interpolate(heat[i:i+1], size=x_disp[i].shape[1:], mode="bilinear", align_corners=False)[0,0])
        title = f"pred:{test_ds.classes[pred[i]]}\ntrue:{test_ds.classes[y[i]]}"
        plt.title(title, fontsize=9)
        plt.imshow(ov)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
