import argparse, yaml, matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
from dlrepo.data_processing.loader import get_dataloader

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=16, help="num samples")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dl = get_dataloader(cfg["dataset"], data_dir=cfg["data"]["root"],
                        batch_size=args.n, train=True, num_workers=0, pin_memory=False, config=cfg)
    x, y = next(iter(dl))
    # de-normalize CIFAR-10 for display
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
    std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1)
    x_disp = (x * std + mean).clamp(0,1)

    grid = make_grid(x_disp, nrow=int(args.n**0.5))
    plt.figure()
    plt.axis("off")
    plt.title("Train-time augmentations")
    plt.imshow(grid.permute(1,2,0))
    plt.show()

if __name__ == "__main__":
    main()
