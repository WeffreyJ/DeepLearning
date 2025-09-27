# tools/tsne_embeddings.py
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from dlrepo.models.cnn import resnet18


def penultimate_features(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Grab features just before the final FC. For ResNet18 this is after avgpool.
    Returns (B, D) tensor.
    """
    feats = None

    def hook(_m, _i, o):
        nonlocal feats
        feats = o.detach()

    h = model.avgpool.register_forward_hook(hook)
    with torch.no_grad():
        _ = model(x)
    h.remove()
    return feats.view(feats.size(0), -1)


def build_test_loader(data_dir: str, batch_size: int = 256, num_workers: int = 2):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tfm)
    return ds, torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers), ds.classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=2000, help="number of test samples to embed")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Load config
    cfg = yaml.safe_load(open(args.config, "r"))
    data_dir = cfg.get("data", {}).get("root", "data/raw")

    # Device + model
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    model = resnet18(num_classes=10, pretrained=False).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # Data
    ds, full_loader, class_names = build_test_loader(data_dir, batch_size=args.batch_size)
    n = min(args.n, len(ds))

    # Collect features
    X_list, Y_list = [], []
    seen = 0
    for x, y in full_loader:
        if seen >= n:
            break
        take = min(x.size(0), n - seen)
        x = x[:take].to(device)
        y = y[:take]
        feats = penultimate_features(model, x).cpu().numpy()
        X_list.append(feats)
        Y_list.append(y.numpy())
        seen += take

    X = np.concatenate(X_list, axis=0).astype("float64", copy=False)  # TSNE prefers float64
    Y = np.concatenate(Y_list, axis=0)

    # Guard against any non-finite rows
    finite_mask = np.isfinite(X).all(axis=1)
    X = X[finite_mask]
    Y = Y[finite_mask]

    # PCA (50D) with whitening stabilizes TSNE and speeds it up
    pca = PCA(n_components=min(50, X.shape[1]), whiten=True, random_state=args.seed)
    X50 = pca.fit_transform(X)

    # TSNE
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate=200.0,
        perplexity=args.perplexity,
        n_iter=1000,
        random_state=args.seed,
        verbose=1,
        n_jobs=None  # sklearn will choose sensible default
    )
    Z = tsne.fit_transform(X50)

    # Plot
    plt.figure(figsize=(7.5, 6.5))
    for cls_idx, cls_name in enumerate(class_names):
        mask = (Y == cls_idx)
        if np.any(mask):
            plt.scatter(Z[mask, 0], Z[mask, 1], s=8, alpha=0.7, label=cls_name)
    plt.legend(loc="best", ncol=5, fontsize=8, title="CIFAR-10 classes")
    plt.title(f"t-SNE of ResNet18 penultimate features (n={X.shape[0]}, perplexity={args.perplexity})")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
