import argparse, yaml, torch, numpy as np
from torchvision import datasets, transforms
from dlrepo.models.cnn import resnet18, resnet34, resnet50, SimpleCNN

def build_model(cfg):
    name = cfg.get("model",{}).get("name","resnet18").lower()
    nc   = cfg.get("model",{}).get("num_classes",10)
    if name=="resnet18": return resnet18(nc, False)
    if name=="resnet34": return resnet34(nc, False)
    if name=="resnet50": return resnet50(nc, False)
    if name=="simplecnn": return SimpleCNN(nc)
    raise SystemExit(f"Unsupported model: {name}")

def get_loader(dataset, root, bs=256):
    norm = ((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*norm)])
    d = dataset.lower()
    if d in ("cifar10","cifar-10"):
        ds = datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)
    elif d in ("cifar100","cifar-100"):
        ds = datasets.CIFAR100(root=root, train=False, download=True, transform=tfm)
    else:
        raise SystemExit("Unsupported dataset")
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2), ds.classes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ds  = cfg.get("dataset","cifar10")
    root= cfg.get("data",{}).get("root","data/raw")

    loader, classes = get_loader(ds, root)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    model = build_model(cfg).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    correct = np.zeros(len(classes), dtype=np.int64)
    totals  = np.zeros(len(classes), dtype=np.int64)

    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            for c in range(len(classes)):
                mask = (y==c)
                totals[c]  += int(mask.sum())
                correct[c] += int((pred[mask]==c).sum())

    acc = (correct / np.maximum(totals,1)) * 100.0
    for c,a in enumerate(acc):
        print(f"{classes[c]:<12} {a:5.1f}%  (n={totals[c]})")
    print(f"\nOverall: {correct.sum()/totals.sum():.4f}")

if __name__ == "__main__":
    main()
