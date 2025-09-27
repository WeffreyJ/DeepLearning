import argparse, yaml, torch
from torchvision import datasets, transforms
from dlrepo.models.cnn import resnet18, resnet34, resnet50, SimpleCNN

def build_model_by_name(name: str, num_classes: int):
    name = name.lower()
    if name == "resnet18": return resnet18(num_classes=num_classes, pretrained=False)
    if name == "resnet34": return resnet34(num_classes=num_classes, pretrained=False)
    if name == "resnet50": return resnet50(num_classes=num_classes, pretrained=False)
    if name == "simplecnn": return SimpleCNN(num_classes=num_classes)
    raise SystemExit(f"Unsupported model for eval: {name}")

def get_test_loader(dataset: str, data_dir: str, batch_size: int = 256):
    dataset = dataset.lower()
    norm = ((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*norm)])
    if dataset in ("cifar10", "cifar-10"):
        ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tfm)
    elif dataset in ("cifar100", "cifar-100"):
        ds = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=tfm)
    else:
        raise SystemExit(f"Unsupported dataset for eval: {dataset}")
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2), ds.classes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    ds_name = cfg.get("dataset", "cifar10")
    data_dir = cfg.get("data", {}).get("root", "data/raw")
    num_classes = cfg.get("model", {}).get("num_classes", 10)
    model_name = cfg.get("model", {}).get("name", "resnet18")

    loader, _ = get_test_loader(ds_name, data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = build_model_by_name(model_name, num_classes=num_classes).to(device).eval()

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    correct = 0; n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            n += y.size(0)
    print(f"Test accuracy: {correct / n:.4f}")

if __name__ == "__main__":
    main()
