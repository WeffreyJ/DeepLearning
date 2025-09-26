import argparse, torch, yaml
from torchvision import datasets, transforms
from dlrepo.models.cnn import resnet18

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    data_dir = cfg.get("data", {}).get("root", "data/raw")
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tfm)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)
    model = resnet18(num_classes=10, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"]); model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model.to(device)
    correct = n = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item(); n += y.size(0)
    print(f"Test accuracy: {correct/n:.4f}")
if __name__ == "__main__":
    main()
