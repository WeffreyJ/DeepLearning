import argparse, yaml, torch, torch.nn as nn, torch.optim as optim, os
from src.dlrepo.data_processing.loader import get_dataloaders
from src.dlrepo.models.cnn import SimpleCNN, resnet18
from src.dlrepo.models.transformer import TinyTransformerClassifier
from src.dlrepo.training.trainer import Trainer

def build_model(cfg):
    name = cfg.get("model", {}).get("name", "resnet18").lower()
    if name == "resnet18":
        return resnet18(num_classes=cfg["model"].get("num_classes", 10), pretrained=False)
    if name == "simplecnn":
        return SimpleCNN(num_classes=cfg["model"].get("num_classes", 10))
    if name in ("tiny_transformer", "transformer"):
        mcfg = cfg["model"]
        return TinyTransformerClassifier(
            vocab_size=mcfg.get("vocab_size", 20000),
            d_model=mcfg.get("d_model", 128),
            nhead=mcfg.get("nhead", 4),
            num_layers=mcfg.get("num_layers", 2),
            num_classes=mcfg.get("num_classes", 2),
        )
    raise ValueError(f"Unknown model: {name}")

def build_optimizer(cfg, model):
    ocfg = cfg.get("optimizer", {})
    name = ocfg.get("name", "adam").lower()
    lr = ocfg.get("lr", 1e-3)
    wd = ocfg.get("weight_decay", 0.0)
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Only CIFAR10 loader wired right now
    if cfg.get("task", "") == "vision_classification":
        train_loader, val_loader = get_dataloaders(cfg)
        model = build_model(cfg)
        optimizer = build_optimizer(cfg, model)
        loss_fn = nn.CrossEntropyLoss()
        out_dir = cfg.get("output", {}).get("dir", "saved_models")
        os.makedirs(out_dir, exist_ok=True)
        trainer = Trainer(model, optimizer, loss_fn, device=cfg.get("training", {}).get("device", "auto"), ckpt_dir=out_dir)
        best = trainer.fit(cfg["training"]["epochs"], train_loader, val_loader, ckpt_name="best.pth")
        print(f"Best val acc: {best:.4f}")
    else:
        raise SystemExit("This starter wires up CIFAR10 vision training. Extend loader.py for NLP or other tasks.")

if __name__ == "__main__":
    main()
