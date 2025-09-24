import argparse, yaml, torch, torch.nn as nn, torch.optim as optim, os
from src.dlrepo.data_processing.loader import get_dataloaders
from src.dlrepo.models.cnn import SimpleCNN, resnet18
from src.dlrepo.models.transformer import TinyTransformerClassifier
from src.dlrepo.models.mlp import MLP_MNIST
from src.dlrepo.training.trainer import Trainer

def build_model(cfg):
    name = cfg.get("model", {}).get("name", "resnet18").lower()
    if name == "resnet18":
        return resnet18(num_classes=cfg["model"].get("num_classes", 10), pretrained=False)
    if name == "simplecnn":
        return SimpleCNN(num_classes=cfg["model"].get("num_classes", 10))
    if name == "mlp_mnist":
        mcfg = cfg["model"]
        return MLP_MNIST(hidden=mcfg.get("hidden", 256), num_classes=mcfg.get("num_classes", 10))
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
    name = str(ocfg.get("name", "adam")).lower()
    lr = float(ocfg.get("lr", 1e-3))
    wd = float(ocfg.get("weight_decay", 0.0))
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        momentum = float(ocfg.get("momentum", 0.9))
        return optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    raise ValueError(f"Unknown optimizer: {name}")

def build_scheduler(cfg, optimizer):
    scfg = cfg.get("scheduler", {})
    name = scfg.get("name", None)
    if not name:
        return None
    name = name.lower()
    if name == "step_lr":
        return optim.lr_scheduler.StepLR(optimizer, step_size=int(scfg.get("step_size", 1)), gamma=float(scfg.get("gamma", 0.9)))
    if name == "multistep_lr":
        milestones = scfg.get("milestones", [5,10])
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=float(scfg.get("gamma", 0.1)))
    if name == "cosine":
        T_max = int(scfg.get("t_max", 10))
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg.get("task", "") == "vision_classification":
        train_loader, val_loader = get_dataloaders(cfg)
        model = build_model(cfg)
        optimizer = build_optimizer(cfg, model)
        scheduler = build_scheduler(cfg, optimizer)
        loss_fn = nn.CrossEntropyLoss()
        out_dir = cfg.get("output", {}).get("dir", "saved_models")
        os.makedirs(out_dir, exist_ok=True)
        csv_log = cfg.get("output", {}).get("csv_log", None)
        patience = cfg.get("training", {}).get("early_stopping_patience", None)
        trainer = Trainer(
            model, optimizer, loss_fn,
            device=cfg.get("training", {}).get("device", "auto"),
            ckpt_dir=out_dir,
            scheduler=scheduler,
            csv_log_path=csv_log,
            early_stopping_patience=patience
        )
        best = trainer.fit(cfg["training"]["epochs"], train_loader, val_loader, ckpt_name="best.pth")
        print(f"Best val acc: {best:.4f}")
    else:
        raise SystemExit("This starter wires up CIFAR10 & MNIST vision training. Extend loader.py for NLP or other tasks.")

if __name__ == "__main__":
    main()
