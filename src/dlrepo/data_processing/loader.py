from typing import Tuple, Dict, Any
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def build_transforms(dataset_name: str):
    if dataset_name.lower() == "cifar10":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
    return transforms.ToTensor()

def get_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    ds_name = config.get("dataset", "cifar10").lower()
    batch_size = config.get("training", {}).get("batch_size", 128)
    num_workers = config.get("training", {}).get("num_workers", 2)
    val_split = config.get("training", {}).get("val_split", 0.1)

    tfm = build_transforms(ds_name)

    if ds_name == "cifar10":
        train_ds = datasets.CIFAR10(root="data/raw", train=True, download=True, transform=tfm)
        val_size = int(len(train_ds) * val_split)
        train_size = len(train_ds) - val_size
        train_ds, val_ds = random_split(train_ds, [train_size, val_size])
    else:
        raise ValueError(f"Unsupported dataset: {ds_name}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
