# src/dlrepo/data_processing/loader.py
from typing import Tuple, Dict, Any, Optional
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

__all__ = ["build_transforms", "get_dataloader", "get_dataloaders"]

def build_transforms(dataset_name: str, train: bool = True) -> transforms.Compose:
    """Basic transforms per dataset. CIFAR-10/100 share stats here."""
    name = dataset_name.lower()
    if name in ("cifar10", "cifar-10", "cifar100", "cifar-100"):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    elif name == "mnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    else:
        # fallback for quick prototyping
        return transforms.ToTensor()

def _load_dataset(name: str, data_dir: str, train: bool, tfm: transforms.Compose):
    """Factory to instantiate a torchvision dataset by name."""
    n = name.lower()
    if n in ("cifar10", "cifar-10"):
        return datasets.CIFAR10(root=data_dir, train=train, download=True, transform=tfm)
    if n in ("cifar100", "cifar-100"):
        return datasets.CIFAR100(root=data_dir, train=train, download=True, transform=tfm)
    if n == "mnist":
        return datasets.MNIST(root=data_dir, train=train, download=True, transform=tfm)
    raise ValueError(f"Unsupported dataset: {name}")

def get_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) following config schema."""
    ds_name: str = config.get("dataset", "cifar10").lower()
    data_dir: str = config.get("data", {}).get("root", "data/raw")
    batch_size: int = config.get("training", {}).get("batch_size", 128)
    num_workers: int = config.get("training", {}).get("num_workers", 2)
    pin_memory: bool = config.get("training", {}).get("pin_memory", True)
    val_split: float = config.get("training", {}).get("val_split", 0.1)
    seed: Optional[int] = config.get("training", {}).get("seed", 42)

    tfm_train = build_transforms(ds_name, train=True)
    full_train = _load_dataset(ds_name, data_dir, train=True, tfm=tfm_train)

    # deterministic split
    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    g = torch.Generator().manual_seed(seed if seed is not None else 0)
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=g)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader

def get_dataloader(dataset_name: str,
                   data_dir: str = "data/raw",
                   batch_size: int = 128,
                   train: bool = True,
                   num_workers: int = 2,
                   pin_memory: bool = True) -> DataLoader:
    """Single-loader helper (useful for quick scripts/tests)."""
    tfm = build_transforms(dataset_name, train=train)
    ds = _load_dataset(dataset_name, data_dir, train=train, tfm=tfm)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=train,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
