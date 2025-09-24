from typing import Tuple, Dict, Any, Optional
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch

__all__ = ["build_transforms", "get_dataloader", "get_dataloaders"]

def build_transforms(dataset_name: str, train: bool = True, config: Optional[Dict[str, Any]] = None) -> transforms.Compose:
    name = dataset_name.lower()
    cfg_aug = (config or {}).get("augment", {})
    if name in ("cifar10", "cifar-10"):
        t = []
        if train:
            aug = cfg_aug.get("cifar10_train", {})
            if aug.get("random_crop"):
                t.append(transforms.RandomCrop(aug.get("random_crop", 32), padding=aug.get("padding", 4)))
            if aug.get("random_horizontal_flip", True):
                t.append(transforms.RandomHorizontalFlip())
            ra = aug.get("randaugment", None)
            if isinstance(ra, dict):
                try:
                    t.append(transforms.RandAugment(num_ops=int(ra.get("n",2)), magnitude=int(ra.get("m",9))))
                except Exception:
                    pass
        t += [transforms.ToTensor(),
              transforms.Normalize((0.4914, 0.4822, 0.4465),
                                   (0.2023, 0.1994, 0.2010))]
        return transforms.Compose(t)
    elif name == "mnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    else:
        return transforms.ToTensor()

def _load_dataset(name: str, data_dir: str, train: bool, tfm: transforms.Compose):
    n = name.lower()
    if n in ("cifar10", "cifar-10"):
        return datasets.CIFAR10(root=data_dir, train=train, download=True, transform=tfm)
    elif n == "mnist":
        return datasets.MNIST(root=data_dir, train=train, download=True, transform=tfm)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

def get_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    ds_name = config.get("dataset", "cifar10").lower()
    data_dir = config.get("data", {}).get("root", "data/raw")
    batch_size = int(config.get("training", {}).get("batch_size", 128))
    num_workers = int(config.get("training", {}).get("num_workers", 2))
    pin_memory = bool(config.get("training", {}).get("pin_memory", True))
    val_split = float(config.get("training", {}).get("val_split", 0.1))
    seed = int(config.get("training", {}).get("seed", 42))

    tfm_train = build_transforms(ds_name, train=True, config=config)
    full_train = _load_dataset(ds_name, data_dir, train=True, tfm=tfm_train)

    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=num_workers > 0)
    # force eval transforms for val
    tfm_val = build_transforms(ds_name, train=False, config=config)
    val_ds.dataset.transform = tfm_val
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory,
                            persistent_workers=num_workers > 0)
    return train_loader, val_loader

def get_dataloader(dataset_name: str, data_dir: str = "data/raw", batch_size: int = 128,
                   train: bool = True, num_workers: int = 2, pin_memory: bool = True,
                   config: Optional[Dict[str, Any]] = None) -> DataLoader:
    tfm = build_transforms(dataset_name, train=train, config=config)
    ds = _load_dataset(dataset_name, data_dir, train=train, tfm=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=num_workers > 0)
