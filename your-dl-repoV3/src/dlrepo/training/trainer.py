from typing import Optional
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..utils.metrics import accuracy
from ..utils.logging import get_logger
from ..utils.csv_logger import CSVLogger
from .callbacks import EarlyStopping

class Trainer:
    def __init__(self, model, optimizer, loss_fn, device: Optional[str] = None, ckpt_dir: str = "saved_models",
                 scheduler=None, csv_log_path: Optional[str] = None, early_stopping_patience: Optional[int] = None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device if device and device != "auto" else ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")))
        self.model.to(self.device)
        self.logger = get_logger("Trainer")
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.scheduler = scheduler
        self.csv_logger = CSVLogger(csv_log_path, ["epoch","train_loss","train_acc","val_loss","val_acc"]) if csv_log_path else None
        self.early = EarlyStopping(patience=early_stopping_patience) if early_stopping_patience else None

    def train_one_epoch(self, loader: DataLoader):
        self.model.train()
        running_loss, running_acc, n = 0.0, 0.0, 0
        for x, y in tqdm(loader, desc="Train", leave=False):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            loss.backward()
            self.optimizer.step()
            b = y.size(0)
            running_loss += loss.item() * b
            running_acc += accuracy(logits, y) * b
            n += b
        return running_loss / n, running_acc / n

    @torch.no_grad()
    def evaluate(self, loader: DataLoader):
        self.model.eval()
        running_loss, running_acc, n = 0.0, 0.0, 0
        for x, y in tqdm(loader, desc="Eval", leave=False):
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            b = y.size(0)
            running_loss += loss.item() * b
            running_acc += accuracy(logits, y) * b
            n += b
        return running_loss / n, running_acc / n

    def fit(self, epochs: int, train_loader: DataLoader, val_loader: DataLoader, ckpt_name: str = "model.pth"):
        best_acc = 0.0
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)
        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = self.train_one_epoch(train_loader)
            va_loss, va_acc = self.evaluate(val_loader)
            if self.scheduler is not None:
                self.scheduler.step()
            self.logger.info(f"Epoch {epoch}: train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={va_loss:.4f} acc={va_acc:.4f}")
            if self.csv_logger:
                self.csv_logger.log({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
            if va_acc > best_acc:
                best_acc = va_acc
                torch.save({"model_state": self.model.state_dict()}, ckpt_path)
                self.logger.info(f"Saved new best to {ckpt_path} (val_acc={best_acc:.4f})")
            if self.early:
                self.early.step(va_acc)
                if self.early.should_stop:
                    self.logger.info("Early stopping triggered.")
                    break
        return best_acc
