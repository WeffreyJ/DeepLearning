from typing import Optional
import os, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..utils.metrics import accuracy
from ..utils.logging import get_logger
from ..utils.csv_logger import CSVLogger
from ..utils.ema import ModelEMA
from .callbacks import EarlyStopping
from ..utils.augment import mixup_data, mixup_criterion
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, model, optimizer, loss_fn, device: Optional[str] = None, ckpt_dir: str = "saved_models",
                 scheduler=None, csv_log_path: Optional[str] = None, early_stopping_patience: Optional[int] = None,
                 mixup_alpha: float = 0.0):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(
            device if device and device != "auto"
            else ("cuda" if torch.cuda.is_available()
                  else ("mps" if torch.backends.mps.is_available() else "cpu"))
        )
        self.model.to(self.device)
        self.logger = get_logger("Trainer")
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.scheduler = scheduler
        self.csv_logger = CSVLogger(csv_log_path, ["epoch","train_loss","train_acc","val_loss","val_acc"]) if csv_log_path else None
        self.early = EarlyStopping(patience=early_stopping_patience) if early_stopping_patience else None
        self.mixup_alpha = float(mixup_alpha)

        # EMA shadow weights
        self.ema = ModelEMA(self.model, decay=float(os.environ.get("EMA_DECAY", "0.9999")))

        # TensorBoard
        self.writer = SummaryWriter(log_dir=os.path.join(self.ckpt_dir, "tb"))

    def train_one_epoch(self, loader: DataLoader):
        self.model.train()
        running_loss = running_acc = 0.0; n = 0
        for x, y in tqdm(loader, desc="Train", leave=False):
            x, y = x.to(self.device), y.to(self.device)
            if self.mixup_alpha > 0.0:
                x, y_a, y_b, lam = mixup_data(x, y, self.mixup_alpha)
                logits = self.model(x)
                loss = mixup_criterion(self.loss_fn, logits, y_a, y_b, lam)
                acc = accuracy(logits, y)  # approx
            else:
                logits = self.model(x)
                loss = self.loss_fn(logits, y)
                acc = accuracy(logits, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # update EMA every step
            self.ema.update(self.model)
            b = y.size(0)
            running_loss += loss.item() * b
            running_acc  += acc * b
            n += b
        return running_loss / n, running_acc / n

    @torch.no_grad()
    def evaluate(self, loader: DataLoader):
        self.model.eval()
        running_loss = running_acc = 0.0; n = 0
        for x, y in tqdm(loader, desc="Eval", leave=False):
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            b = y.size(0)
            running_loss += loss.item() * b
            running_acc  += accuracy(logits, y) * b
            n += b
        return running_loss / n, running_acc / n

    def fit(self, epochs: int, train_loader: DataLoader, val_loader: DataLoader, ckpt_name: str = "model.pth"):
        best_acc = 0.0
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)
        try:
            for epoch in range(1, epochs + 1):
                tr_loss, tr_acc = self.train_one_epoch(train_loader)

                # ---- Evaluate with EMA weights ----
                # Make a true copy of current (training) weights so we can restore exactly
                _backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                self.ema.copy_to(self.model)
                va_loss, va_acc = self.evaluate(val_loader)
                self.model.load_state_dict(_backup, strict=True)
                # -----------------------------------

                if self.scheduler is not None:
                    self.scheduler.step()

                # Console & CSV
                self.logger.info(
                    f"Epoch {epoch}: train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                    f"val_loss={va_loss:.4f} acc={va_acc:.4f}"
                )
                if self.csv_logger:
                    self.csv_logger.log({
                        "epoch": epoch,
                        "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss,   "val_acc":  va_acc
                    })

                # TensorBoard scalars
                if self.writer:
                    self.writer.add_scalar("train/loss", tr_loss, epoch)
                    self.writer.add_scalar("train/acc",  tr_acc,  epoch)
                    self.writer.add_scalar("val/loss",   va_loss, epoch)
                    self.writer.add_scalar("val/acc",    va_acc,  epoch)
                    if self.optimizer.param_groups:
                        self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], epoch)

                # Checkpoint on improvement (save EMA weights)
                if va_acc > best_acc:
                    best_acc = va_acc
                    torch.save({"model_state": self.ema.ema.state_dict()}, ckpt_path)
                    self.logger.info(f"Saved new best to {ckpt_path} (val_acc={best_acc:.4f})")
                    if self.writer:
                        self.writer.add_scalar("val/best_acc", best_acc, epoch)

                # TensorBoard: hardest wrong predictions (every 5 epochs)
                if self.writer and (epoch % 5 == 0):
                    try:
                        self.model.eval()
                        x_vis, y_vis = next(iter(val_loader))
                        x_vis, y_vis = x_vis.to(self.device), y_vis.to(self.device)
                        with torch.no_grad():
                            logits = self.model(x_vis)
                            probs = torch.softmax(logits, dim=1)
                            conf, pred = probs.max(1)
                            wrong = pred.ne(y_vis)
                            if wrong.any():
                                import torch as _torch
                                idx = conf[wrong].topk(min(16, int(wrong.sum().item()))).indices
                                sel = _torch.where(wrong)[0][idx]
                                show = x_vis[sel]
                                # de-normalize CIFAR-ish stats for display
                                mean = _torch.tensor([0.4914,0.4822,0.4465], device=self.device).view(1,3,1,1)
                                std  = _torch.tensor([0.2023,0.1994,0.2010], device=self.device).view(1,3,1,1)
                                show = (show * std + mean).clamp(0,1)
                                self.writer.add_images("val/hard_examples", show, epoch)
                    except Exception:
                        pass

                # Early stopping
                if self.early:
                    self.early.step(va_acc)
                    if self.early.should_stop:
                        self.logger.info("Early stopping triggered.")
                        break
        finally:
            if self.writer:
                self.writer.flush()
                self.writer.close()
        return best_acc
