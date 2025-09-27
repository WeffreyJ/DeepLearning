#!/usr/bin/env python3
"""
Pack runner: apply feature packs to this repo safely, without copy/paste.

Usage:
  python tools/pack_apply.py --pack ema_and_hard_examples [--commit]
  python tools/pack_apply.py --list

Design goals:
- Idempotent (safe to run multiple times)
- Backups for changed files: <file>.bak.<timestamp>
- Verifies .git, warns on dirty tree
- Compiles changed .py files
- Runs pytest -q
"""

import argparse, hashlib, os, re, shutil, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # repo root
PY = sys.executable

def sh(cmd, check=True, capture=False):
    if capture:
        return subprocess.run(cmd, cwd=REPO, shell=True, check=check, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
    return subprocess.run(cmd, cwd=REPO, shell=True, check=check)

def ensure_repo():
    if not (REPO / ".git").exists():
        print("ERROR: run from a git repo root ('.git' missing).")
        sys.exit(1)
    # warn on dirty tree
    out = sh("git status --porcelain", check=False, capture=True)
    if out.strip():
        print("WARNING: working tree not clean. Uncommitted changes present.\n")
    return True

def backup_file(p: Path):
    if not p.exists(): return
    ts = time.strftime("%Y%m%d-%H%M%S")
    bak = p.with_suffix(p.suffix + f".bak.{ts}")
    bak.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, bak)
    print(f"  - backup: {p} -> {bak}")

def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            old = f.read()
        if old == content:
            print(f"  = up-to-date: {path}")
            return False
        backup_file(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  + wrote: {path}")
    return True

def py_compile(paths):
    if not paths: return
    joined = " ".join(str(p) for p in paths)
    print(f"\n==> Compiling: {joined}")
    sh(f"{PY} -m py_compile {joined}")

def run_pytest():
    print("\n==> Running tests: pytest -q")
    try:
        sh("pytest -q")
        print("==> Tests passed.")
    except subprocess.CalledProcessError as e:
        print(e)
        sys.exit(2)

def patch_file(path: Path, mutator):
    """mutator: callable(text)->(new_text, changed_bool)"""
    changed = False
    if not path.exists():
        print(f"  ! missing file to patch: {path}")
        return False
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    new, ch = mutator(src)
    if ch:
        backup_file(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new)
        print(f"  * patched: {path}")
        changed = True
    else:
        print(f"  = already patched: {path}")
    return changed

# --------------------- PACK: EMA + Hard Examples --------------------- #

EMA_FILE = """\
import torch
import copy

class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.ema = copy.deepcopy(model).to(next(model.parameters()).device)
        self.ema.eval()
        self.decay = float(decay)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for ema_p, p in zip(self.ema.state_dict().values(), model.state_dict().values()):
            if isinstance(ema_p, torch.Tensor):
                ema_p.copy_(ema_p * d + p.detach() * (1. - d))

    @torch.no_grad()
    def copy_to(self, model):
        model.load_state_dict(self.ema.state_dict(), strict=True)
"""


def pack_ema_and_hard_examples(commit=False):
    changed_files = []

    # 1) Write EMA utility
    ema_py = REPO / "src/dlrepo/utils/ema.py"
    if write_file(ema_py, EMA_FILE):
        changed_files.append(ema_py)

    # 2) Patch trainer: import EMA, instantiate, update per step, eval with EMA, save EMA in best, and TB hard-examples
    trainer_py = REPO / "src/dlrepo/training/trainer.py"

    def mutate(src: str):
        changed = False

        # a) import
        if "from ..utils.ema import ModelEMA" not in src:
            src = src.replace("from ..utils.csv_logger import CSVLogger", 
                              "from ..utils.csv_logger import CSVLogger\nfrom ..utils.ema import ModelEMA")
            changed = True

        # b) in __init__: create EMA after mixup_alpha
        if "self.ema =" not in src:
            src = re.sub(r"(self\.mixup_alpha\s*=\s*float\(mixup_alpha\)\s*)",
                         r"\1\n        # EMA shadow weights\n        self.ema = ModelEMA(self.model, decay=float(os.environ.get(\"EMA_DECAY\", \"0.9999\")))\n",
                         src, count=1)
            changed = True

        # c) after optimizer.step() in train_one_epoch: update EMA
        if re.search(r"optimizer\.step\(\)[^\n]*\n\s*self\.ema\.update", src) is None:
            src = re.sub(r"(self\.optimizer\.step\(\))",
                         r"\1\n            self.ema.update(self.model)",
                         src, count=1)
            changed = True

        # d) in fit(): evaluate with EMA weights; also save EMA weights as best
        # Swap for val
        if "ema.copy_to(self.model)" not in src:
            src = re.sub(
                r"(va_loss,\s*va_acc\s*=\s*self\.evaluate\(val_loader\))",
                "            # Evaluate with EMA weights\n            _backup = self.model.state_dict()\n            self.ema.copy_to(self.model)\n            \\1\n            # Restore training weights\n            self.model.load_state_dict(_backup, strict=True)",
                src, count=1)
            changed = True

        # Save best from EMA
        if re.search(r"torch\.save\(\{\"model_state\":\s*self\.ema\.ema\.state_dict\(\)\}", src) is None:
            src = re.sub(
                r"torch\.save\(\{\"model_state\":\s*self\.model\.state_dict\(\)\},\s*ckpt_path\)",
                r"torch.save({\"model_state\": self.ema.ema.state_dict()}, ckpt_path)",
                src, count=1)
            changed = True

        # e) TB hard examples every 5 epochs (robust, works only if writer exists)
        if "val/hard_examples" not in src:
            snippet = r"""
                # --- TensorBoard: hardest wrong predictions (every 5 epochs) ---
                if getattr(self, "writer", None) and (epoch % 5 == 0):
                    try:
                        self.model.eval()
                        import torch
                        x_vis, y_vis = next(iter(val_loader))
                        x_vis, y_vis = x_vis.to(self.device), y_vis.to(self.device)
                        with torch.no_grad():
                            logits = self.model(x_vis)
                            probs = torch.softmax(logits, dim=1)
                            conf, pred = probs.max(1)
                            wrong = pred.ne(y_vis)
                            if wrong.any():
                                idx = conf[wrong].topk(min(16, int(wrong.sum().item()))).indices
                                sel = torch.where(wrong)[0][idx]
                                show = x_vis[sel]
                                # de-normalize CIFAR stats for display (safe clamp)
                                mean = torch.tensor([0.4914,0.4822,0.4465], device=self.device).view(1,3,1,1)
                                std  = torch.tensor([0.2023,0.1994,0.2010], device=self.device).view(1,3,1,1)
                                show = (show * std + mean).clamp(0,1)
                                self.writer.add_images("val/hard_examples", show, epoch)
                    except Exception:
                        pass
            """
            # append just before 'if self.early:' or before 'return best_acc'
            if "if self.early" in src:
                src = src.replace("if self.early:", snippet + "\n        if self.early:", 1)
            else:
                src = src.replace("return best_acc", snippet + "\n        return best_acc", 1)
            changed = True

        return src, changed

    if patch_file(trainer_py, mutate):
        changed_files.append(trainer_py)

    # 3) Compile changed files
    py_compile(changed_files)

    # 4) Run tests
    run_pytest()

    # 5) Commit (optional)
    if commit:
        msg = "feat(train): EMA weights + TensorBoard hard-examples"
        print(f"\n==> Committing: {msg}")
        sh("git add -A && git commit -m \"%s\"" % msg)
        print("==> Done. You can now push.")

# --------------------- PACK REGISTRY --------------------- #

PACKS = {
    "ema_and_hard_examples": pack_ema_and_hard_examples,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", help="Pack name to apply")
    ap.add_argument("--list", action="store_true", help="List available packs")
    ap.add_argument("--commit", action="store_true", help="Commit after successful apply+tests")
    args = ap.parse_args()

    ensure_repo()

    if args.list or not args.pack:
        print("Available packs:")
        for k in sorted(PACKS):
            print(f"  - {k}")
        if not args.pack:
            return

    fn = PACKS.get(args.pack)
    if not fn:
        print(f"Unknown pack: {args.pack}")
        sys.exit(1)

    fn(commit=args.commit)

if __name__ == "__main__":
    main()
