#!/usr/bin/env bash
set -euo pipefail

# Run from repo root (the one with .git/)
if [ ! -d .git ]; then
  echo "Run this from the repo root (where .git/ lives)"; exit 1
fi

PYTHON="${PYTHON:-python3}"

echo "==> Ensure tools/ exists"
mkdir -p tools

# ---------- helper: python in-place editor ----------
py_edit() {
"$PYTHON" - "$1" <<'PYEOF'
import io,sys,re,os
path=sys.argv[1]
src=open(path,"r",encoding="utf-8").read()

def w(s):
  with open(path,"w",encoding="utf-8") as f: f.write(s)

def add_cifar100_in_loader(s):
  if "datasets.CIFAR100" in s: return s, False
  # Insert CIFAR100 branch in _load_dataset
  pat = r"(def _load_dataset\(.*?\):\s*.*?CIFAR10.*?return datasets\.CIFAR10.*?\n)"
  m = re.search(pat, s, flags=re.S)
  if not m: return s, False
  insert = """
    if n in ("cifar100", "cifar-100"):
        return datasets.CIFAR100(root=data_dir, train=train, download=True, transform=tfm)
"""
  # n variable is used in many implementations; normalize to 'n = name.lower()'
  if "n = name.lower()" not in s:
    s = re.sub(r"(def _load_dataset\(name: str, data_dir: str, train: bool, tfm: transforms\.Compose\):\s*)",
               r"\1    n = name.lower()\n", s)
  s = re.sub(r'(return datasets\.CIFAR10\(.*?\)\n)',
             r'\1' + insert, s, count=1, flags=re.S)
  return s, True

def add_resnet_factories(s):
  changed=False
  if "from torchvision.models import (" not in s:
    s = s.replace("import torch\n", "import torch\nfrom torchvision.models import (\n    resnet18 as tv_resnet18,\n    resnet34 as tv_resnet34,\n    resnet50 as tv_resnet50,\n)\n", 1)
    changed=True
  if "ResNet50_Weights" not in s:
    s = s.replace("from torchvision.models import (", "from torchvision.models import (", 1)
    s = s.replace(")\n", ")\ntry:\n    from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights\nexcept Exception:\n    ResNet18_Weights = ResNet34_Weights = ResNet50_Weights = None\n", 1)
    changed=True
  if "def resnet18(" in s and "weights = ResNet18_Weights" not in s:
    s = re.sub(r"def resnet18\(.*?\):\s*?\n\s*m = tv_resnet18\(.*?\)\n\s*m\.fc = .*?\n\s*return m",
               "def resnet18(num_classes=10, pretrained=False):\n    weights = ResNet18_Weights.IMAGENET1K_V1 if (pretrained and ResNet18_Weights) else None\n    m = tv_resnet18(weights=weights)\n    import torch.nn as nn\n    m.fc = nn.Linear(m.fc.in_features, num_classes)\n    return m", s, flags=re.S)
    changed=True
  if "def resnet34(" not in s:
    s += "\n\ndef resnet34(num_classes=10, pretrained=False):\n    weights = ResNet34_Weights.IMAGENET1K_V1 if (pretrained and ResNet34_Weights) else None\n    m = tv_resnet34(weights=weights)\n    import torch.nn as nn\n    m.fc = nn.Linear(m.fc.in_features, num_classes)\n    return m\n"
    changed=True
  if "def resnet50(" not in s:
    s += "\n\ndef resnet50(num_classes=10, pretrained=False):\n    weights = ResNet50_Weights.IMAGENET1K_V1 if (pretrained and ResNet50_Weights) else None\n    m = tv_resnet50(weights=weights)\n    import torch.nn as nn\n    m.fc = nn.Linear(m.fc.in_features, num_classes)\n    return m\n"
    changed=True
  return s, changed

def patch_train_build_model(s):
  changed=False
  # imports
  if "resnet50" not in s:
    s = s.replace("from dlrepo.models.cnn import SimpleCNN, resnet18",
                  "from dlrepo.models.cnn import SimpleCNN, resnet18, resnet34, resnet50")
    changed=True
  # build_model
  if "def build_model" not in s:
    # Assume it already exists in your file; if not, skip
    return s, changed
  # Ensure branches
  if "name == \"resnet34\"" not in s or "name == \"resnet50\"" not in s:
    s = re.sub(r"(if name == \"resnet18\".*?return resnet18.*?\n)",
               r"\1    if name == \"resnet34\":\n        return resnet34(num_classes=num_classes, pretrained=False)\n    if name == \"resnet50\":\n        return resnet50(num_classes=num_classes, pretrained=False)\n",
               s, flags=re.S)
    changed=True
  return s, changed

path = sys.argv[1]
if path.endswith("loader.py"):
  new, ch = add_cifar100_in_loader(src); 
  if ch: w(new); print("  + loader: added CIFAR-100")
elif path.endswith("cnn.py"):
  new, ch = add_resnet_factories(src);
  if ch: w(new); print("  + cnn: added resnet34/50 factories (and weights handling)")
elif path.endswith("train.py"):
  new, ch = patch_train_build_model(src);
  if ch: w(new); print("  + train: added resnet34/50 to build_model()")
else:
  print("No-op", path)
PYEOF
}

echo "==> Patch loader (add CIFAR-100)"
py_edit src/dlrepo/data_processing/loader.py || true

echo "==> Patch cnn.py (add resnet34/50)"
py_edit src/dlrepo/models/cnn.py || true

echo "==> Patch train.py (wire resnet34/50)"
py_edit train.py || true

echo "==> Write CIFAR-100 configs"
mkdir -p configs
cat > configs/sgd_cosine_cifar100.yaml <<'YAML'
task: vision_classification
dataset: cifar100
data:
  root: data/raw

model:
  name: resnet50
  num_classes: 100

optimizer:
  name: sgd
  lr: 0.1
  weight_decay: 5e-4
  momentum: 0.9

scheduler:
  name: cosine
  t_max: 200

training:
  epochs: 200
  batch_size: 128
  num_workers: 2
  val_split: 0.1
  seed: 42
  device: auto
  pin_memory: true
  label_smoothing: 0.1
  mixup_alpha: 0.2

augment:
  cifar10_train:
    random_crop: 32
    padding: 4
    random_horizontal_flip: true
    randaugment: { n: 2, m: 9 }

output:
  dir: saved_models/resnet50_cifar100_sgdcos
  csv_log: saved_models/resnet50_cifar100_sgdcos/train_log.csv
YAML

cat > configs/sgd_cosine_cifar100_fast.yaml <<'YAML'
task: vision_classification
dataset: cifar100
data:
  root: data/raw

model:
  name: resnet50
  num_classes: 100

optimizer:
  name: sgd
  lr: 0.1
  weight_decay: 5e-4
  momentum: 0.9

scheduler:
  name: cosine
  t_max: 3

training:
  epochs: 3
  batch_size: 128
  num_workers: 2
  val_split: 0.1
  seed: 42
  device: auto
  pin_memory: true
  label_smoothing: 0.1
  mixup_alpha: 0.2

augment:
  cifar10_train:
    random_crop: 32
    padding: 4
    random_horizontal_flip: true
    randaugment: { n: 2, m: 9 }

output:
  dir: saved_models/resnet50_cifar100_sgdcos_fast
  csv_log: saved_models/resnet50_cifar100_sgdcos_fast/train_log.csv
YAML

echo "==> Update eval tool to read model/dataset from config"
cat > tools/eval_on_test.py <<'PY'
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
PY

echo "==> Install & test"
$PYTHON -m pip install -e . >/dev/null
pytest -q

echo "==> Done. Try:"
echo "python train.py --config configs/sgd_cosine_cifar100_fast.yaml"
echo "python tools/eval_on_test.py --config configs/sgd_cosine_cifar100_fast.yaml --checkpoint saved_models/resnet50_cifar100_sgdcos_fast/best.pth"
