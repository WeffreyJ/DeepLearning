

---

# DeepLearning Repo — Guide & Playbook

> A compact, batteries-included workspace for image classification and lightweight perception (monocular depth + video input). Optimized for quick iteration on laptop/Jetson and reproducible runs.

## What’s inside

* **Vision classification**

  * CIFAR-10/100 via `torchvision`
  * Models: `SimpleCNN`, `ResNet18/34/50`, `TinyTransformer`, `MLP_MNIST`
  * Features: EMA shadow weights, cosine scheduler, mixup, CSV logs, TensorBoard scalars + “hard examples”
  * Tools: confusion matrices, per-class accuracy, t-SNE embeddings, augment viz, test evaluator
* **Perception**

  * **Monocular depth (MiDaS-small)** with robust transforms
  * **Video I/O** from webcam, files, or YouTube (via `yt-dlp`)
  * **Side-by-side depth visualization**, downscaling, stride sampling
  * Ready stubs for BYTETRACK + lightweight costmap (for future nav)
  * ONNX/TensorRT export helper

---

## Repo layout (high level)

```
configs/                         # YAML configs for training & tools
src/dlrepo/
  data_processing/loader.py      # CIFAR-10/100, MNIST loaders + transforms
  models/                        # CNNs/Transformer/MLP + depth models
  training/                      # Trainer (EMA, TB), callbacks
  utils/                         # metrics, logging, CSV, EMA, augment, etc.
  perception/                    # video IO, core graph, depth utilities
  tracking/, nav/                # tracker + costmap stubs
tools/
  train.py (entry)               # CLI training driver
  eval_on_test.py                # accuracy on test set
  per_class_accuracy.py          # confusion + per-class metrics
  tsne_embeddings.py             # embedding visualization
  show_augmentations.py          # view augment pipeline
  grad_cam.py                    # (WIP, see notes)
  depth_infer.py                 # batch images → depth PNGs
  depth_demo.py                  # webcam/file/YouTube → depth video
  yt_depth_demo.py               # legacy YouTube demo (kept for reference)
  export_onnx_trt.py             # model export helper
  pack_apply*.py                 # patch-pack installers (features/perception)
```

---

## Requirements

* Python 3.9+ (3.10+ recommended by some libs; repo works on 3.9)
* PyTorch ≥ 2.1 (you’re on 2.8.0) with CUDA **or** MPS **or** CPU
* Core deps: `torchvision`, `tqdm`, `numpy>=2.0`, `pandas`, `imageio[ffmpeg]`, `opencv-python`, `timm`, `tensorboard`, `yt-dlp`
* macOS notes: you’ll see `urllib3` **LibreSSL** warnings—harmless for this workflow.

Install:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick sanity sweep (5–10 min)

```bash
# 1) Unit tests
pytest -q

# 2) CIFAR-10 smoke training (6 epochs w/ TensorBoard + EMA)
EMA_DECAY=0.999 python train.py --config configs/_smoke_cifar10.yaml

# 3) Inspect TensorBoard scalars & images
tensorboard --logdir saved_models/smoke_tb/tb --port 6006

# 4) Evaluate the best checkpoint
python tools/eval_on_test.py \
  --config configs/_smoke_cifar10.yaml \
  --checkpoint saved_models/smoke_tb/best.pth

# 5) Per-class accuracy
python tools/per_class_accuracy.py \
  --config configs/_smoke_cifar10.yaml \
  --checkpoint saved_models/smoke_tb/best.pth

# 6) t-SNE embeddings (sample 1k)
python tools/tsne_embeddings.py \
  --config configs/_smoke_cifar10.yaml \
  --checkpoint saved_models/smoke_tb/best.pth \
  --n 1000 --perplexity 30
```

---

## Training recipes

### A) CIFAR-10 (ResNet18 baseline)

```bash
python train.py --config configs/_smoke_cifar10.yaml
```

Key fields (edit the YAML):

```yaml
model: { name: resnet18, num_classes: 10 }
optimizer: { name: sgd, lr: 0.05, weight_decay: 5e-4, momentum: 0.9 }
scheduler: { name: cosine, t_max: 6 }
training:
  epochs: 6
  batch_size: 128
  num_workers: 2
  device: auto        # cuda | mps | cpu
  label_smoothing: 0.1
  mixup_alpha: 0.0
output:
  dir: saved_models/smoke_tb
  csv_log: saved_models/smoke_tb/train_log.csv
```

### B) CIFAR-100 (ResNet50)

```bash
python train.py --config configs/sgd_cosine_cifar100.yaml
# or fast short run
python train.py --config configs/sgd_cosine_cifar100_fast.yaml
```

The CIFAR-100 dataset is automatically downloaded by `torchvision`.

---

## Trainer features (already wired)

* **EMA weights**: better eval stability on checkpoints
  Set env: `EMA_DECAY=0.999` (default 0.9999)
* **TensorBoard**:

  * Scalars: `train/acc`, `train/loss`, `train/lr`, `val/acc`, `val/loss`, `val/best_acc`
  * Images: `val/hard_examples` (top-confidence wrong predictions every 5 epochs)
* **Early stopping** (configurable patience; off by default)
* **MixUp** (`training.mixup_alpha > 0`)
* **Label smoothing** (`training.label_smoothing`)

> Checkpoints save **EMA** weights on improvement as `best.pth`.

---

## Evaluation & visualization tools

* **Test set accuracy**

  ```bash
  python tools/eval_on_test.py \
    --config <your_config.yaml> \
    --checkpoint <path_to_best.pth>
  ```

* **Per-class accuracy**

  ```bash
  python tools/per_class_accuracy.py \
    --config <config> \
    --checkpoint <ckpt>
  ```

* **t-SNE embeddings**

  ```bash
  python tools/tsne_embeddings.py \
    --config <config> \
    --checkpoint <ckpt> \
    --n 2000 --perplexity 30
  ```

* **Augmentation viewer**

  ```bash
  python tools/show_augmentations.py --config <config>
  ```

* **Grad-CAM** *(WIP; requires grad-enabled forward and target layer selection)*

---

## Perception: monocular depth demos

### 1) Batch images → depth PNGs

```bash
# Create a few sample images (optional)
python - <<'PY'
import imageio.v2 as iio, numpy as np, os
os.makedirs("demo", exist_ok=True)
for i in range(3):
    iio.imwrite(f"demo/random_{i}.jpg",(np.random.rand(240,320,3)*255).astype('uint8'))
print("Wrote 3 images to ./demo/")
PY

# Infer depth maps
python tools/depth_infer.py --images "demo/*.jpg" --config configs/depth_midas_infer.yaml
# → outputs/depth/depth_XXX.png
```

### 2) Live or file/URL → side-by-side depth video

```bash
# Webcam (5s, stride=2, downscale max side 640)
python tools/depth_demo.py \
  --source 0 --seconds 5 --stride 2 --max-size 640 \
  --output outputs/depth_cam_smoke.mp4

# File
python tools/depth_demo.py \
  --source path/to/video.mp4 --seconds 10 --stride 2 --max-size 640 \
  --output outputs/depth_file.mp4

# YouTube (auto-resolve streaming URL via yt-dlp)
python tools/depth_demo.py \
  --youtube "https://www.youtube.com/watch?v=<id>" \
  --seconds 10 --stride 3 --max-size 640 \
  --output outputs/depth_yt_smoke.mp4
```

**Notes**

* Uses `MiDaS_small` with robust transform handling; normalizes depth to `[0,1]`, applies MAGMA colormap, and writes a two-panel video (RGB | Depth).
* YouTube path selects a playable stream via `yt-dlp`. If HLS (`.m3u8`), it switches to an OpenCV adapter for reliability.
* `--max-size` and `--stride` control throughput (useful on Jetson).

---

## Export helpers

* `tools/export_onnx_trt.py` provides a starting point for exporting models to ONNX and (later) TensorRT. Adapt input sizes and dynamic axes for your deployment.

---

## Configuration anatomy (training)

```yaml
task: vision_classification
dataset: cifar10 | cifar100 | mnist
data:
  root: data/raw

model:
  name: resnet50             # resnet18|34|50|simplecnn|tiny_transformer|mlp_mnist
  num_classes: 100

optimizer:
  name: sgd
  lr: 0.1
  momentum: 0.9
  weight_decay: 5e-4

scheduler:
  name: cosine
  t_max: 200

training:
  epochs: 200
  batch_size: 128
  num_workers: 4
  device: auto               # cuda|mps|cpu
  val_split: 0.1
  seed: 42
  label_smoothing: 0.1
  mixup_alpha: 0.0
  pin_memory: true

output:
  dir: saved_models/resnet50_cifar100
  csv_log: saved_models/resnet50_cifar100/train_log.csv
```

---

## “Pack” installers (optional automation)

* `tools/pack_apply.py` — patches trainer, adds EMA/hard-examples, etc.
* `tools/pack_apply_perception.py` — installs perception core, depth model, video I/O, configs, runners.

Use:

```bash
python tools/pack_apply_perception.py --pack perception_core_pack --commit
python tools/pack_apply_perception.py --pack depth_track_nav_pack --commit
```

Each pack writes files, compiles them, optionally runs tests, and commits in one go.

---

## Troubleshooting (real-world gotchas we handled)

* **NumPy 2.0** removed `ndarray.ptp()`
  Fixed in code by using `np.ptp(arr)`. If you hit it elsewhere, replace `arr.ptp()` → `np.ptp(arr)` and `arr.min()` → `np.min(arr)`.

* **Missing deps**

  * `tensorboard`: `pip install tensorboard`
  * `timm` (MiDaS backbone): `pip install timm`
  * `yt-dlp` (YouTube URLs): `pip install yt-dlp`

* **YouTube errors** (`pytube` 400s, storyboard URLs)
  We use `yt-dlp` now. If `imageio` can’t open HLS, our script falls back to OpenCV adapter automatically.

* **MPS pin_memory warning (macOS)**
  Harmless: PyTorch warns pin_memory isn’t used on MPS.

* **TensorBoard shows no data**
  Ensure you ran a TB-enabled training (e.g., `_smoke_cifar10.yaml`) and point TB to the run’s TB dir:
  `tensorboard --logdir saved_models/smoke_tb/tb`

* **Large output files in git**
  The repo currently stores some MP4s (fine for now). If you want to stop tracking artifacts later, add them to `.gitignore` and remove from history.

---

## How we run & track

* **Reproducibility**

  * Use YAML configs for all experiments.
  * Model checkpoints saved on validation improvement (EMA weights).
  * CSV logs at `output.csv_log`.

* **TensorBoard**

  * Write scalars every epoch.
  * Hardest wrong predictions (images) every 5 epochs.

* **Git workflow**

  * Pull with rebase if editing online & locally:

    ```bash
    git stash push -u -m "tmp: local"
    git pull --rebase origin main
    git stash pop
    ```
  * Tag milestones:

    ```bash
    git tag -a v0.2-perception-sanity -m "Perception core + depth demos + training smoke OK"
    git push origin --tags
    ```

---

## Milestone (what’s accomplished)

* CIFAR-10/100 training/eval (ResNet18/50), CSV logs, EMA, mixup, label smoothing, cosine scheduler.
* TensorBoard scalars + “hard examples” image panel.
* Per-class accuracy + t-SNE embedding visualizer + augment viewer.
* Robust MiDaS-small depth pipeline:

  * Batch images → depth PNGs
  * Webcam/file/YouTube → side-by-side video
  * HLS handling via OpenCV adapter; downscale/stride controls
* Patch packs to add features & perception in one command.
* Ready stubs for detection/tracking/nav and ONNX/TRT export.

---

## Jetson notes (when you migrate)

* Install JetPack and the **matching** PyTorch wheels for your CUDA/cuDNN.
* Use `--max-size 480/640` and `--stride 2/3` for real-time depth.
* Consider converting MiDaS-small to ONNX → TensorRT (your `export_onnx_trt.py` is the starting point).
* Run headless without preview windows and write to disk or a socket for downstream stacks.

---

## Appendix: Handy one-liners

```bash
# Environment snapshot
python - <<'PY'
import torch, platform
dev = ("cuda" if torch.cuda.is_available() else
       "mps"  if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available()
              else "cpu")
print("Python:", platform.python_version(), "| Torch:", torch.__version__, "| Device:", dev)
PY

# Clean compile a few files
python -m py_compile src/dlrepo/training/trainer.py src/dlrepo/utils/ema.py

# List best checkpoints we’ve produced
ls -lh saved_models/*/*best*.pth
```


