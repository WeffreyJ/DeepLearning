#!/usr/bin/env bash
set -euo pipefail

pip install -q -r requirements.txt yt-dlp imageio-ffmpeg tensorboard

python - <<'PY'
import torch, platform
dev=("cuda" if torch.cuda.is_available() else
     "mps" if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available()
           else "cpu")
print("Python", platform.python_version(), "| Torch", torch.__version__, "| Device", dev)
PY

pytest -q

EMA_DECAY=0.999 python train.py --config configs/_smoke_cifar10.yaml
python tools/eval_on_test.py --config configs/_smoke_cifar10.yaml --checkpoint saved_models/smoke_tb/best.pth

python - <<'PY'
import imageio.v2 as iio, numpy as np, os, glob
os.makedirs("demo", exist_ok=True)
if not glob.glob("demo/*.jpg"):
    for i in range(3):
        iio.imwrite(f"demo/random_{i}.jpg",(np.random.rand(240,320,3)*255).astype('uint8'))
print("demo images ready")
PY

python tools/depth_infer.py --images "demo/*.jpg" --config configs/depth_midas_infer.yaml
python tools/depth_demo.py --source 0 --seconds 5 --stride 2 --max-size 640 --output outputs/depth_cam_smoke.mp4 || true

echo "All good ✅"
