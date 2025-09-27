# MY (SOON TO BE YOUR) DL Repo

A production-ready, config-driven PyTorch workspace that covers core deep learning areas:
Foundations, Computer Vision, NLP, Generative AI, RL, GNNs, and SSL.

## Quickstart

```bash
# 1) Unzip and enter the repo
unzip your-dl-repo.zip && cd your-dl-repo

# 2) Create a virtual env
python3 -m venv .venv && source .venv/bin/activate

# 3) Install deps and the package
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 4) Run tests
pytest -q

# 5) Train a baseline model (ResNet18 on CIFAR10)
python train.py --config configs/cnn_resnet_cifar10.yaml
```

## Repo Layout
```
your-dl-repo/
├── .github/workflows/main.yml
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
├── configs/
├── data/
├── notebooks/
├── saved_models/
├── src/dlrepo/
├── tests/
├── train.py
└── predict.py
```

## Notes
- Config-first: edit `configs/*.yaml` to control experiments.
- Installable package: `pip install -e .` enables `from dlrepo...` imports anywhere.
- CI: GitHub Actions runs `pytest` on push/PR.

<!-- 🐾 secret breadcrumb: revisit TensorBoard (images + scalars UX; add confusion matrix) -->

- YOLO detect demo:
  `python tools/detect_demo.py --source 0 --seconds 5 --stride 2 --max-size 640 --config configs/detect_yolo.yaml --output outputs/detect_cam_smoke.mp4`
