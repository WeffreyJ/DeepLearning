# Your DL Repo

A production-ready, config-driven PyTorch workspace.

## Quickstart
```bash
unzip your-dl-repo.zip && cd your-dl-repo
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest -q
python train.py --config configs/cnn_resnet_cifar10.yaml
```
## Higher-performance CIFAR-10
```bash
python train.py --config configs/sgd_cosine_cifar10.yaml
python test_eval.py --config configs/sgd_cosine_cifar10.yaml --checkpoint saved_models/resnet18_cifar10_sgdcos/best.pth
python tools/plot_metrics.py --csv saved_models/resnet18_cifar10_sgdcos/train_log.csv
python tools/plot_confusion.py --config configs/sgd_cosine_cifar10.yaml --checkpoint saved_models/resnet18_cifar10_sgdcos/best.pth
```
