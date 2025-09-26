import argparse, torch, yaml, itertools, matplotlib.pyplot as plt, numpy as np
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
from dlrepo.models.cnn import resnet18
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    data_dir = cfg.get("data", {}).get("root", "data/raw")
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tfm)
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)
    model = resnet18(num_classes=10, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"]); model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model.to(device)
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); p = model(x).argmax(1).cpu()
            ys.append(y); ps.append(p)
    y_true = torch.cat(ys).numpy(); y_pred = torch.cat(ps).numpy()
    cm = confusion_matrix(y_true, y_pred)
    classes = ds.classes
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest'); ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)), xticklabels=classes, yticklabels=classes, ylabel='True', xlabel='Predicted', title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max()/2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i,j], 'd'), ha="center", va="center")
    fig.tight_layout(); plt.show()
if __name__ == "__main__":
    main()
