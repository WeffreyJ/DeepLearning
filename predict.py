import argparse, torch
from PIL import Image
from torchvision import transforms
from dlrepo.models.cnn import resnet18

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True, help="Path to an image file")
    args = ap.parse_args()

    model = resnet18(num_classes=10, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    img = tfm(Image.open(args.image).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(img)
        pred = torch.argmax(logits, dim=1).item()
    print(f"Predicted class index: {pred}")

if __name__ == "__main__":
    main()
