# src/dlrepo/models/cnn.py
import torch
import torch.nn as nn

# Torchvision ResNet backbones
from torchvision.models import (
    resnet18 as tv_resnet18,
    resnet34 as tv_resnet34,
    resnet50 as tv_resnet50,
)
try:
    # Available in torchvision>=0.13
    from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
except Exception:
    ResNet18_Weights = ResNet34_Weights = ResNet50_Weights = None


# ---- Simple baseline CNN for CIFAR-like images (3x32x32) ----
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # 1x1
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ---- ResNet factories ----
def resnet18(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if (pretrained and ResNet18_Weights) else None
    m = tv_resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def resnet34(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    weights = ResNet34_Weights.IMAGENET1K_V1 if (pretrained and ResNet34_Weights) else None
    m = tv_resnet34(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def resnet50(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V1 if (pretrained and ResNet50_Weights) else None
    m = tv_resnet50(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m
