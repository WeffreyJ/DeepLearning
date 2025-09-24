import torch
import torch.nn as nn

class MLP_MNIST(nn.Module):
    """Simple 2-layer MLP for MNIST (1x28x28)."""
    def __init__(self, hidden: int = 256, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
