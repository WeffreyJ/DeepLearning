import torch.nn as nn
class MLP_MNIST(nn.Module):
    def __init__(self, hidden=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, hidden), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )
    def forward(self,x): return self.net(x)
