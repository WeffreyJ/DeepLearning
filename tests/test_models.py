import torch
from dlrepo.models.cnn import SimpleCNN

def test_simplecnn_forward():
    model = SimpleCNN(num_classes=10)
    x = torch.randn(2,3,32,32)
    y = model(x)
    assert y.shape == (2,10)
