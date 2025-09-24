import torch
from dlrepo.models.mlp import MLP_MNIST
def test_mlp_mnist_forward():
    m = MLP_MNIST(hidden=64, num_classes=10)
    x = torch.randn(4,1,28,28)
    y = m(x)
    assert y.shape == (4,10)
