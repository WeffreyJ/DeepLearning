from dlrepo.data_processing.loader import build_transforms

def test_build_transforms_cifar10():
    tfm = build_transforms("cifar10")
    assert tfm is not None
