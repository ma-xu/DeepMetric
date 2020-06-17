from .BN_Inception import BN_Inception
from .resnet_old import *
from torchvision.models.resnet import ResNet

__factory = {
    'BN-Inception': BN_Inception,
    'resnet50': resnet50,
    'resnet18': resnet18
}

def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown network:", name)
    return __factory[name](*args, **kwargs)
