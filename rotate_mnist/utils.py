import torch
import torchvision.transforms.functional as TF
import numpy as np
import random

def get_device(gpuID):
    if torch.cuda.is_available():
        device = "cuda:" + str(gpuID)
    else:
        device = "cpu"
    return device


def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True # input doesn't vary
    np.random.seed(random_seed)
    random.seed(random_seed)


class MyRandomRotation:
    def __init__(self, range):
        self.range = range

    def __call__(self, x):
        angle = np.random.uniform(self.range[0], self.range[1])
        return TF.rotate(x, angle)