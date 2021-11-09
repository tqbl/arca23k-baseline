import numpy as np
import torch


def determine_device(cuda=True):
    if torch.cuda.is_available() and cuda:
        return torch.device('cuda:0')
    return torch.device('cpu')


def ensure_reproducibility(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
