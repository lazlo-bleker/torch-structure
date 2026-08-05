import torch
import numpy as np

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

TORCH_FLOAT = torch.get_default_dtype()
NUMPY_FLOAT = np.dtype(float)
DEVICE = get_device()