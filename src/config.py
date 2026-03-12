import numpy as np
import torch

# Define floating point variable precision
NP_FLOAT = np.float64
TORCH_FLOAT = torch.float64


def torch_to_np_float(x):
    """
    Converts a torch tensor to a numpy array of floating point variables
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")
    return x.detach().cpu().numpy().astype(NP_FLOAT, copy=False)


def np_to_torch_float(x, device=None):
    """
    Converts a numpy array to a torch tensor of floating point variables
    """
    x = np.asarray(x, dtype=NP_FLOAT)
    return torch.as_tensor(x, dtype=TORCH_FLOAT, device=device)
