import numpy as np
import torch

# Define floating point variable precision
NP_FLOAT = np.float64
TORCH_FLOAT = torch.float64
optimizer_export_paraview = True
optimizer_export_tensorboard = True


# if torch.cuda.is_available():
#     DEVICE = torch.device("cuda")
#     TORCH_BACKEND = "cuda"
# else:
# CPU is faster for some reason
DEVICE = torch.device("cpu")
TORCH_BACKEND = "cpu"


def torch_to_np_float(x):
    """
    Converts a torch tensor to a numpy array of floating point variables
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")
    return x.detach().cpu().numpy().astype(NP_FLOAT, copy=False)


def np_to_torch_float(x, device=DEVICE):
    """
    Converts a numpy array to a torch tensor of floating point variables
    """
    x = np.asarray(x, dtype=NP_FLOAT)
    return torch.as_tensor(x, dtype=TORCH_FLOAT, device=device)