import numpy as np
import torch

# Match datatype for consistency in the interoperability between scipy and torch
float_np = np.float64
float_torch = torch.float64

# Optimization tolerance
ftol = 1e-8

# CEM iterations when solving graph
max_iters_cem = 100