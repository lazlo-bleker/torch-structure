from torch_structure.generators import CapCeilingAssemblyGenerator
import numpy as np


def f(u, v):
    coords_hat = np.zeros([u.shape[0], u.shape[1], 3])
    coords_hat[:, :, 0] = u
    coords_hat[:, :, 1] = v
    coords_hat[:, :, 2] = -(u**2)
    return coords_hat


generator = CapCeilingAssemblyGenerator(
    n_u=10,
    n_v=10,
    seed=0,
)
data = generator()
data = data.fdm()
data.plot()
print("f")
