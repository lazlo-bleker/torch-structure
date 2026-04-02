from torch_structure.data import StructData
import torch
from config import TORCH_FLOAT


def orthogonal_func(graph_solved: "StructData", quad_indices):
    coords = graph_solved.coords
    x_qni = coords[quad_indices]
    length_q0 = torch.linalg.norm(x_qni[:, 0] - x_qni[:, 2], dim=1)
    length_q1 = torch.linalg.norm(x_qni[:, 1] - x_qni[:, 3], dim=1)
    diff = length_q0 - length_q1

    loss = torch.sum(diff * diff)

    return loss


def orthogonal_cache(graph_solved: StructData):
    kwargs = {}
    nodes_uv = graph_solved.uv_coords
    nu = int(max(nodes_uv[:, 0]) + 1)
    nv = int(max(nodes_uv[:, 1]) + 1)
    n_elements = (nu) * (nv - 1)
    quad_indices = -torch.ones([n_elements, 4], dtype=torch.int)
    for v in range(nv - 1):
        for u in range(nu):
            k = u * (nv - 1) + v
            quad_indices[k, 0] = int(((u) % nu) * (nv) + (v))
            quad_indices[k, 1] = int(((u) % nu) * (nv) + (v + 1))
            quad_indices[k, 2] = int(((u + 1) % nu) * (nv) + (v + 1))
            quad_indices[k, 3] = int(((u + 1) % nu) * (nv) + (v))
    kwargs["quad_indices"] = quad_indices
    return kwargs
