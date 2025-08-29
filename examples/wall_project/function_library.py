import torch
from torch_structure.data import StructData

from params import nu, sw_point, se_point, mw_point, me_point, nw_point, ne_point, s

move = 0.0
k = 1. * torch.pi

def evaluate_top_coords_target():
    # Linearly interpolate southwest point and southeast point
    ts = torch.linspace(0.0, 1.0, nu).unsqueeze(dim=1)
    coords_target = nw_point * (1-ts) + ne_point * ts
    coords_target = coords_target + torch.tensor([0.0, 0.0, 0.0]) * torch.sin(k * ts)
    return coords_target

def evaluate_center_coords_target():
    # Linearly interpolate southwest point and southeast point
    ts = torch.linspace(0.0, 1.0, nu).unsqueeze(dim=1)
    coords_target = mw_point * (1-ts) + me_point * ts
    coords_target = coords_target - torch.tensor([0.0, 0.0, 0.0]) * torch.sin(k * ts)
    return coords_target

def evaluate_bottom_coords_target():
    # Linearly interpolate southwest point and southeast point
    ts = torch.linspace(0.0, 1.0, nu).unsqueeze(dim=1)
    coords_target = sw_point * (1-ts) + se_point * ts
    coords_target = coords_target - torch.tensor([0.0, 0.0, 0.0]) * torch.sin(k * ts)
    return coords_target

def center_coords_constr_func(graph_solved : StructData, target_coords):
    # Get bottom coords
    v_coords = graph_solved.uv_coords[:,1]
    v_coord = int(max(v_coords)/2)
    center_mask = (v_coords == v_coord)
    center_coords = graph_solved.coords[center_mask]
    center_coords = torch.flatten(center_coords)
    center_coords = torch.reshape(center_coords, [-1,3])
    # Compare current coords with target coords
    diff = diff[:,0]
    return torch.linalg.norm(diff) / (diff.numel() ** 0.5)

def bottom_coords_constr_func(graph_solved : StructData, target_coords):
    # Get bottom coords
    v_coords = graph_solved.uv_coords[:,1]
    bottom_mask = (v_coords == max(v_coords))
    bottom_coords = graph_solved.coords[bottom_mask]
    bottom_coords = torch.flatten(bottom_coords)
    bottom_coords = torch.reshape(bottom_coords, [-1,3])
    # Compare current coords with target coords
    diff = bottom_coords - target_coords
    diff = diff[:,[0,2]]
    return diff.flatten()

def bottom_coords_obj_func(graph_solved : StructData, target_coords):
    # Get bottom coords
    v_coords = graph_solved.uv_coords[:,1]
    bottom_mask = (v_coords == max(v_coords))
    bottom_coords = graph_solved.coords[bottom_mask]
    bottom_coords = torch.flatten(bottom_coords)
    bottom_coords = torch.reshape(bottom_coords, [-1,3])
    # Compare current coords with target coords
    diff = bottom_coords - target_coords
    diff = diff[:,1]
    return torch.linalg.norm(diff) / (diff.numel() ** 0.5)


def evaluate_edge_pairs(graph : StructData):
    nodes_uv = graph.uv_coords
    nu = max(nodes_uv[:,0]) + 1 
    nv = max(nodes_uv[:,1]) + 1

    edge_pairs = []
    for u,v in nodes_uv:
        kp = int((u+1) + v * nu)
        k  = int(u + v * nu)
        km = int((u-1) + v * nu)
        if u == 0:
            deviation_edges = [[k, kp]]
        elif u == (nu-1):
            deviation_edges = [[km,k]]
        else:
            deviation_edges = [[km, k], [k,kp]]

        kp = int(u + (v+1) * nu)
        k  = int(u + v * nu)
        km = int(u + (v-1) * nu)
        if v == 0:
            trail_edges = [[k, kp]]
        elif v == (nv-1):
            trail_edges = [[km,k]]
        else:
            trail_edges = [[km, k], [k,kp]]    

        for trail_edge in trail_edges:
            for deviation_edge in deviation_edges:
                edge_pairs.append([trail_edge, deviation_edge])

    return  torch.tensor(edge_pairs)

def orthogonal_intersections_normed(graph_solved : StructData, edge_pairs : torch.tensor):
    trail_edges = edge_pairs[:,0]
    deviation_edges = edge_pairs[:,1]

    trail_edge_directions = graph_solved.coords[trail_edges[:,1]] - graph_solved.coords[trail_edges[:,0]]
    trail_edge_directions = trail_edge_directions / (torch.linalg.norm(trail_edge_directions, dim=1, keepdim=True))

    deviation_edge_directions = graph_solved.coords[deviation_edges[:,1]] - graph_solved.coords[deviation_edges[:,0]]
    deviation_edge_directions = deviation_edge_directions / (torch.linalg.norm(deviation_edge_directions, dim=1, keepdim=True))

    edge_cos = torch.sum(trail_edge_directions * deviation_edge_directions, dim=1) 
    edge_cos_sq = edge_cos * edge_cos
    edge_cot_sq = edge_cos_sq / (1 - edge_cos_sq)
    return  torch.linalg.norm(edge_cot_sq) / (edge_cot_sq.numel() ** 0.5)

def load_path(graph_solved : StructData):
    edge_load_path = torch.abs(graph_solved.force * graph_solved.length_from_coords)
    return  torch.sum(edge_load_path) / (edge_load_path.numel())


def cache_laplacian(graph_solved : StructData):
    kwargs = {}
    # Pre-compute gauss points for quadrature
    gauss_points = []
    s = (1/3) ** 0.5
    w = 1.0
    gp_1d = [(w,-s),(w,s)]
    for weight_xi, coord_xi in gp_1d:
        for weight_eta, coord_eta in gp_1d:
            weight = weight_xi * weight_eta
            gauss_points.append((weight, coord_xi, coord_eta))
    kwargs["gauss_points"] = gauss_points

    # Pre compute element indices
    nodes_uv = graph_solved.uv_coords
    nu = int(max(nodes_uv[:,0])+1)
    nv = int(max(nodes_uv[:,1])+1)
    n_elements = (nu-1) * (nv-1)
    quad_indices = -torch.ones([n_elements,4], dtype=torch.int)
    for u in range(nu-1):
        for v in range(nv-1):
            k = v * (nu - 1) + u
            quad_indices[k,0] = int((v+1) * (nu) + (u))
            quad_indices[k,1] = int((v+1) * (nu) + (u+1))
            quad_indices[k,2] = int(  (v) * (nu) + (u+1))
            quad_indices[k,3] = int(  (v) * (nu) + (u))
    kwargs["quad_indices"] = quad_indices

    return kwargs

def laplacian(graph_solved: "StructData", gauss_points, quad_indices):
    nodes_uv = graph_solved.uv_coords   # (N,2)
    coords   = graph_solved.coords      # (N,3) surface in R^3

    device = coords.device
    dtype  = coords.dtype

    loss  = torch.zeros((), dtype=dtype, device=device)
    _loss = torch.zeros(coords.shape[0], dtype=dtype, device=device)

    for k in range(quad_indices.shape[0]):
        node_keys = quad_indices[k]
        x_ni = coords[node_keys]          # (4,3)
        u_n  = nodes_uv[node_keys, 0]     # (4,)
        v_n  = nodes_uv[node_keys, 1]     # (4,)

        for gp_w, gp_xi, gp_eta in gauss_points:
            dN = dN_dxi(gp_xi, gp_eta, dtype=dtype, device=device)  # (4,2)

            # 3x2 surface Jacobian J = [x_ξ x_η]
            J = x_ni.T @ dN                         # (3,2)

            # metric G = JᵀJ (2x2), its det and inverse
            G     = J.T @ J                          # (2,2)
            detG  = G[0,0]*G[1,1] - G[0,1]*G[1,0]    # scalar
            Ginv  = inv2x2(G, detG)                  # (2,2)

            # Moore–Penrose on a 3x2 full-rank J: J⁺ = (JᵀJ)⁻¹ Jᵀ
            J_pinv = Ginv @ J.T                      # (2,3)

            # surface area element (|∂x/∂ξ × ∂x/∂η|) = √det(G)
            dA = torch.sqrt(detG)

            # surface gradients of u,v in R^3 (tangent vectors)
            grad_u = (u_n @ dN) @ J_pinv             # (3,)
            grad_v = (v_n @ dN) @ J_pinv             # (3,)

            U = torch.stack([grad_u, grad_v])
            s = torch.linalg.svdvals(U)
            lscm = (s[0]-s[1]) ** 2

            w = torch.as_tensor(gp_w, dtype=dtype, device=device)
            contrib = dA * w * lscm

            loss = loss + contrib
            _loss[node_keys] += contrib

    graph_solved.loss = _loss
    return loss


def dN_dxi(xi, eta, dtype, device):
    # rows: nodes (N1..N4), cols: [∂N/∂ξ, ∂N/∂η]
    return 0.25 * torch.tensor([
        [-(1-eta), -(1-xi)],
        [+(1-eta), -(1+xi)],
        [+(1+eta), +(1+xi)],
        [-(1+eta), +(1-xi)],
    ], dtype=dtype, device=device)


def inv2x2(G, detG):
    # avoid torch.tensor([...]) with tensor entries (breaks grad & device)
    a, b = G[0,0], G[0,1]
    c, d = G[1,0], G[1,1]
    # adj(G) = [[ d, -b], [-c,  a]]
    adj11 =  d
    adj12 = -b
    adj21 = -c
    adj22 =  a
    return torch.stack([
        torch.stack([adj11, adj12]),
        torch.stack([adj21, adj22]),
    ]) / detG