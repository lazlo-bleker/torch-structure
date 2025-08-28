import torch
from torch_structure.data import StructData

from params import nu, sw_point, se_point, mw_point, me_point, nw_point, ne_point, s

move = 0.0
k = 1. * torch.pi

def evaluate_top_coords_target():
    # Linearly interpolate southwest point and southeast point
    ts = torch.linspace(0.0, 1.0, nu).unsqueeze(dim=1)
    coords_target = nw_point * (1-ts) + ne_point * ts
    coords_target = coords_target + torch.tensor([0.0, 0.0, 0.2]) * torch.sin(k * ts)
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
    coords_target = coords_target - torch.tensor([0.0, 0.0, 0.2]) * torch.sin(k * ts)
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
    # Pre compute matrix rows and columns
    rows = []
    cols = []
    for e in range(n_elements):
        _quad_indices = quad_indices[e]
        for _quad_index_1 in _quad_indices:
            for _quad_index_2 in _quad_indices:
                rows.append(_quad_index_1.item())
                cols.append(_quad_index_2.item())
    n_nodes = nodes_uv.shape[0]
    rows = torch.tensor(rows)
    cols = torch.tensor(cols)
    flat_index = rows * n_nodes + cols
    unique, inverse = torch.unique(flat_index, return_inverse=True)
    coal_rows = unique // n_nodes
    coal_cols = unique % n_nodes
    coal_indices = torch.stack([coal_rows, coal_cols])
    kwargs["coal_indices"] = coal_indices
    kwargs["unique"] = unique
    kwargs["inverse"] = inverse
    # Pre-compute inner node mask
    u_coords = graph_solved.uv_coords[:,0]
    v_coords = graph_solved.uv_coords[:,1]
    kwargs["east_mask"] = (u_coords == max(u_coords))
    kwargs["west_mask"] = (u_coords == 0) 
    kwargs["south_mask"] = (v_coords == max(v_coords))
    kwargs["north_mask"] = (v_coords == 0)

    return kwargs

def laplacian(graph_solved : StructData, gauss_points, quad_indices, unique, inverse, coal_indices, east_mask, west_mask, north_mask, south_mask):
    nodes_uv = graph_solved.uv_coords
    coords = graph_solved.coords

    dtype = coords.dtype
    n_entries = len(inverse)
    n_nodes = coords.shape[0]
    values = torch.zeros(n_entries, dtype=dtype)
    for k in range(quad_indices.shape[0]):
            node_keys = quad_indices[k]
            _x_ni = coords[node_keys]
            _K_mn_e = torch.zeros([4,4])
            for _gp_w, _gp_xi, _gp_eta in gauss_points:
                _dN_dxi = dN_dxi(_gp_xi, _gp_eta, dtype=dtype)
                _jac_ij = _x_ni.T @ _dN_dxi

                _jac_inv = jac_inv(_jac_ij)
                _B_ni = _dN_dxi @ _jac_inv
                _weight = _gp_w 
                _K_mn_e += _weight * _B_ni @ _B_ni.T

            values[16*k:16*(k+1)] = _K_mn_e.reshape(-1)

    bound_mask = east_mask + west_mask + north_mask + south_mask
    coal_vals = torch.zeros_like(unique, dtype=values.dtype).scatter_add(0, inverse, values)
    L_mn = torch.sparse_coo_tensor(coal_indices, coal_vals, (n_nodes,n_nodes))
    loss_u = L_mn @ nodes_uv[:,0]
    loss_v = L_mn @ nodes_uv[:,1]
    # loss_u = L_mn @ loss_u
    # loss_v = L_mn @ loss_v
    loss_u[bound_mask] = 0.0
    loss_v[bound_mask] = 0.0
    graph_solved.loss_u = torch.abs(loss_u.detach().clone())
    graph_solved.loss_v = torch.abs(loss_v.detach().clone())
    return torch.linalg.norm(loss_v) + torch.linalg.norm(loss_u) 

def dN_dxi(xi, eta, dtype):
    return 0.25 * torch.tensor([
        [-(1-eta),-(1-xi)],
        [+(1-eta),-(1+xi)],
        [+(1+eta),+(1+xi)],
        [-(1+eta),+(1-xi)],
    ], dtype=dtype)

def jac_inv(jac_ij):
    _jac_cov = jac_ij.T @ jac_ij
    _jac_inv_sub = torch.tensor([
        [_jac_cov[1,1], -_jac_cov[1,0]],
        [-_jac_cov[0,1], _jac_cov[0,0]],
    ])
    res = _jac_inv_sub @ jac_ij.T
    # res = 1 / jac_det(jac_ij) * _jac_inv_sub @ jac_ij.T
    return res

def jac_det(jac_ij):
    _jac_cov = jac_ij.T @ jac_ij
    return torch.sqrt(_jac_cov[0,0]*_jac_cov[1,1]-_jac_cov[0,1]*_jac_cov[1,0])
