import torch
from torch_structure.data import StructData

from params import nu, sw_point, se_point, mw_point, me_point, nw_point, ne_point, s

move = 0.0
k = 1. * torch.pi

def evaluate_top_coords_target():
    # Linearly interpolate southwest point and southeast point
    ts = torch.linspace(0.0, 1.0, nu).unsqueeze(dim=1)
    coords_target = nw_point * (1-ts) + ne_point * ts
    coords_target = coords_target - torch.tensor([0.0, 0.0, move]) * torch.sin(k * ts)
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

def fe_orthogonal(graph_solved : StructData):
    nodes_uv = graph_solved.uv_coords
    coords = graph_solved.coords
    nu = int(max(nodes_uv[:,0])+1)
    nv = int(max(nodes_uv[:,1])+1)

    gp_weights = []
    gp_xi = []
    gp_eta = []
    # _gp_x = [0.0]
    # _gp_w = [2.0]
    s = (1/3) ** 0.5
    w = 1.0
    _gp_x = [-s,  s]
    _gp_w = [w, w]
    for i in range(len(_gp_x)):
        for j in range(len(_gp_x)):
            gp_weights.append(_gp_w[i]*_gp_w[j])
            gp_xi.append(_gp_x[i])
            gp_eta.append(_gp_x[j])
    n_gp = len(gp_weights)

    dtype = coords.dtype
    loss = 0.0
    for u in range(nu-1):
        for v in range(nv-1):
            node_keys = []
            node_keys.append(int((u) + (v) * nu))
            node_keys.append(int((u+1) + (v) * nu))
            node_keys.append(int((u+1) + (v+1) * nu))
            node_keys.append(int((u) + (v+1) * nu))

            _x_ni = coords[node_keys]
            _u_n = nodes_uv[node_keys,0]
            _v_n = nodes_uv[node_keys,1]
            for i_gp in range(n_gp):
                _gp_w = gp_weights[i_gp]
                _gp_xi = gp_xi[i_gp]
                _gp_eta = gp_eta[i_gp]

                _dN_dxi = dN_dxi(_gp_xi, _gp_eta, dtype=dtype)
                _jac_ij = _x_ni.T @ _dN_dxi


                _jac_inv = jac_inv(_jac_ij)
                # _jac_det = jac_det(_jac_ij)
                _grad_u = _u_n @ _dN_dxi @ _jac_inv
                _grad_v = _v_n @ _dN_dxi @ _jac_inv

                # weight = _gp_w * _jac_det 
                _weight = _gp_w 
                # _cos_sq = (torch.sum(_grad_v * _grad_u) ** 2)
                # _cot_sq = _cos_sq / (1 - _cos_sq)
                # loss += _weight * _cot_sq
                loss += _weight * torch.sum(_grad_v * _grad_v)
                loss += _weight * torch.sum(_grad_u * _grad_u)

    return loss

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
