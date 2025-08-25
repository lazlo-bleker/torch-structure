import torch
from torch_structure.data import StructData

from params import nu, sw_point, se_point, mw_point, me_point, nw_point, ne_point, s

move = 0.
k = 2 * torch.pi

def evaluate_top_coords_target():
    # Linearly interpolate southwest point and southeast point
    ts = torch.linspace(0.0, 1.0, nu).unsqueeze(dim=1)
    coords_target = nw_point * (1-ts) + ne_point * ts
    coords_target = coords_target - torch.tensor([0.0, move, 0.0]) * torch.cos(k * ts)
    return coords_target

def evaluate_center_coords_target():
    # Linearly interpolate southwest point and southeast point
    ts = torch.linspace(0.0, 1.0, nu).unsqueeze(dim=1)
    coords_target = mw_point * (1-ts) + me_point * ts
    coords_target = coords_target - torch.tensor([0.0, 0.0, 0.0]) * torch.cos(k * ts)
    return coords_target

def evaluate_bottom_coords_target():
    # Linearly interpolate southwest point and southeast point
    ts = torch.linspace(0.0, 1.0, nu).unsqueeze(dim=1)
    coords_target = sw_point * (1-ts) + se_point * ts
    coords_target = coords_target - torch.tensor([0.0, 4*move, 0.0]) * torch.cos(k * ts)
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