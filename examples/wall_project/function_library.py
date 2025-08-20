import torch
from torch_structure.data import StructData

from params import nu, sw_point, se_point

def evaluate_bottom_coords_target():
    # Linearly interpolate southwest point and southeast point
    ts = torch.linspace(0.0, 1.0, nu).unsqueeze(dim=1)
    bottom_coords_target = sw_point * (1-ts) + se_point * ts
    return bottom_coords_target

def bottom_coords_constr_func(graph_solved : StructData, target_coords):
    # Get bottom coords
    v_coords = graph_solved.uv_coords[:,1]
    bottom_mask = (v_coords == max(v_coords))
    bottom_coords = graph_solved.coords[bottom_mask]
    bottom_coords = torch.flatten(bottom_coords)
    bottom_coords = torch.reshape(bottom_coords, [-1,3])
    # Compare current coords with target coords
    diff = bottom_coords - target_coords
    return diff.flatten()

def orthogonal_intersections_normed(graph_solved : StructData):
    nodes_uv = graph_solved.uv_coords
    nu = max(nodes_uv[:,0]) + 1 
    nv = max(nodes_uv[:,1]) + 1
    loss = 0.0
    counter = 0
    for u,v in nodes_uv:
        kp = int((u+1) + v * nu)
        k  = int(u + v * nu)
        km = int((u-1) + v * nu)
        if u == 0:
            deviation_edges = [(k, kp)]
        elif u == (nu-1):
            deviation_edges = [(km,k)]
        else:
            deviation_edges = [(km, k), (k,kp)]

        kp = int(u + (v+1) * nu)
        k  = int(u + v * nu)
        km = int(u + (v-1) * nu)
        if v == 0:
            trail_edges = [(k, kp)]
        elif v == (nv-1):
            trail_edges = [(km,k)]
        else:
            trail_edges = [(km, k), (k,kp)]    

        for trail_edge in trail_edges:
            trail_edge_dir = graph_solved.coords[trail_edge[1]] - graph_solved.coords[trail_edge[0]]
            trail_edge_dir = trail_edge_dir / (torch.linalg.norm(trail_edge_dir))
            for deviation_edge in deviation_edges:
                deviation_edge_dir = graph_solved.coords[deviation_edge[1]] - graph_solved.coords[deviation_edge[0]]
                deviation_edge_dir = deviation_edge_dir / (torch.linalg.norm(deviation_edge_dir))
                inner_td = torch.dot(trail_edge_dir, deviation_edge_dir) 
                loss += inner_td ** 2
                counter += 1

    return  loss / counter