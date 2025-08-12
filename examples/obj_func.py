from torch_structure.data import StructData
import torch

class Obj_Function():
    def __init__(self, function, kwargs={}):
        self.function = function
        self.kwargs = kwargs
    
    def __call__(self, graph):
        return self.function(graph, **self.kwargs)

def orthogonal_intersections(graph, omega):
    nodes_uv = graph.uv_coords
    nu = max(nodes_uv[:,0]) + 1 
    nv = max(nodes_uv[:,1]) + 1
    loss = 0.0
    for u,v in nodes_uv:
        if u == 0:
            up, um = (1,0)
        elif u == (nu-1):
            up, um = (nu-1,nu-2)
        else:
            up, um = (u+1,u-1)

        kp = int(up + v * nu)
        km = int(um + v * nu)
        deviations_tangent = graph.coords[kp] - graph.coords[km]
        
        if v == 0:
            vp, vm = (1,0)
        elif v == (nv-1):
            vp, vm = (nv-1,nv-2)
        else:
            vp, vm = (v+1,v-1)

        kp = int(u + vp * nu)
        km = int(u + vm * nu)

        trails_tangent = graph.coords[kp] - graph.coords[km]     

        inner_td = torch.dot(trails_tangent, deviations_tangent) 
        loss += inner_td ** 2

    return omega * loss

def bottom_planar_function(graph, omega, coord_z):
    v_coords = graph.uv_coords[:,1]
    bottom_mask = (v_coords == max(v_coords))
    bottom_coords = graph.coords[bottom_mask]
    diff = bottom_coords[:, 2] - coord_z
    return omega * torch.sum(diff ** 2)

def bottom_coords_function(graph, omega, bottom_coords_target):
    v_coords = graph.uv_coords[:,1]
    bottom_mask = (v_coords == max(v_coords))
    bottom_coords = graph.coords[bottom_mask]
    diff = bottom_coords-bottom_coords_target
    return omega * torch.sum(diff ** 2)

def load_path_function(graph, omega):
    fd = graph.length_from_coords * graph.force * graph.force_sign
    return omega * torch.sum(fd)