from torch_structure.data import StructData
import torch
import numpy as np
import torch_structure as ts
from utils import (
    apply_design_variables,
    get_deviation_element_mask
    )
from params import max_iters_cem

@ts.utils.scipy_jacobian  # Decorator to make torch function compatible with scipy
def obj_function(value_attrs, graph, attr_access_dicts, obj_function_dict):
    # Update value of design variables
    apply_design_variables(graph, value_attrs, attr_access_dicts)
    # Solve CEM
    data = graph.mpcem(max_iter=max_iters_cem)
    # Compute total loss based on CEM solution
    total_loss = 0
    for _, obj_function in obj_function_dict.items():
        # Add contribution of obj function from list
        total_loss += obj_function(data)
    return total_loss

# Define callback for logging progress (optional)
def make_callback():
    def callback(x):
        print(
            f"Iteration {callback.iteration:3d} | Loss: {obj_function.loss:.6f}"
        )
        callback.iteration += 1
    callback.iteration = 0
    return callback

class Obj_Function():
    def __init__(self, function, kwargs={}):
        self.function = function
        self.kwargs = kwargs
    
    def __call__(self, graph):
        return self.function(graph, **self.kwargs)

def orthogonal_intersections_cd(graph, omega):
    nodes_uv = graph.uv_coords
    nu = max(nodes_uv[:,0]) + 1 
    nv = max(nodes_uv[:,1]) + 1
    loss = 0.0
    for u,v in nodes_uv:
        if u == 0:
            up, um = (u+1,u)
        elif u == (nu-1):
            up, um = (u,u-1)
        else:
            up, um = (u+1,u-1)

        kp = int(up + v * nu)
        km = int(um + v * nu)
        deviations_tangent = graph.coords[kp] - graph.coords[km]
        
        if v == 0:
            vp, vm = (v+1,v)
        elif v == (nv-1):
            vp, vm = (v,v-1)
        else:
            vp, vm = (v+1,v-1)

        kp = int(u + vp * nu)
        km = int(u + vm * nu)

        trails_tangent = graph.coords[kp] - graph.coords[km]     

        inner_td = torch.dot(trails_tangent, deviations_tangent) 
        loss += inner_td ** 2 / (1-inner_td ** 2)

    return omega * loss

def orthogonal_intersections_weights_bottom(graph, omega):
    nodes_uv = graph.uv_coords
    nu = max(nodes_uv[:,0]) + 1 
    nv = max(nodes_uv[:,1]) + 1
    loss = 0.0
    v=int(nv)-1
    for u in range(int(nu)):
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
            trail_edge_dir = graph.coords[trail_edge[1]] - graph.coords[trail_edge[0]]
            for deviation_edge in deviation_edges:
                deviation_edge_dir = graph.coords[deviation_edge[1]] - graph.coords[deviation_edge[0]]
                inner_td = torch.dot(trail_edge_dir, deviation_edge_dir) 
                loss += inner_td ** 2 / (1-inner_td ** 2)

    return omega * loss

    return omega * loss


def orthogonal_intersections_weights(graph, omega):
    nodes_uv = graph.uv_coords
    nu = max(nodes_uv[:,0]) + 1 
    nv = max(nodes_uv[:,1]) + 1
    loss = 0.0
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
            trail_edge_dir = graph.coords[trail_edge[1]] - graph.coords[trail_edge[0]]
            for deviation_edge in deviation_edges:
                deviation_edge_dir = graph.coords[deviation_edge[1]] - graph.coords[deviation_edge[0]]
                inner_td = torch.dot(trail_edge_dir, deviation_edge_dir) 
                loss += inner_td ** 2 / (1-inner_td ** 2)

    return omega * loss

def orthogonal_intersections_normed(graph, omega):
    nodes_uv = graph.uv_coords
    nu = max(nodes_uv[:,0]) + 1 
    nv = max(nodes_uv[:,1]) + 1
    loss = 0.0
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
            trail_edge_dir = graph.coords[trail_edge[1]] - graph.coords[trail_edge[0]]
            trail_edge_dir = trail_edge_dir / (torch.linalg.norm(trail_edge_dir))
            for deviation_edge in deviation_edges:
                deviation_edge_dir = graph.coords[deviation_edge[1]] - graph.coords[deviation_edge[0]]
                deviation_edge_dir = deviation_edge_dir / (torch.linalg.norm(deviation_edge_dir))
                inner_td = torch.dot(trail_edge_dir, deviation_edge_dir) 
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
    diff = bottom_coords - bottom_coords_target
    return omega * torch.sum(torch.abs(diff))

def load_path_function(graph, omega):
    fd = graph.length_from_coords * graph.force
    mask = ~graph.is_trail_edge
    fd = fd[mask]
    return omega * torch.sum(fd ** 2)

class Bottom_Coords_Constraint():
    def __init__(self, graph, attr_access_dicts, bottom_coords_target):
        self.graph = graph
        self.attr_access_dicts = attr_access_dicts
        self.bottom_coords_target = torch.tensor(bottom_coords_target, dtype=torch.float64)

    def _bottom_coords(self, x):
        # Apply values of x
        apply_design_variables(self.graph, x, self.attr_access_dicts)
        # Solve CEM
        data = self.graph.mpcem(max_iter=max_iters_cem)
        # Get bottom coords
        v_coords = data.uv_coords[:,1]
        bottom_mask = (v_coords == max(v_coords))
        bottom_coords = data.coords[bottom_mask]
        bottom_coords = torch.flatten(bottom_coords)
        return bottom_coords
    
    def _coords_dist(self, x):
        bottom_coords = self._bottom_coords(x)
        bottom_coords = torch.reshape(bottom_coords, [-1,3])
        diff = bottom_coords - self.bottom_coords_target
        return torch.sum(diff * diff, dim=1)

    def forward(self, x_np):
        x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        f = lambda z: self._coords_dist(z)

        y = f(x)
        J = torch.func.jacrev(f)(x)
        self.J = J.detach().cpu().numpy()
        return y.detach().cpu().numpy()

    def backward(self, x_np):
        return self.J
    
class Length_Constraint():
    def __init__(self, graph, attr_access_dicts):
        self.graph = graph
        self.attr_access_dicts = attr_access_dicts

    def _lengths(self, x):
        # Apply values of x
        apply_design_variables(self.graph, x, self.attr_access_dicts)
        # Solve CEM
        data = self.graph.mpcem(max_iter=max_iters_cem)
        # Get bottom coords
        lengths = torch.flatten(data.length_from_coords)
        return lengths

    def forward(self, x_np):
        x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        f = lambda z: self._lengths(z)

        y = f(x)
        J = torch.func.jacrev(f)(x)
        self.J = J.detach().cpu().numpy()
        return y.detach().cpu().numpy()

    def backward(self, x_np):
        return self.J
    