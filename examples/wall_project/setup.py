import numpy as np
import torch
from torch_structure.data import StructData
from scipy.optimize import LinearConstraint, NonlinearConstraint
from utils import (
    get_origin_node_mask,
    get_deviation_element_mask, 
    get_trail_element_mask, 
    access_graph_attr
)
from obj_func import (
    Obj_Function, 
    Bottom_Coords_Constraint,
    Length_Constraint,
    bottom_coords_function, 
    bottom_planar_function, 
    load_path_function, 
    orthogonal_intersections_weights,
    orthogonal_intersections_normed,
    orthogonal_intersections_cd,
    orthogonal_intersections_weights_bottom
)
from params import(
    se_point, 
    sw_point,
    nu,
    omega_orthogonal,
)

def setup_design_ariables(uv_grid_graph : StructData):
    # Initialize data to access design variables
    design_variables_access_dicts = {
        "origin_node_coords" : {
                "attr_name" : "coords",
                "attr_mask" : get_origin_node_mask(uv_grid_graph),
                "is_dual_edge" : False,
                "in_log_scale" : False
            },
        "deviation_element_forces" : {
                "attr_name" : "force",
                "attr_mask" : get_deviation_element_mask(uv_grid_graph), 
                "is_dual_edge" : True,
                "in_log_scale" : False
            },
        "trail_element_lengths" : {
                "attr_name" : "length",
                "attr_mask" : get_trail_element_mask(uv_grid_graph),
                "is_dual_edge" : True,
                "in_log_scale" : True
            },
    }

    # Start tracking the number of design variables
    start_dv = 0
    for _, access_dict in design_variables_access_dicts.items():
        # Assign an interval of the array to store variables
        end_dv = start_dv + sum(access_dict["attr_mask"])
        access_dict["start-end"] = (start_dv, end_dv)
        # Update count of the number of design variables
        start_dv = end_dv

    # Initialize design variables
    design_variables_values = np.zeros(end_dv, dtype=np.float64)
    for _, access_dict in design_variables_access_dicts.items():
        # Access initial values
        init_values = access_graph_attr(uv_grid_graph, access_dict)
        # Assign initial values to array in their assigned interval
        start_dv, end_dv = access_dict["start-end"]
        design_variables_values[start_dv:end_dv] = init_values.detach().numpy().flatten()
    
    return design_variables_access_dicts, design_variables_values

def setup_obj_function_dict(uv_grid_graph : StructData):
    obj_function_dict = {}

    # ORTHOGONAL INTERSECTIONS
    if omega_orthogonal is not None:
        func_dict = {
            "omega" : omega_orthogonal,
        }
        obj_function_dict["OI"]=Obj_Function(orthogonal_intersections_weights, func_dict)

    if omega_orthogonal is not None:
        func_dict = {
            "omega" : 10*omega_orthogonal,
        }
        obj_function_dict["OI_bottom"]=Obj_Function(orthogonal_intersections_weights_bottom, func_dict)

    # # LOAD PATH (experimental)
    # func_dict = {
    #     "omega" : .10,
    # }
    # obj_function_dict["LP"] = Obj_Function(load_path_function, func_dict)
    
    return obj_function_dict

def setup_constraint_list(uv_grid_graph : StructData, attr_access_dicts):
    obj_function_list = []

    # BOTTOM COORDS
    eps = 1e-4
    ts = torch.linspace(0.0, 1.0, nu).unsqueeze(dim=1)
    bottom_coords_target = sw_point * (1-ts) + se_point * ts
    bottom_coords_target = bottom_coords_target.detach().numpy()

    # Bounding box for nodes
    n_nodes = bottom_coords_target.shape[0]
    lb = np.zeros(n_nodes)
    ub = eps * np.ones(n_nodes)

    constraint_obj = Bottom_Coords_Constraint(uv_grid_graph, attr_access_dicts, bottom_coords_target)
    obj_function_list.append(NonlinearConstraint(
        fun=constraint_obj.forward,
        lb=lb.flatten(),
        ub=ub.flatten(),
        jac=constraint_obj.backward,
        ))

    # EDGE LENGTHS
    # Bounds for lengths
    min_length = 5e-2
    max_length = 1e1
    n_edges = len(uv_grid_graph.length)
    lb = np.zeros(n_edges) + min_length
    ub = np.zeros(n_edges) + max_length

    constraint_obj = Length_Constraint(uv_grid_graph, attr_access_dicts)
    obj_function_list.append(NonlinearConstraint(
        fun=constraint_obj.forward,
        lb=lb.flatten(),
        ub=ub.flatten(),
        jac=constraint_obj.backward,
        ))
    
    display_obj_constraints = {
        "support_nodes_target" : bottom_coords_target
    }

    return obj_function_list, display_obj_constraints

def post_process_plot(data_plot, display_obj_constraints):
    target_coords = display_obj_constraints["support_nodes_target"]
    data_plot.scatter(target_coords[:,0], target_coords[:,1], target_coords[:,2], c="#007F00")