import torch
import torch_structure as ts
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from torch_structure.data import StructData
import numpy as np

# ------------------------------
# 1. Create initial bridge setup
# ------------------------------
input_params = {
    "nu": 9, 
    "nv": 7, 
}

# Generate bridge structure
uv_grid_generator = ts.generators.UVGridGenerator(**input_params)
uv_grid_graph = uv_grid_generator()
data = uv_grid_graph.mpcem(max_iter=1000, verbose=True)
# data.plot(title="Initial Wall", legend=False)
# plt.show()

# --------------------------------
# 2. Define optimization variables
# --------------------------------
def get_origin_node_mask(graph):
    return graph.data.is_origin_node.squeeze()
def get_deviation_element_mask(graph):
    return  (~graph.is_trail_edge & graph.directed_mask & graph.active_dof).detach().clone()
def get_trail_element_mask(graph):
    return (graph.is_trail_edge & graph.directed_mask & graph.active_dof).detach().clone()

def access_graph_attr(graph : StructData, access_dict):
    attr = getattr(graph, access_dict["attr_name"])
    return attr[access_dict["attr_mask"]]

def get_dual_mask(graph, mask):        
    dual_edge_id = graph.reciprocal_edge[mask]
    dual_mask = torch.zeros(graph.num_edges, dtype=torch.bool).unsqueeze(1)
    dual_mask[dual_edge_id] = True
    return dual_mask

def set_graph_attr(graph : StructData, access_dict, values):
    attr = getattr(graph, access_dict["attr_name"]).clone()
    target_shape = attr[access_dict["attr_mask"]].shape
    values = torch.reshape(values, target_shape)
    attr[access_dict["attr_mask"]] = values
    if access_dict["is_dual_edge"]:
        dual_mask = get_dual_mask(graph, access_dict["attr_mask"])
        attr[dual_mask] = values
    
    setattr(graph, access_dict["attr_name"], attr)

# Initialize data to access design variables
design_variables_access_dicts = {
    # # Disabled, not of interest currently
    # "origin_node_coords" : {
    #         "attr_name" : "coords",
    #         "attr_mask" : get_origin_node_mask(uv_grid_graph),
    #         "is_dual_edge" : False,
    #     },
    "deviation_element_forces" : {
            "attr_name" : "force",
            "attr_mask" : get_deviation_element_mask(uv_grid_graph), 
            "is_dual_edge" : True,
        },
    "trail_element_lengths" : {
            "attr_name" : "length",
            "attr_mask" : get_trail_element_mask(uv_grid_graph),
            "is_dual_edge" : True,
        },
}

# Initialize design variables
n_dofs = 0
for key, access_dict in design_variables_access_dicts.items():
    init_values = access_graph_attr(uv_grid_graph, access_dict)
    n_dofs += len(init_values.flatten())

design_variables_values = np.zeros(n_dofs, dtype=float)
start_dof = 0
for key, access_dict in design_variables_access_dicts.items():
    init_values = access_graph_attr(uv_grid_graph, access_dict)
    end_dof = start_dof + len(init_values.flatten())
    access_dict["start-end"] = (start_dof, end_dof)
    design_variables_values[start_dof:end_dof] = init_values.detach().numpy().flatten()
    start_dof = end_dof


# -------------------------------
# 3. Define optimization function
# -------------------------------
def apply_design_variables(value_attrs, graph, attr_access_dicts):
    for _, attr_access_dict in attr_access_dicts.items():
        start_dof, end_dof = attr_access_dict["start-end"]
        value_attr = value_attrs[start_dof:end_dof]
        set_graph_attr(graph, attr_access_dict, value_attr)
    
@ts.utils.scipy_jacobian  # Decorator to make torch function compatible with scipy
def obj_function(value_attrs, graph, attr_access_dicts, obj_function_list):
    apply_design_variables(value_attrs, graph, attr_access_dicts)
    
    data = graph.mpcem(max_iter=1000, damping_factor=0.5)

    loss = 0
    for obj_function in obj_function_list:
        loss += obj_function(data)
    return loss

# Define callback for logging progress (optional)
def make_callback():
    def callback(x):
        print(
            f"Iteration {callback.iteration:3d} | Loss: {obj_function.best_loss:.6f}"
        )
        callback.iteration += 1
    callback.iteration = 0
    return callback


# -------------------
# 4. Run optimization
# -------------------
from obj_func import Obj_Function, bottom_coords_function, bottom_planar_function, load_path_function, orthogonal_intersections

obj_function_list = []

# # BOTTOM PLANAR
# func_dict = {
#     "omega" : 1.0,
#     "coord_z" : 0.0
# }
# obj_function_list.append(Obj_Function(bottom_planar_function, func_dict))


# ORTHOGONAL INTERSECTIONS
func_dict = {
    "omega" : 1.0,
}
obj_function_list.append(Obj_Function(orthogonal_intersections, func_dict))


# BOTTOM COORDS
s = 2.0
v_coords = uv_grid_graph.uv_coords[:,1]
top_mask = (v_coords == 0)
top_coords = uv_grid_graph.coords[top_mask]
func_dict = {
    "omega" : 1.0,
    "bottom_coords_target" : s * (top_coords +  torch.tensor([-0.5, 0.0, -1.0])) + s / 2 * torch.tensor([0.5, 0.0, 0.0])
}
obj_function_list.append(Obj_Function(bottom_coords_function, func_dict))

# # LOAD PATH (experimental)
# func_dict = {
#     "omega" : 0.001,
# }
# obj_function_list.append(Obj_Function(load_path_function, func_dict))

    
# test = obj_function(design_variables_values, uv_grid_graph, design_variables_access_dicts, obj_function_list)
# print(test)

result = minimize(
    fun=obj_function,
    args=(uv_grid_graph, design_variables_access_dicts, obj_function_list),
    x0=design_variables_values,
    method="SLSQP",
    jac=True,
    callback=make_callback(),
    options={"disp": True, "ftol": 1e-7, "gtol": 1e-7, "maxiter": 200},
)


# ---------------------------------------
# 5. Apply optimized forces and visualize
# ---------------------------------------
apply_design_variables(torch.tensor(result.x, dtype=torch.float), uv_grid_graph, design_variables_access_dicts)
# Final run to compute the equilibrium structure with optimized forces
data = uv_grid_graph.mpcem(max_iter=1000, verbose=True)

# Plot the optimized structure
data.plot(title="Optimized Wall", legend=False)
plt.show()

print("f")