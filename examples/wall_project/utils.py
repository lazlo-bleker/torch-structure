from torch_structure.data import StructData
import torch

def apply_design_variables(graph : StructData, value_attrs, attr_access_dicts):
    for _, attr_access_dict in attr_access_dicts.items():
        start_dof, end_dof = attr_access_dict["start-end"]
        value_attr = value_attrs[start_dof:end_dof]
        set_graph_attr(graph, attr_access_dict, value_attr)

def access_graph_attr(graph : StructData, access_dict):
    attr_values = getattr(graph, access_dict["attr_name"])
    attr_values = attr_values[access_dict["attr_mask"]]
    if access_dict["in_log_scale"]:
        # Encode to log scale
        attr_values = torch.log(torch.abs(attr_values))
    return attr_values

def set_graph_attr(graph : StructData, access_dict, set_values):
    attr_values = getattr(graph, access_dict["attr_name"]).clone()
    target_shape = attr_values[access_dict["attr_mask"]].shape

    if access_dict["in_log_scale"]:
        # Decode log scale
        set_values = torch.exp(set_values)
        
    set_values = torch.reshape(set_values, target_shape)
    attr_values[access_dict["attr_mask"]] = set_values

    if access_dict["is_dual_edge"]:
        dual_mask = get_dual_mask(graph, access_dict["attr_mask"])
        attr_values[dual_mask] = set_values
    
    setattr(graph, access_dict["attr_name"], attr_values)

def get_origin_node_mask(graph):
    return (graph.data.is_origin_node & graph.active_ndof).squeeze()

def get_deviation_element_mask(graph):
    return  (~graph.is_trail_edge & graph.directed_mask & graph.active_edof).detach().clone()

def get_trail_element_mask(graph):
    return (graph.is_trail_edge & graph.directed_mask & graph.active_edof).detach().clone()

def get_dual_mask(graph, mask):        
    dual_edge_id = graph.reciprocal_edge[mask]
    dual_mask = torch.zeros(graph.num_edges, dtype=torch.bool).unsqueeze(1)
    dual_mask[dual_edge_id] = True
    return dual_mask