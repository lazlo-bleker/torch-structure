import numpy as np
import torch

from torch_structure.data import StructData
from .config import float_np, float_torch

class Design_Variable_Set():
    def __init__(self, 
                 target_obj, 
                 attr_name : str, 
                 attr_mask : np.ndarray, 
                 array_start : int, 
                 array_end : int, 
                 is_dual_edge = False, 
                 in_log_scale = False,
                ):
        """
        Creates an object that acts as an interface between the design variable vector of a scipy optimizer and an instance of StrucData
        """
        # StrucData object
        self.target_obj = target_obj
        # Name of attribute in StrucData object
        self.attr_name = attr_name
        # Mask of attribute in StrucData object
        self.mask = attr_mask
        # Start index of design variable for set
        self.start = array_start
        # End index of design variable for set
        self.end = array_end
        # Flag to handle dual edges (their value must match)
        self.is_dual_edge = is_dual_edge
        # Flag to handle design variables in logarithmic scale (prevents from sign flipping)
        self.in_log_scale = in_log_scale
        
    
    def pull_values(self, values):
        """
        Assign values from attribute in target object to array for storage
        # Get values from source
        """
        attr_values_source_all = getattr(self.target_obj, self.attr_name)
        # Apply mask to filter attribute values
        attr_values_source = attr_values_source_all[self.mask]
        attr_values_source = attr_values_source.flatten()
        if self.in_log_scale:
            # Encode to log scale
            attr_values_source = torch.log(torch.abs(attr_values_source))
        # Pull values from object (source) to array (target)
        values[self.start:self.end] = attr_values_source

    def push_vaues(self, values : torch.tensor):
        """
        Assign values from array for storage to attribute in target object
        """
        # Clone values from object
        attr_values_target =  getattr(self.target_obj, self.attr_name).detach().clone()
        # Set target shape of values to assign
        target_shape = attr_values_target[self.mask].shape

        # Get values from array
        attr_values_source = values[self.start:self.end]
        if self.in_log_scale:
            # Decode from log scale
            attr_values_source = torch.exp(attr_values_source)
        # Change to target shape compatible with target object
        attr_values_source = torch.reshape(attr_values_source, target_shape)

        # Assign values from source to target
        attr_values_target[self.mask] = attr_values_source
        if self.is_dual_edge:
            # Apply values to dual edges
            dual_mask = get_dual_mask(self.target_obj, self.mask)
            attr_values_target[dual_mask] = attr_values_source

        # Move values from local copy to target object
        setattr(self.target_obj, self.attr_name, attr_values_target)


class Design_Variable_Handler():
    def __init__(self, 
                 graph : StructData,
                 dv_dicts : dict = None
                 ):
        """
        Initialize object that manages the value of the design variables of an instance of StrucData
        """
        # Assign graph as object that contains the design variables
        self.target = graph

        # Initialize design variable sets
        self.design_variable_sets = {}
        # Start tracking the number of design variables
        dv_start = 0
        for dv_name, dv_args in dv_dicts.items():
            # Unpack args
            attr_name = dv_args[0]
            attr_mask = mask_dicts[dv_args[1]](graph)
            is_dual_edge = dv_args[2]
            in_log_scale = dv_args[3]
            # Assign an interval of the array to store variables
            dv_end = dv_start + sum(attr_mask)

            # Create design variable set based on input
            self.design_variable_sets[dv_name] = Design_Variable_Set(
                graph,
                attr_name,
                attr_mask,
                dv_start,
                dv_end,
                is_dual_edge,
                in_log_scale,
            )
            # Update count
            dv_start = dv_end

        # Finalize count
        self.n_dv = dv_end
    
    def apply(self, x):
        """
        Pass the values of the design variables to the instance of DataStruc
        """
        for _, design_variable_set in self.design_variable_sets.items():
            # Use method of the self-defined class Design_Variable_Set
            design_variable_set.push_vaues(x)
    
    def get_values(self):
        """
        Get the values from the isntance of DataStruc to a numpy array
        """
        # Initialize numpy array
        x = torch.zeros(self.n_dv, dtype=float_torch)
        for _, design_variable_set in self.design_variable_sets.items():
            # Access initial values and pass their value to the array
            design_variable_set.pull_values(x)
        return x.detach().cpu().numpy()

def get_origin_node_mask(graph : StructData):
    """
    Gets the mask for the active dofs of the origin nodes
    """
    return (graph.data.is_origin_node & graph.active_ndof).squeeze()

def get_deviation_element_mask(graph : StructData):
    """
    Gets the mask for the active dofs of the deviation elements (directed edge)
    """
    return  (~graph.is_trail_edge & graph.directed_mask & graph.active_edof).detach().clone()

def get_trail_element_mask(graph : StructData):
    """
    Gets the mask for the active dofs of the trail elements (directed edge)
    """
    return (graph.is_trail_edge & graph.directed_mask & graph.active_edof).detach().clone()

def get_dual_mask(graph : StructData, mask):      
    """
    Gets the mask of the reciprocal edges to an input edge mask
    """  
    dual_edge_id = graph.reciprocal_edge[mask]
    dual_mask = torch.zeros(graph.num_edges, dtype=torch.bool).unsqueeze(1)
    dual_mask[dual_edge_id] = True
    return dual_mask

mask_dicts = {
    "origin_nodes" : get_origin_node_mask,
    "trail_elements" : get_trail_element_mask,
    "deviation_elements" : get_deviation_element_mask
}