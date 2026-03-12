import torch
from dataclasses import dataclass

from torch_structure.data import StructData
from config import torch_to_np_float, TORCH_FLOAT


@dataclass
class DesignVariableConfig:
    """
    Data container to initialize a DesignVariable object
    """

    name: str
    attr_name: str
    mask_keyword: str
    is_dual_edge: bool = False
    lower_bound: float | None = None
    upper_bound: float | None = None


class DesignVariable:
    def __init__(
        self, target_obj, array_start: int, dv_config: DesignVariableConfig
    ) -> None:
        """
        Initializes an object that manages design variables contained in a target object and places then in a vector
        """
        # Assign attrivutes from configuration
        self.target_obj = target_obj
        self.name = dv_config.name
        self.attr_name = dv_config.attr_name
        self.is_dual_edge = dv_config.is_dual_edge
        self.start = array_start

        # Attempt mask creation from dictionary of functions
        try:
            self.mask = mask_dicts[dv_config.mask_keyword](target_obj)
        except KeyError as exc:
            raise KeyError(f"Unknown mask_keyword: {dv_config.mask_keyword}") from exc

        # Assign storage indices in vector
        self.n_dv = int(sum(self.mask))
        self.end = self.start + self.n_dv

        # Handle bounds of the design variable
        if dv_config.lower_bound is None:
            dv_config.lower_bound = -torch.inf
        if dv_config.upper_bound is None:
            dv_config.upper_bound = torch.inf
        if dv_config.upper_bound <= dv_config.lower_bound:
            raise ValueError("upper_bound must be greater than lower_bound.")
        self.lower_bound = dv_config.lower_bound
        self.upper_bound = dv_config.upper_bound

    def pull_values(self, values):
        """
        Assign values from attribute in target object to array for storage
        (Get values from source object)
        """
        attr_values_source_all = getattr(self.target_obj, self.attr_name)
        # Apply mask to filter attribute values
        attr_values_source = attr_values_source_all[self.mask]
        attr_values_source = attr_values_source.flatten()
        # Pull values from object (source) to array (target)
        values[self.start : self.end] = attr_values_source

    def push_vaues(self, values: torch.tensor):
        """
        Assign values from array for storage to attribute in target object
        (Apply the values)
        """
        # Clone values from object
        attr_values_target = getattr(self.target_obj, self.attr_name).detach().clone()
        # Set target shape of values to assign
        target_shape = attr_values_target[self.mask].shape

        # Get values from array
        attr_values_source = values[self.start : self.end]
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


class DesignVariableHandler:
    def __init__(
        self, graph: StructData, dv_config_list: list[DesignVariableConfig] = None
    ):
        """
        Initialize object that manages the value of the design variables of an instance of StrucData
        """
        assert dv_config_list is not None, "You must provide design variables"
        # Assign graph as object that contains the design variables
        self.target = graph

        # Initialize design variable sets
        self.design_variable_sets = {}
        # Start tracking the number of design variables
        dv_counter = 0
        for dv_config in dv_config_list:
            # Create design variable set based on input
            dv_set = DesignVariable(graph, dv_counter, dv_config)
            self.design_variable_sets[dv_config.name] = dv_set

            # Update count
            dv_counter = dv_set.end

        # Finalize count
        self.n_dv = dv_counter

    def apply(self, x):
        """
        Pass the values of the design variables to the instance of DataStruc
        """
        for _, design_variable_set in self.design_variable_sets.items():
            # Use method of the self-defined class Design_Variable_Set
            design_variable_set.push_vaues(x)

    def get_values(self):
        """
        Get the values from the isntance of DataStruc to a single vector
        """
        # Initialize numpy array
        x = torch.zeros(self.n_dv, dtype=TORCH_FLOAT)
        for _, design_variable_set in self.design_variable_sets.items():
            # Access initial values and pass their value to the array
            design_variable_set.pull_values(x)
        return x

    def get_values_scipy(self):
        """
        Get the values from the isntance of DataStruc to a single vector
        (interfaces with scipy by converting to numpy arrays)
        """
        return torch_to_np_float(self.get_values())

    def get_bounds(self):
        """
        Get the bounds of all the design variables
        """
        # Initialize numpy array
        lb = torch.zeros(self.n_dv, dtype=TORCH_FLOAT)
        ub = torch.zeros(self.n_dv, dtype=TORCH_FLOAT)
        for _, design_variable_set in self.design_variable_sets.items():
            # Access initial values and pass their value to the array
            _start = design_variable_set.start
            _end = design_variable_set.end
            lb[_start:_end] = design_variable_set.lower_bound
            ub[_start:_end] = design_variable_set.upper_bound
        return lb, ub

    def get_bounds_scipy(self):
        """
        Get the bounds of all the design variables
        (interfaces with scipy by converting to numpy arrays)
        """
        lb, ub = self.get_bounds()
        return torch_to_np_float(lb), torch_to_np_float(ub)


def get_origin_node_mask(graph: StructData):
    """
    Gets the mask for the active dofs of the origin nodes
    """
    return (graph.data.is_origin_node & graph.active_ndof).squeeze()


def get_deviation_element_mask(graph: StructData):
    """
    Gets the mask for the active dofs of the deviation elements (directed edge)
    """
    return (
        (~graph.is_trail_edge & graph.directed_mask & graph.active_edof)
        .detach()
        .clone()
    )


def get_trail_element_mask(graph: StructData):
    """
    Gets the mask for the active dofs of the trail elements (directed edge)
    """
    return (
        (graph.is_trail_edge & graph.directed_mask & graph.active_edof).detach().clone()
    )


def get_dual_mask(graph: StructData, mask):
    """
    Gets the mask of the reciprocal edges to an input edge mask
    """
    dual_edge_id = graph.reciprocal_edge[mask]
    dual_mask = torch.zeros(graph.num_edges, dtype=torch.bool).unsqueeze(1)
    dual_mask[dual_edge_id] = True
    return dual_mask


mask_dicts = {
    "origin_nodes": get_origin_node_mask,
    "trail_elements": get_trail_element_mask,
    "deviation_elements": get_deviation_element_mask,
}
