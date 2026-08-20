import torch

from ..data.data import StructData
from dataclasses import dataclass
from typing import List, Callable

@dataclass
class SolverConfig:
    """
    Data container to specify a solver for datastruc
    """
    solver: str | Callable
    solver_kwargs : dict
@dataclass
class VariableConfig:
    """
    Data container to initialize a DesignVariableGroup object
    """
    name: str
    attr_name: str
    mask: torch.Tensor | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None

class Variable():
    def __init__(self, config : VariableConfig, x_index_start):
        self.attr_name = config.attr_name
        self.attr_mask = config.mask
        self.n_vars = sum(config.mask)
        self.lower_bound = config.lower_bound
        self.upper_bound = config.lower_bound
        self.x_index_start = x_index_start
        self.x_index_end = x_index_start + self.n_vars

    def apply(self, struc_data : StructData, x):
        # Clone values from object
        attr_values_target = getattr(struc_data, self.attr_name).detach().clone()

        # Set target shape of values to assign
        target_shape = attr_values_target[self.attr_mask[0]].shape

        # Get values from array
        attr_values_source = x[self.x_index_start : self.x_index_end]

        # Change to target shape compatible with target object
        attr_values_source = torch.reshape(attr_values_source, target_shape)

        # Assign values from source to target
        for i in range(self.attr_mask.shape[0]):
            attr_values_target[self.attr_mask[i]] = attr_values_source

        # Move values from local copy to target object
        setattr(self.struc_data, self.attr_name, attr_values_target)
    
    def query(self, struc_data : StructData):
        attr_values_source_all = getattr(struc_data, self.attr_name)
        # Apply mask to filter attribute values
        return attr_values_source_all[self.mask].flatten()

class DesignVariableHandler():
    def __init__(
        self,
        struc_data: StructData,
        dv_config_list: List[VariableConfig],
        solver_config: SolverConfig,
    ):
        self.struc_data = struc_data
        self.dv_dict, self.n_vars = self._init_dv_groups(dv_config_list)
        _resolve_solver_config(solver_config)
        self.solver_config = solver_config

    def _init_dv_groups(self, dv_config_list : List[VariableConfig]):
        dv_dict = {}
        n_vars = 0
        for dv_config in dv_config_list:
            dv_config.mask = _resolve_mask(self.struc_data, dv_config)
            dv_group = Variable(dv_config, n_vars)
            dv_dict[dv_config.name] = dv_group
            n_vars += dv_group.n_vars
        return dv_dict, n_vars

    def solve_graph(self, x):
        self.apply(x)
        self.solver_config.solver(self.solver_config.solver_kwargs)

    def query_x(self):
        x = torch.zeros(self.n_vars)
        for dv in self.dv_dict:
            x[dv.x_index_start : dv.x_index_end] = dv.query(self.struc_data)


    def apply(self, x):
        for dv in self.dv_dict:
            dv.apply(self.struc_data, x)

def _resolve_mask(struc_data : StructData, dv_config : VariableConfig):
    return dv_config.mask.unsqueeze(0)
    # TODO: Based on the attribute, resolve mask
    # On edge attributes, apply on the reciprocal

def _resolve_solver_config(struc_data, solver_config : SolverConfig):
    if type(solver_config.solver) == str:
        try:
            solver_config.solver = getattr(struc_data, solver_config.solver)
        except:
            print(f"Solver {solver_config.solver} not found")
    try:
        solver_config.solver(solver_config.solver_kwargs)
    except:
        print(f"Error when running solver")
