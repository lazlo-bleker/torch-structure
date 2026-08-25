import torch

from ..data.data import StructData
from dataclasses import dataclass, field
from typing import List, Callable, Any


@dataclass
class SolverConfig:
    """
    Data container to specify a solver for datastruc
    """

    solver: str | Callable
    solver_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class VariableConfig:
    """
    Data container to initialize a DesignVariableGroup object
    """

    name: str
    attr_name: str
    mask: torch.Tensor | None = None
    lower_bound: float | None = -torch.inf
    upper_bound: float | None = torch.inf


class Variable:
    def __init__(self, config: VariableConfig, x_index_start):
        self.attr_name = config.attr_name
        self.attr_masks = config.resolved_masks
        self.n_vars = torch.sum(config.resolved_masks[0])
        self.lower_bound = config.lower_bound
        self.upper_bound = config.upper_bound
        self.x_index_start = x_index_start
        self.x_index_end = x_index_start + self.n_vars

    def apply(self, struc_data: StructData, x):
        # Clone values from object
        attr_values_target = getattr(struc_data, self.attr_name).detach().clone()
        # Set target shape of values to assign
        target_shape = attr_values_target[self.attr_masks[0]].shape
        # Get values from array
        attr_values_source = x[self.x_index_start : self.x_index_end]
        # Change to target shape compatible with target object
        attr_values_source = torch.reshape(attr_values_source, target_shape)
        # Assign values from source to target
        for i in range(self.attr_masks.shape[0]):
            attr_values_target[self.attr_masks[i]] = attr_values_source
        # Move values from local copy to target object
        setattr(struc_data, self.attr_name, attr_values_target)

    def query(self, struc_data: StructData):
        attr_values_source_all = getattr(struc_data, self.attr_name)
        # Apply mask to filter attribute values
        return attr_values_source_all[self.attr_masks[0]].flatten()


class DesignVariableHandler:
    def __init__(
        self,
        struc_data: StructData,
        dv_config_list: List[VariableConfig],
        solver_config: SolverConfig,
    ):
        self.struc_data = struc_data
        self.dv_dict, self.n_vars = self._init_dv_groups(dv_config_list)
        _resolve_solver_config(self.struc_data, solver_config)
        # solver_config.solver_kwargs["inplace"] = False
        self.solver_config = solver_config

    def _init_dv_groups(self, dv_config_list: List[VariableConfig]):
        dv_dict = {}
        n_vars = 0
        for dv_config in dv_config_list:
            _resolve_mask(self.struc_data, dv_config)
            dv_group = Variable(dv_config, n_vars)
            dv_dict[dv_config.name] = dv_group
            n_vars += dv_group.n_vars
        return dv_dict, n_vars

    def solve_graph(self, x):
        self.apply(x)
        return self.solver_config.solver(self.solver_config.solver_kwargs)

    @property
    def x(self):
        x = torch.zeros(self.n_vars)
        for _, dv in self.dv_dict.items():
            x[dv.x_index_start : dv.x_index_end] = dv.query(self.struc_data)
        return x

    @property
    def log(self):
        return {"values": self.x}

    @property
    def bounds(self):
        bounds = torch.zeros([self.n_vars, 2])
        for _, dv in self.dv_dict.items():
            bounds[dv.x_index_start : dv.x_index_end, 0] = dv.lower_bound
            bounds[dv.x_index_start : dv.x_index_end, 1] = dv.upper_bound
        return bounds

    def apply(self, x):
        for _, dv in self.dv_dict.items():
            dv.apply(self.struc_data, x)


def _resolve_mask(struc_data: StructData, dv_config: VariableConfig):
    # NOTE: Saved metadata may be discontinued
    if dv_config.attr_name in struc_data.metadata["edge_attr_list"]:
        dv_config.resolved_masks = torch.stack(
            [dv_config.mask, _get_dual_mask(struc_data, dv_config.mask)]
        )

    else:
        dv_config.resolved_masks = dv_config.mask.unsqueeze(0)


def _get_dual_mask(graph: StructData, mask):
    """
    Gets the mask of the reciprocal edges to an input edge mask
    """
    dual_edge_id = graph.reciprocal_edge[mask]
    dual_mask = torch.zeros(graph.num_edges, dtype=torch.bool).unsqueeze(1)
    dual_mask[dual_edge_id] = True
    return dual_mask


def _resolve_solver_config(struc_data, solver_config: SolverConfig):
    if type(solver_config.solver) == str:
        try:
            solver_config.solver = getattr(struc_data, solver_config.solver)
        except:
            print(f"Solver {solver_config.solver} not found")
    try:
        solver_config.solver(solver_config.solver_kwargs)
    except:
        print(f"Error when running solver")
