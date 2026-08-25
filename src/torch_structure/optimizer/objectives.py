from ..data.data import StructData
import torch
from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class ObjectiveConfig:
    """
    Data container to initialize a Constraint object
    """

    name: str
    obj_function: Callable
    weight: float = 1.0
    kwargs: dict[str, Any] = field(default_factory=dict)


class Objective:
    def __init__(self, objective_config: ObjectiveConfig):
        """
        Initializes an objective for an optimization problem
        """
        # NOTE: Having a function as an object allows to cache variables (a function can have a sate, store in kwargs)
        self.obj_function = objective_config.obj_function
        self.weight = objective_config.weight
        self.kwargs = dict(objective_config.kwargs)
        self.state = {}

    def forward(self, struc_data: StructData):
        """
        Calls a function that acts on a solved graph resulting from StrucData.cem and optional stored variables
        """
        # Evaluate function
        value = self.obj_function(struc_data, **self.kwargs)
        # Log state
        self.state["value"] = value
        return value

    @property
    def log(self):
        """
        Returns log data as a dictionary
        """
        return self.state


class ObjectiveHandler:
    def __init__(self, solve_graph, obj_func_config_list: list[ObjectiveConfig]):
        """
        Initializes an instance of a class that manages the objectives in an optimization problem
        """
        # Link method to apply the value of the design variables and solve the system
        self.solve_graph = solve_graph
        # Init collection of Objective instances
        self.objectives_dict = {}
        # Add objective functions
        for obj_config in obj_func_config_list:
            # Add object to dictionary
            self.objectives_dict[obj_config.name] = Objective(obj_config)

    def forward(self, x):
        """
        Evaluates the objective function
        """
        # Apply design variables & solve CEM
        graph_solved = self.solve_graph(x)
        # Compute total loss based on solution
        total_loss = torch.tensor(0.0)
        for _, objective in self.objectives_dict.items():
            # Add contribution of obj function from list
            total_loss += objective.weight * objective.forward(graph_solved)
        # Save total loss too
        return total_loss

    @property
    def log(self):
        """
        Collect log of objectives
        """
        out = {}
        for obj_name, obj in self.objectives_dict.items():
            out[obj_name] = obj.log
        return out
