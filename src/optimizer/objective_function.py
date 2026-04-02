import torch
from dataclasses import dataclass, field
from typing import Callable, Any

from config import torch_to_np_float, TORCH_FLOAT


@dataclass
class ObjectiveConfig:
    name: str
    obj_function: Callable
    weight: float = 1.0
    kwargs: dict[str, Any] = field(default_factory=dict)


class Objective:
    def __init__(self, objective_config: ObjectiveConfig):
        self.obj_function = objective_config.obj_function
        self.weight = objective_config.weight
        self.kwargs = dict(objective_config.kwargs)

    def __call__(self, graph) -> torch.Tensor:
        self.y = self.obj_function(graph, **self.kwargs)
        return self.y

    @property
    def log(self):
        return self.y.item()


class ObjectiveHandler:
    def __init__(self, solve_graph, obj_func_config_list: list[ObjectiveConfig]):
        """
        Creates an object that evaluates the objective function used in the optimization, which provides gradients from pytorch
        """
        # Link method to apply the value of the design variables and solve the system
        self.solve_graph = solve_graph
        # Init partial loss weights (compromise vector)
        self.weights = {}
        # Init collection of Function_Objects
        self.function_objects = {}
        # Add objective functions
        for obj_func_config in obj_func_config_list:
            # Add object to dictionary
            self.function_objects[obj_func_config.name] = Objective(obj_func_config)

    def forward(self, x):
        """
        Evaluates the objective function
        """
        # Apply design variables & solve CEM
        graph_solved = self.solve_graph(x)
        # Compute total loss based on CEM solution
        total_loss = torch.tensor(0.0)
        for _, obj_function in self.function_objects.items():
            # Add contribution of obj function from list
            total_loss += obj_function.weight * obj_function(graph_solved)
        # Save total loss too
        return total_loss

    def func_grad(self, x):
        """
        Combines forward method with an automatically generated backward method
        """
        # Generate grad and value modes from forward
        func_grad_value = torch.func.grad_and_value(self.forward)
        # Evaluate grad and value together
        return func_grad_value(x)

    def func_grad_scipy(self, x_np):
        # Convert numpy to torch with gradients being required
        x = torch.tensor(x_np, dtype=TORCH_FLOAT)
        # Generate grad and value modes from forward
        func_grad_value = torch.func.grad_and_value(self.forward)
        # Evaluate grad and value together
        grad, loss = func_grad_value(x)
        # Cast to numpy to use in scipy
        return loss.item(), torch_to_np_float(grad)

    @property
    def log(self):
        """
        Collect log of objectives
        """
        out = {}
        for obj_name, obj in self.function_objects.items():
            out[obj_name] = obj.log
        return out
