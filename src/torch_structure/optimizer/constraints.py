import torch
from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class ConstraintConfig:
    """
    Data container to initialize a Constraint object
    """

    name: str
    constr_function: Callable
    lower_bound: torch.Tensor = -torch.inf
    upper_bound: torch.Tensor = torch.inf
    kwargs: dict[str, Any] = field(default_factory=dict)


class Constraint:
    def __init__(self, solve_graph, constr_config: ConstraintConfig):
        """
        Initialize constraint object that evaluates the constrained values and its Jacobian matrix with respect to the design variables
        """
        # Link method to solve graph
        self.solve_graph = solve_graph
        # Store external function that acts on a solved graph
        self.constr_function = constr_config.constr_function
        # Store additional keyword arguments if needed
        self.kwargs = constr_config.kwargs
        # Store bounds as attributes
        if constr_config.upper_bound <= constr_config.lower_bound:
            raise ValueError("upper_bound must be greater than lower_bound.")
        self.lower_bound = torch.tensor(constr_config.lower_bound)
        self.upper_bound = torch.tensor(constr_config.upper_bound)
        # State variable
        self.state = {}

    def forward(self, x):
        """
        Evaluates the constrained values
        """
        # Apply design variables from x and solve graph
        graph_solved = self.solve_graph(x)
        # Evaluate the constraint function
        values = self.constr_function(graph_solved, **self.kwargs)
        # Save state
        self.state["values"] = values
        return values

    @property
    def log(self):
        """
        Returns log data as a dictionary
        """
        under = torch.relu(self.lower_bound - self.state["values"])
        over = torch.relu(-self.upper_bound + self.state["values"])
        constr_violation = under + over
        self.state["violation"] = constr_violation
        return self.state


class ConstraintHandler:
    def __init__(self, solve_graph, constr_config_list):
        """
        Initialize object to store constraint function objects
        """
        # Link to method that applies the design variables and solves the structure
        self.solve_graph = solve_graph
        # Initialize variables to store the constraints
        self.constr_dict = {}
        # Add constraints
        for constr_config in constr_config_list:
            # Add object to dictionary
            self.constr_dict[constr_config.name] = Constraint(
                self.solve_graph, constr_config
            )

    @property
    def log(self):
        """
        Aggregates the logged data in the constraint objects
        """
        log = {}
        for name, constr_function in self.constr_dict.items():
            log[name] = constr_function.log
        return log
