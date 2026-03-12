import torch
from scipy.optimize import NonlinearConstraint
from dataclasses import dataclass, field
from typing import Callable, Any

from config import torch_to_np_float, np_to_torch_float


@dataclass
class ConstraintConfig:
    """
    Data container to initialize a Constraint object
    """

    name: str
    constr_function: Callable
    lower_bound: torch.Tensor | None = None
    upper_bound: torch.Tensor | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


class ConstraintObject:
    def __init__(self, solve_graph_method, constr_config: ConstraintConfig):
        """
        Initialize constraint object that evaluates the constrained values and its Jacobian matrix with respect to the design variables
        """
        # Link method to solve graph
        self.solve_graph_method = solve_graph_method
        # Store external function that acts on a solved graph
        self.constr_function = constr_config.constr_function
        # Store additional keyword arguments if needed
        self.kwargs = constr_config.kwargs
        # Store bounds as attributes
        self.lower_bound = constr_config.lower_bound
        self.upper_bound = constr_config.upper_bound
        # Create attributes to store the constrainted values and their Jacobian matrix
        self.y = None
        self.J = None

    def forward(self, x):
        """
        Evaluates the constrained values
        """
        # Apply design variables from x and solve graph
        graph_solved = self.solve_graph_method(x)
        # Evaluate external function using the solution
        self.y = self.constr_function(graph_solved, **self.kwargs)
        return self.y

    def backward(self, x):
        """
        Evaluates the Jacobian matrix of the constrained values with respect to the design variables
        """
        # Evaluate Jacovian
        self.J = torch.func.jacrev(self.forward)(x)
        return self.J

    def forward_scipy(self, x):
        """
        Evaluates the constrained values
        (interfaces with scipy by converting to numpy arrays)
        """
        x = np_to_torch_float(x)
        return torch_to_np_float(self.forward(x))

    def backward_scipy(self, x):
        """
        Evaluates the Jacobian matrix of the constrained values with respect to the design variables
        (interfaces with scipy by converting to numpy arrays)
        """
        x = np_to_torch_float(x)
        return torch_to_np_float(self.backward(x))

    def log(self):
        """
        TODO: option to log data on a Constraint object
        """
        raise NotImplementedError


class ConstraintHandler:
    def __init__(self, solve_graph_method, constr_config_list):
        """
        Initialize object to store constraint function objects
        """
        # Link to method that applies the design variables and solves the structure
        self.solve_graph_method = solve_graph_method
        # Initialize variables to store the constraints
        self.constraint_objects = {}

        # Add constraints
        for constr_config in constr_config_list:
            # Add object to dictionary
            self.add_constraint(constr_config)

    def add_constraint(self, constr_config: ConstraintConfig):
        """
        Adds a constraint and stores it in the locally
        """
        # Create instance of self-defined class
        constraint_object = ConstraintObject(self.solve_graph_method, constr_config)
        # Add instance to dict
        self.constraint_objects[constr_config.name] = constraint_object

    def build_scipy_constraints(self):
        """
        Initializes objects of type NonlinearConstraint to be used in scipy.minimize
        """
        constraint_list_scipy = []
        # Create scipy nonlinear constraint using methods of the self-defined class
        for _, constraint_object in self.constraint_objects.items():
            non_linear_constraint = NonlinearConstraint(
                fun=constraint_object.forward_scipy,
                jac=constraint_object.backward_scipy,
                lb=torch_to_np_float(constraint_object.lower_bound),
                ub=torch_to_np_float(constraint_object.upper_bound),
            )
            # Add scipy constraint to list
            constraint_list_scipy.append(non_linear_constraint)
        return constraint_list_scipy

    def log(self):
        """
        Aggregates the logged data in the constraint objects
        """
        log = {}
        for name, obj_function in self.constraint_objects.items():
            log[name] = obj_function.log()
        return log
