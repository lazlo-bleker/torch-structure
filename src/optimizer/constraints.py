import torch
import numpy as np
from scipy.optimize import NonlinearConstraint

from .config import max_iters_cem

class Constraint_Object():
    def __init__(self, solve_graph_method, function, lb, ub, kwargs={}):
        """
        Initialize constraint object that evaluates the constrained values and its Jacobian matrix with respect to the design variables
        """
        # Link method to solve graph
        self.solve_graph_method = solve_graph_method
        # Store bounds as attributes
        self.lb = lb
        self.ub = ub
        # Store additional keyword arguments if needed
        self.kwargs = kwargs
        # Store external function that acts on a solved graph
        self.ext_function = function
        # Create attributes to store the constrainted values and their Jacobian matrix
        self.y = None
        self.J = None

    def fun(self, x_torch):
        """
        Evaluates the constrained values
        """
        # Apply design variables from x_torch and solve graph
        graph_solved = self.solve_graph_method(x_torch)
        # Evaluate external function using the solution
        return self.ext_function(graph_solved, **self.kwargs)
    
    def forward(self, x_np):
        """
        Evaluates the constrained values
        """
        # Cast design variables to torch tensors with gradients
        x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        # Evaluate function
        self.y = self.fun(x)
        # Cast to numpy to use in scipy
        return self.y.detach().cpu().numpy()
    
    def backward(self, x_np):
        """
        Evaluates the Jacobian matrix of the constrained values with respect to the design variables
        """
        # Cast design variables to torch tensors with gradients
        x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        # Evaluate Jacovian
        self.J = torch.func.jacrev(self.fun)(x)
        # Cast to numpy to use in scipy
        return self.J.detach().cpu().numpy()
    
    def get_constraint_violation(self):
        """
        Evaluate element-wise constraint violation
        """
        # Cast constrained variables to numpy
        y = self.y.detach().cpu().numpy()
        # Violation below lower bound
        under = np.maximum(self.lb - y, 0.0)
        # Violation above upper bound
        over = np.minimum(self.ub - y, 0.0)
        # Return signed violation
        return under - over 
    
    def get_log_dict(self):
        """
        For the optimiuation log, measures constraint violation norms and saves the result in a dictionary
        """
        # Initialize dictionary
        log_dict = {}
        # Evaluate constriant violation
        viol = self.get_constraint_violation()
        # Evaluate norms
        log_dict["l1"] = np.linalg.norm(viol, 1)
        log_dict["l2"] = np.linalg.norm(viol, 2)
        log_dict["linf"] = np.linalg.norm(viol, np.inf)
        log_dict["count"] = np.sum(np.abs(viol) > 0.0)
        # Return populated dict
        return log_dict  

class Constraint_Function_Handler():
    def __init__(self, solve_graph_method):
        """
        Initialize object to store constraint function objects
        """
        # Link to method that applies the design variables and solves the structure
        self.solve_graph_method = solve_graph_method
        # Initialize variables to store the constraints
        self.scipy_constraint_list = []
        self.constraint_objects = {}

    def add_constraint(self, name, fun, lb, ub, kwargs):
        """
        Adds a constraint and stores it in the locally
        """
        # Create instance of self-defined class
        constraint_object = Constraint_Object(self.solve_graph_method, fun, lb, ub, kwargs)
        # Add instance to dict
        self.constraint_objects[name] = constraint_object
        # Create scipy nonlinear constraint using methods of the self-defined class
        non_linear_constraint = NonlinearConstraint(
            fun = constraint_object.forward,
            lb = lb.flatten(),
            ub = ub.flatten(),
            jac = constraint_object.backward
        )
        # Add scipy constraint to list
        self.scipy_constraint_list.append(non_linear_constraint)
