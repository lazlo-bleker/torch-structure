import torch

from torch_structure.data import StructData
from .design_variables import Design_Variable_Handler
from .objective_function import Objective_Function_Handler
from .constraints import Constraint_Function_Handler
from .logger import Logger

from .config import max_iters_cem, ftol
from scipy.optimize import minimize

class Optimizer():
    def __init__(self, 
                 graph : StructData,
                 dv_dicts = None
                 ):
        """
        Constructor for the optimizer of a SturcData object
        """
        # Design variables must be provided
        assert dv_dicts is not None

        # Link instance of object to graph
        self.graph = graph
        # Initialize child objects
        self.design_variable_handler = Design_Variable_Handler(self.graph, dv_dicts)
        self.objective_function_handler = Objective_Function_Handler(self.solve_graph)
        self.constraint_function_handler = Constraint_Function_Handler(self.solve_graph)
        self.logger = Logger(self)
    
    def add_obj_function(self, name, weight, func, kwargs={}):
        """
        Pass task to child instance of Objective_Function_Handler
        """
        self.objective_function_handler.add_function(name, weight, func, kwargs)

    def add_constr_function(self, name : str, fun, lb, ub, kwargs):
        """
        Pass task to child instance of Constraint_Function_Handler
        """
        self.constraint_function_handler.add_constraint(name, fun, lb, ub, kwargs)
    
    def solve_graph(self, x):
        """
        Use the optimizer as an intermediator to apply the values of the design variables through the instance of Design_Variable_Handler and solve the modified graph via CEM
        """
        # Apply design variables
        self.design_variable_handler.apply(x)
        # Solve CEM
        self.graph.cem(max_iter=max_iters_cem, inplace=True)
        return self.graph
    
    def run(self, max_iters_opt : int, n_shots : int, log = True):
        """
        Run optimization using the gradients computed with pytorch and the SLSQP optimizer of scipy
        """
        # Get number of iterations between snapshots
        n_iters = int(max_iters_opt/n_shots)
        # Get initial values
        x0 = self.design_variable_handler.get_values()

        # Use the method from child Objective_Function_Handler
        obj_func = self.objective_function_handler.func_grad_value_scipy
        # Use the constraints defined inside child Constraint_Function_Handler
        constr_list = self.constraint_function_handler.scipy_constraint_list
        # Use the logger as callback to log results to tensorboard
        if log:
            callback = self.logger
        else:
            callback = None

        obj_func(x0)
        for constr in constr_list:
            constr.fun(x0)
        # For each snapshot to capture
        for n_shot in range(n_shots):
            # Plot structure (capture snapshot)
            if log:
                self.logger.plot(self.graph, n_shot)

            # Run scipy with gradients from torch_structure
            result = minimize(
                fun = obj_func,
                args=(),
                x0 = x0,
                method="SLSQP",
                jac=True,
                constraints = constr_list,
                callback = callback,
                options={"disp": True, "ftol": ftol, "maxiter": n_iters},
            )

            # Update design variable values
            x0 = result.x

        # Generate GIF from snapshots
        if log:
            self.logger.generate_gif()
        return result