from dataclasses import dataclass, field
from scipy.optimize import minimize, Bounds

from torch_structure.data import StructData
from torch_structure.mixins import solver_names
from .design_variables import DesignVariableHandler
from .objective_function import ObjectiveHandler
from .constraints import ConstraintHandler
from .logger import Logger


@dataclass
class SolverConfig:
    """
    Data container to specify structural solver and optimization method with their parameters
    """

    solver_name: str
    solver_kwargs: dict = field(default_factory=dict)
    export_dir: str = "results/run"
    log_interval: int = 10
    opt_method: str = "SLSQP"
    ftol: float = 1e-7


class Optimizer:
    def __init__(
        self,
        graph: StructData,
        solver_config: SolverConfig,
        dv_config_list,
        obj_func_config_list,
        constr_config_list=None,
    ):
        """
        Constructor for the optimizer of a SturcData object
        """
        # If constraints are not provided, set to empty list
        if constr_config_list is None:
            constr_config_list = []
        # Link instance of object to graph
        self.graph = graph

        # Assign solver-related attributes
        if solver_config.solver_name not in solver_names:
            raise ValueError(
                f"Solver {solver_config.solver_name} not in {solver_names}"
            )
        self.solver = getattr(self.graph, solver_config.solver_name)
        self.solver_kwargs = dict(solver_config.solver_kwargs)
        self.solver_kwargs["inplace"] = True
        self.opt_method = solver_config.opt_method
        self.ftol = solver_config.ftol

        # Initialize child objects
        self.design_variable_handler = DesignVariableHandler(self.graph, dv_config_list)
        self.objective_function_handler = ObjectiveHandler(
            self.solve_graph, obj_func_config_list
        )
        self.constraint_function_handler = ConstraintHandler(
            self.solve_graph, constr_config_list
        )
        self.logger = Logger(
            self,
            log_interval=solver_config.log_interval,
            export_dir=solver_config.export_dir,
        )

    def solve_graph(self, x):
        """
        Use the optimizer as an intermediator to apply the values of the design variables through the instance of Design_Variable_Handler and solve the modified graph via CEM
        """
        # Apply design variables
        self.design_variable_handler.apply(x)
        # Solve structure
        self.solver(**self.solver_kwargs)
        return self.graph

    def run(self, max_iters_opt: int, log=True):
        """
        Run optimization using the gradients computed with pytorch and the SLSQP optimizer of scipy
        """
        # Get initial values
        x0 = self.design_variable_handler.get_values_scipy()
        # Use the method from child Objective_Function_Handler
        obj_func = self.objective_function_handler.func_grad_scipy
        # Use the constraints defined inside child Constraint_Function_Handler
        constr_list = self.constraint_function_handler.build_scipy_constraints()
        # Use the logger as callback to log results to tensorboard
        if log:
            callback = self.logger
        else:
            callback = None
        # Test objectuve and constraint functions
        obj_func(x0)
        for constr in constr_list:
            constr.fun(x0)

        # Get bounds of design variables
        lb, ub = self.design_variable_handler.get_bounds_scipy()

        # Run scipy with gradients from torch_structure
        result = minimize(
            fun=obj_func,
            args=(),
            x0=x0,
            method=self.opt_method,
            jac=True,
            constraints=constr_list,
            callback=callback,
            bounds=Bounds(lb, ub),
            options={"disp": True, "ftol": self.ftol, "maxiter": max_iters_opt},
        )

        return result