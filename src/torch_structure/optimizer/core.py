from scipy.optimize import minimize, Bounds
import torch
from ..data import StructData
from .design_variables import DesignVariableHandler
from .objectives import ObjectiveHandler
from .constraints import ConstraintHandler
from .logger import Logger

from ..utils import scipy_objective, scipy_constraint, torch_to_numpy_float


class Optimizer:
    def __init__(
        self,
        struc_data: StructData,
        solver_config,
        dv_config_list,
        obj_func_config_list,
        logger_config,
        constr_config_list=[],
    ):
        self.dv_handler = DesignVariableHandler(
            struc_data, dv_config_list, solver_config
        )
        self.obj_handler = ObjectiveHandler(
            self.dv_handler.solve_graph, obj_func_config_list
        )
        self.constr_handler = ConstraintHandler(
            self.dv_handler.solve_graph, constr_config_list
        )
        self.logger = Logger(
            logger_config, self.dv_handler, self.obj_handler, self.constr_handler
        )
        self.opt_method = "SLSQP"
        self.minimize_kwargs = {"disp": True, "ftol": 1e-7, "maxiter": 100}

    def run(self, log=True):
        """
        Run optimization using the gradients computed with pytorch and the SLSQP optimizer of scipy
        """
        # Get initial values
        x0 = torch_to_numpy_float(self.dv_handler.x)
        # Use the method from child Objective_Function_Handler
        obj_func = scipy_objective(self.obj_handler.forward)
        # Use the constraints defined inside constr_handler
        constr_list = []
        for _, constr in self.constr_handler.constr_dict.items():
            constr_list.append(scipy_constraint(constr))

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
        bounds = torch_to_numpy_float(self.dv_handler.bounds)

        # Run scipy with gradients from torch_structure
        result = minimize(
            fun=obj_func,
            args=(),
            x0=x0,
            method=self.opt_method,
            jac=True,
            constraints=constr_list,
            callback=callback,
            bounds=Bounds(bounds[:, 0], bounds[:, 1]),
            options=self.minimize_kwargs,
        )

        self.logger.close()

        return result
