from ..data.data import StructData
from .design_variables import DesignVariableHandler
from .objective_function import ObjectiveHandler
from .constraints import ConstraintHandler
from .logger import Logger

class Optimizer():
    def __init__(
            self, 
            struc_data :StructData, 
            solver_config, 
            dv_config_list, 
            obj_func_config_list,
            logger_config, 
            constr_func_config_list = None
            ):

        # Constraints are optional
        if constr_config_list is None:
            constr_config_list = []

        self.dv_handler = DesignVariableHandler(struc_data, dv_config_list, solver_config)
        self.obj_handler = ObjectiveHandler(obj_func_config_list)
        self.constr_handler = ConstraintHandler(constr_func_config_list)
        self.logger = Logger(logger_config)

    def run(self):
        raise NotImplementedError