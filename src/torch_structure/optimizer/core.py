from ..data.data import StructData
from .design_variables import DesignVariableHandler
from .objective_function import ObjectiveHandler
from .constraints import ConstraintHandler
from .logger import Logger

class Optimizer():
    def __init__(self, struc_data :StructData):
        self.struc_data = struc_data
        self.dv_handler = DesignVariableHandler()
        self.obj_handler = ObjectiveHandler()
        self.constr_handler = ConstraintHandler
        self.logger = Logger()

    def run(self):
        raise NotImplementedError