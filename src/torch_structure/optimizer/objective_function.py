from ..data.data import StructData
from dataclasses import dataclass
@dataclass
class ObjectiveConfig:
    """
    Data container to initialize a Constraint object
    """
    name: str
class Objective():
    def __init__(self):
        self.function = None
        self.weight = None
        self.kwargs = None

    def forward(self, struc_data : StructData):
        raise NotImplementedError

    def log(self):
        raise NotImplementedError
    
class ObjectiveHandler():
    def __init__(self):
        self.obj_func_dict = None

    def forward(self, x):
        raise NotImplementedError

    def log(self):
        raise NotImplementedError