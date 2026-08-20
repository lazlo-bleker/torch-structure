from ..data.data import StructData
from dataclasses import dataclass
@dataclass
class ConstraintConfig:
    """
    Data container to initialize a Constraint object
    """
    name: str
class Constraint():
    def __init__(self):
        self.function = None
        self.lower_bound = None
        self.upper_bound = None
        self.kwargs = None

    def forward(self, struc_data : StructData):
        raise NotImplementedError

    def log(self):
        raise NotImplementedError
class ConstraintHandler():
    def __init__(self):
        self.constraint_dict = None

    def forward(self, x):
        raise NotImplementedError

    def log(self):
        raise NotImplementedError