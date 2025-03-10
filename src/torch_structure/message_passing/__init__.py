from .fem import FEM, StiffnessAggregator
from .laplacian_smooth import LaplacianSmoothing
from .lignn import LocalLogicLoss, WeightedMax
from .residual_force import ResidualForce

__all__ = [
    "FEM",
    "StiffnessAggregator",
    "LaplacianSmoothing",
    "LocalLogicLoss",
    "WeightedMax",
    "ResidualForce",
]
