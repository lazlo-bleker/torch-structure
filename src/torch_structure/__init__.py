"""
TorchStructure is a python package for graph-based autodifferentiable structural design and engineering.
"""

from . import data
from . import formfinding
from . import generators
from . import loader
from . import message_passing
from . import loss
from . import plot
from . import transforms
from . import utils

__version__ = "0.0.1"

__all__ = [
    "data",
    "formfinding",
    "generators",
    "loader",
    "message_passing",
    "loss",
    "plot",
    "transforms",
    "utils",
]
