"""
TorchStructure is a python package for graph-based autodifferentiable structural design and engineering.
"""
from . import data
from . import formfinding
from . import generators
from . import message_passing
from . import loss
from . import plot
from . import utils

__version__ = "0.0.1"

__all__ = [
    "data",
    "formfinding",
    "generators",
    "message_passing",
    "loss",
    "plot",
    "utils",
]
