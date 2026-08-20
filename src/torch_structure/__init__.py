"""
TorchStructure is a python package for graph-based autodifferentiable structural design
and engineering.
"""

from importlib.metadata import version, PackageNotFoundError

from . import data
from . import formfinding
from . import generators
from . import loader
from . import message_passing
from . import loss
from . import plot
from . import transforms
from . import utils
from . import optimizer

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0"

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
    "optimizer",
]

from .config import DEVICE
import torch
torch.set_default_device(DEVICE)