import torch
from packaging.version import parse

from .data import StructData
from .dataset import Dataset, save, LegacyDataset
from .utils import requires_metadata
from .view import NodeView

__all__ = [
    "StructData",
    "Dataset",
    "save",
    "LegacyDataset",
    "requires_metadata",
    "NodeView",
]

if parse(torch.__version__) >= parse("2.4.0"):
    torch.serialization.add_safe_globals([StructData])
