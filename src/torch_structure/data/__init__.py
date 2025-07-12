import torch
from packaging.version import parse

from .data import Data, StructData
from .dataset import Dataset, save, LegacyDataset
from .view import NodeView

__all__ = ["Data", "StructData", "Dataset", "save", "LegacyDataset", "NodeView"]

if parse(torch.__version__) >= parse("2.4.0"):
    torch.serialization.add_safe_globals([Data])
