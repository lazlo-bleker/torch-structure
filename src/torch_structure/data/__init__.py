import torch

from .data import StructData
from .dataset import Dataset, save, LegacyDataset
from .utils import requires_metadata
from .view import NodeView

__all__ = ["StructData", "Dataset", "save", "LegacyDataset", "requires_metadata", "NodeView"]

torch.serialization.add_safe_globals([StructData])
