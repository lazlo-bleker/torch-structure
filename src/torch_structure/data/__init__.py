import torch
import torch_geometric.typing

from .data import Data, StructData
from .dataset import Dataset, save, LegacyDataset
from .view import NodeView

__all__ = ["Data", "StructData", "Dataset", "save", "LegacyDataset", "NodeView"]

if torch_geometric.typing.WITH_PT24:
    torch.serialization.add_safe_globals([Data])
