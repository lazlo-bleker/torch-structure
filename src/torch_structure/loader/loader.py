import torch
from torch_geometric.data import Batch
from torch_structure.data import StructData, Dataset
from typing import List, Optional, Sequence, Union


class Collater:
    """Collates a list of [StructData][torch_structure.data.data.StructData] samples into a batch."""

    def __init__(
        self,
        dataset: Union[Dataset, Sequence[StructData]],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[StructData]) -> StructData:
        pyg_list = [struct_data.data for struct_data in batch]
        batched = Batch.from_data_list(
            pyg_list, follow_batch=self.follow_batch, exclude_keys=self.exclude_keys
        )
        struct_data = StructData.from_pyg_data(batched)

        struct_data.node_attr_list = batch[0].node_attr_list
        struct_data.edge_attr_list = batch[0].edge_attr_list
        struct_data.graph_attr_list = batch[0].graph_attr_list

        return struct_data


class DataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    [Dataset][torch_structure.data.dataset.Dataset] to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: ``1``)
        shuffle (bool, optional): If set to ``True``, the data will be
            reshuffled at every epoch. (default: ``False``)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: ``None``)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: ``None``)
        **kwargs (optional): Additional arguments of
            `torch.utils.data.DataLoader`.
    """

    # TODO: Data objects can be either of type :class:`~torch_geometric.data.StructData` or
    # :class:`~torch_geometric.data.StructHeteroData`.
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[StructData]],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop("collate_fn", None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(
                dataset=dataset,
                follow_batch=follow_batch,
                exclude_keys=exclude_keys,
            ),
            **kwargs,
        )
