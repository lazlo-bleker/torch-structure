from torch_geometric.data import InMemoryDataset
from torch_structure.data import StructData
import os


class LegacyDataset(InMemoryDataset):
    """In-memory dataset that stores and returns a list of pre-loaded [StructData][torch_structure.data.data.StructData] samples as-is."""

    def __init__(self, data_list):
        super().__init__(None)
        self.data_list = data_list
        self.data, self.slices = self.collate(self.data_list)  # double check this

    def len(self):
        """Return the number of samples in the dataset."""
        return len(self.data_list)

    def get(self, idx):
        """Return the sample at ``idx``."""
        return self.data_list[idx]  # .data


class Dataset(InMemoryDataset):
    """In-memory dataset that loads a pre-processed, collated set of [StructData][torch_structure.data.data.StructData] samples from disk."""

    def __init__(self, root, load=True, transform=None):
        super().__init__(root, transform=transform)
        if load and os.path.exists(self.processed_paths[0]):
            self.load(self.processed_paths[0])
        else:
            self.data, self.slices = None, None

    @property
    def processed_file_names(self):
        """Filename of the collated dataset file, relative to ``root/processed``."""
        return ["data.pt"]

    # def get(self, idx):
    #     pyg_data = super().get(idx)
    #     return StructData.from_pyg_data(pyg_data)


def save(data_list, root, include_metadata=False):
    """Collate a list of [StructData][torch_structure.data.data.StructData] samples and save them as a [Dataset][torch_structure.data.dataset.Dataset] under ``root``.

    Args:
        data_list (list[StructData]): samples to save.
        root (str): dataset root directory; the collated file is written to
            ``root/processed/data.pt``.
        include_metadata (bool): if ``False`` (default), strips each
            sample's ``metadata`` attribute before saving.
    """
    dataset = Dataset(root, load=False)
    if not include_metadata:
        for data in data_list:
            del data.metadata
    dataset.save(data_list, dataset.processed_paths[0])
