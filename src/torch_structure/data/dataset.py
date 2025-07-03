from torch_geometric.data import InMemoryDataset
from torch_structure.data import StructData
import os


class LegacyDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__(None)
        self.data_list = data_list
        self.data, self.slices = self.collate(self.data_list)  # double check this

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]  # .data


class Dataset(InMemoryDataset):
    def __init__(self, root, load=True, transform=None):
        super().__init__(root, transform=transform)
        if load and os.path.exists(self.processed_paths[0]):
            self.load(self.processed_paths[0])
        else:
            self.data, self.slices = None, None

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def get(self, idx):
        pyg_data = super().get(idx)
        return StructData.from_pyg_data(pyg_data)


def save(data_list, root, include_metadata=True):
    dataset = Dataset(root, load=False)
    pyg_data_list = [
        data.export_pyg_data(include_metadata=include_metadata) for data in data_list
    ]
    dataset.save(pyg_data_list, dataset.processed_paths[0])
