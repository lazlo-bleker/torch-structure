import torch_geometric


class Dataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, data_list):
        super().__init__(None)
        self.data_list = data_list
        self.data, self.slices = self.collate(self.data_list)  # double check this

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]  # .data
