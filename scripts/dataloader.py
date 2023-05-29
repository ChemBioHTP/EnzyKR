import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

class Data_loader(InMemoryDataset):

    def __init__(self, path):
        super().__init__(path)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['tutorial.csv']

    @property
    def processed_file_names(self):
        return ['processed_data_tutorial.pt']


if __name__ == "__main__":
    dataset = Data_loader('../')
