from torch.utils import data
import numpy as np
import torch


class LOBDataset(data.Dataset):
    """ Characterizes a dataset for PyTorch. """
    def __init__(
            self,
            path,
            T,
    ):
        self.path = path
        self.T = T
        self._get_data()

    def __len__(self):
        """ Denotes the total number of samples. """
        return len(self.data)-self.T+1

    def __getitem__(self, index):
        return self.data[index:index+self.T].cuda(non_blocking=True)

    def _get_data(self):
        """ Loads the data. """
        self.data = torch.from_numpy(np.load(self.path))
