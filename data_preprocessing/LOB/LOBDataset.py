from torch.utils import data
import numpy as np
import torch
import constants as cst


class LOBDataset(data.Dataset):
    """ Characterizes a dataset for PyTorch. """
    def __init__(
            self,
            path,
            T,

    ):
        self.path = path
        self.get_data()
        self.T = T

    def __len__(self):
        """ Denotes the total number of samples. """
        return len(self.data)-self.T+1

    def __getitem__(self, index):
        return self.data[index:index+self.T]

    def get_data(self):
        """ Loads the data. """
        self.data = torch.from_numpy(np.load(self.path), device=cst.DEVICE_TYPE)
