from torch.utils import data
import numpy as np
import torch
import constants as cst

class LOBDataset(data.Dataset):
    """ Characterizes a dataset for PyTorch. """
    def __init__(
            self,
            path,
            L,
    ):
        self.path = path
        self.L = L          #sequence length
        self._get_data()
        #count the number of 0.0 in the data
        #self.num_zeros = torch.sum(self.data == 0.0).item()
        #print("Number of zeros in the data: ", self.num_zeros)
        #print(self.data[0:10, :])

    def __len__(self):
        """ Denotes the total number of samples. """
        return len(self.data)-self.L+1

    def __getitem__(self, index):
        #index = 0
        #print("Number of zeros in the data: ", torch.sum(self.data[index:index+self.L, :] == 0.0).item())
        #print(self.data[index:index+self.L, :])
        return self.data[index:index+self.L, :]


    def _get_data(self):
        """ Loads the data. """
        self.data = torch.from_numpy(np.load(self.path)).to(cst.DEVICE, torch.float32)




