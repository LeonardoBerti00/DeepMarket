from torch.utils import data
import numpy as np
import torch
import constants as cst

class LOBDataset(data.Dataset):
    """ Characterizes a dataset for PyTorch. """
    def __init__(
            self,
            path,
            seq_size,
            cond_type,
            x_seq_size
    ):
        self.path = path
        self.cond_type = cond_type
        self.seq_size = seq_size          #sequence length
        self.x_seq_size = x_seq_size      #sequence length of the input
        self.cond_seq_size = self.seq_size - self.x_seq_size
        self._get_data()
        #count the number of 0.0 in the data
        self.num_zeros = torch.sum(self.data == 0.0).item()
        print("Number of zeros in the data: ", self.num_zeros)
        print(self.data[0:10, :])

    def __len__(self):
        """ Denotes the total number of samples. """
        return len(self.data)-self.seq_size+1

    def __getitem__(self, index):
        #index = 0
        print()
        print("Number of zeros in the data second step: ", torch.sum(self.data[index:index+self.L, :] == 0.0).item())
        print(self.data[index:index+self.L, :])
        index_cond = self.cond_seq_size + index
        index_x = self.cond_seq_size + index + self.x_seq_size
        if self.cond_type == 'full':
            cond = input[index:index_cond, :],
            x_0 = input[index_cond: index_x, :cst.LEN_EVENT]
        elif self.cond_type == 'only_event':
            cond = input[index:index_cond, :cst.LEN_EVENT],
            x_0 = input[index_cond: index_x, :cst.LEN_EVENT]
        else:
            raise ValueError(f"Unknown cond_type {self.cond_type}")

        return cond, x_0


    def _get_data(self):
        """ Loads the data. """
        self.data = torch.from_numpy(np.load(self.path)).to(cst.DEVICE, torch.float32)




