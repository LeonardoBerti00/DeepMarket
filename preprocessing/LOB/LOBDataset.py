import time

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



    def __len__(self):
        """ Denotes the total number of samples. """
        return len(self.data)-self.seq_size+1

    def __getitem__(self, index):
        index_cond = self.cond_seq_size + index
        index_x = self.cond_seq_size + index + self.x_seq_size
        cond = self.encoded_data[index:index_cond, :]
        x_0 = self.encoded_data[index_cond: index_x, :cst.LEN_EVENT_ONE_HOT]
        return cond, x_0


    def _get_data(self):
        """ Loads the data. """
        self.data = torch.from_numpy(np.load(self.path))

    def one_hot_encode(self):
        ''' one hot encode the second and final column'''
        self.encoded_data = torch.zeros(self.data.shape[0], self.data.shape[1] + 2)
        self.encoded_data[:, 0] = self.data[:, 0]
        #encoding order type
        one_hot_order_type = torch.nn.functional.one_hot((self.data[:, 1]).to(torch.int64), num_classes=3).to(torch.float32)
        self.encoded_data[:, 1:4] = one_hot_order_type
        self.encoded_data[:, 4] = self.data[:, 2]
        self.encoded_data[:, 5] = self.data[:, 3]
        self.encoded_data[:, 6:] = self.data[:, 4:]

    def transform_data(self):
        #transform execution order in add order of the opposite side
        for i in range(self.data.shape[0]):
            if (self.data[i, 1] == 3):
                self.data[i, 1] = 0
                if (self.data[i, 2] < 0):
                    self.data[i, 2] = -self.data[i, 2]
                elif self.data[i, 2] > 0:
                    self.data[i, 2] = -self.data[i, 2]
                else:
                    raise ValueError("Execution order with size 0")









