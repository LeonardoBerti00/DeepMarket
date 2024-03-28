import time

from torch.utils import data
import numpy as np
import torch
import constants as cst
from utils.utils_data import one_hot_encoding_type, tanh_encoding_type


class LOBDataset(data.Dataset):
    """ Characterizes a dataset for PyTorch. """
    def __init__(
            self,
            path,
            seq_size,
            one_hot_encoding_type,
            x_seq_size
    ):
        self.path = path
        self.seq_size = seq_size          #sequence length
        self.x_seq_size = x_seq_size      #sequence length of the input
        self.cond_seq_size = self.seq_size - self.x_seq_size
        self._get_data()
        if one_hot_encoding_type:        
            self.data = one_hot_encoding_type(self.data)

    def __len__(self):
        """ Denotes the total number of samples. """
        return len(self.data)-self.seq_size+1

    def __getitem__(self, index):
        index_cond = self.cond_seq_size + index
        index_x = self.cond_seq_size + index + self.x_seq_size
        cond = self.data[index:index_cond, :]
        x_0 = self.data[index_cond:index_x, :cst.LEN_EVENT]
        return cond, x_0

    def _get_data(self):
        """ Loads the data. """
        self.data = torch.from_numpy(np.load(self.path)).float()        



        











