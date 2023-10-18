from torch.utils import data
import torch.nn.functional as F
import numpy as np
import torch
import pandas as pd
import constants as cst
import collections
import data_preprocessing.preprocessing_utils as ppu

from data_preprocessing.LOB.LOBSTERDataBuilder import LOBSTERDataBuilder
from config import Configuration


class LOBDataset(data.Dataset):
    """ Characterizes a dataset for PyTorch. """

    def __init__(
            self,
            path,
    ):
        self.path = path
        self.get_data()

    def __len__(self):
        """ Denotes the total number of samples. """
        # len(self.indexes_chosen)
        return len(self.y)-self.sample_size

    def __getitem__(self, index):
        """ Generates samples of data. """
        pass

    def get_data(self):
        """ Loads the data. """
        self.data = np.load(self.path)
