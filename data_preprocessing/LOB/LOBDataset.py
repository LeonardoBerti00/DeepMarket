

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
            stock,
            start_end_trading_day,
            vol_price_mu=None,
            vol_price_sig=None,
    ):
        """ Initialization. """
        self.stock = stock
        self.start_end_trading_day = start_end_trading_day

        self.vol_price_mu = vol_price_mu
        self.vol_price_sig = vol_price_sig

        self.data = self.get_data()

    def __len__(self):
        """ Denotes the total number of samples. """
        # len(self.indexes_chosen)
        return len(self.y)-self.sample_size

    def __getitem__(self, index):
        """ Generates samples of data. """
        pass

    def get_data(self):
        """ Generates samples of data. """
        pass