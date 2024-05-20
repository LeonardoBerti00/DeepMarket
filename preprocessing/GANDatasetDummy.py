from torch.utils import data
import numpy as np
import torch
import constants as cst


class GANDatasetDummy(data.Dataset):
    """ Characterizes a dataset for PyTorch. """
    def __init__(
            self,
            seq_size,
            market_feature_dim,
            market_orders_dim,
    ):
        self.seq_size = seq_size          #sequence length
        self.market_feature_dim = market_feature_dim
        self.market_orders_dim = market_orders_dim
        self._get_data()

    def __len__(self):
        """ Denotes the total number of samples. """
        return len(self.data)

    def __getitem__(self, index):
        market_features = self.market_features[index]
        market_orders = self.market_orders[index]
        return market_features, market_orders

    def _get_data(self):
        """ Loads the data. """
        self.market_features = torch.ones(size=(1000, self.seq_size, self.market_feature_dim))
        self.market_orders = torch.ones(size=(1000, self.seq_size, self.market_orders_dim))
        self.data = self.market_features


        











