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
            x_seq_size,
            chosen_model,
            chosen_stock
    ):
        self.path = path
        self.seq_size = seq_size          #sequence length
        self.x_seq_size = x_seq_size      #sequence length of the input
        self.cond_seq_size = self.seq_size - self.x_seq_size
        self.chosen_model = chosen_model
        self.chosen_stock = chosen_stock
        self._get_data()
        if one_hot_encoding_type:        
            self.data = one_hot_encoding_type(self.data)

    def __len__(self):
        """ Denotes the total number of samples. """
        return len(self.data)-self.seq_size+1

    def __getitem__(self, index):
        index_cond = self.cond_seq_size + index
        index_x = self.cond_seq_size + index + self.x_seq_size
        cond = self.orders[index:index_cond]
        x_0 = self.orders[index_cond:index_x]
        lob = self.lob[index:index_x]
        return cond, x_0, lob

    def _get_data(self):
        """ Loads the data. """
        self.data = torch.from_numpy(np.load(self.path)).float().contiguous()
        if self.chosen_model == cst.Models.CGAN:
            self.orders = self.data[:, :cst.LEN_ORDER_CGAN]
            self.market_data = self.data[:, cst.LEN_ORDER_CGAN:]
        else:
            self.orders = self.data[:, :cst.LEN_ORDER]
            self.lob = self.data[:, cst.LEN_ORDER:]
            self.lob = np.roll(self.lob, 1, axis=0)
            self.lob[0, :] = 0
            self.lob = torch.from_numpy(self.lob).float().contiguous()
        

    def _pre_process_gan(self):
        if self.chosen_stock == cst.Stocks.TSLA:
            order_mean_size = cst.TSLA_EVENT_MEAN_SIZE
            order_std_size = cst.TSLA_EVENT_STD_SIZE
            order_mean_price = cst.TSLA_EVENT_MEAN_PRICE
            order_std_price = cst.TSLA_EVENT_STD_PRICE
            lob_mean_size = cst.TSLA_LOB_MEAN_SIZE_10
            lob_std_size = cst.TSLA_LOB_STD_SIZE_10
            lob_mean_price = cst.TSLA_LOB_MEAN_PRICE_10
            lob_std_price = cst.TSLA_LOB_STD_PRICE_10
        elif self.chosen_stock == cst.Stocks.INTC:
            order_mean_size = cst.INTC_EVENT_MEAN_SIZE
            order_std_size = cst.INTC_EVENT_STD_SIZE
            order_mean_price = cst.INTC_EVENT_MEAN_PRICE
            order_std_price = cst.INTC_EVENT_STD_PRICE
            lob_mean_size = cst.INTC_LOB_MEAN_SIZE_10
            lob_std_size = cst.INTC_LOB_STD_SIZE_10
            lob_mean_price = cst.INTC_LOB_MEAN_PRICE_10 
            lob_std_price = cst.INTC_LOB_STD_PRICE_10
            
        cancel_depth = torch.zeros((self.orders.shape[0], 1)) 
        quantity_100 = torch.zeros((self.orders.shape[0], 1))   
        quantity_type = torch.ones((self.orders.shape[0], 1))
        quantity = (self.orders[:, 2] * order_std_size + order_mean_size).int()
        orders_price = (self.orders[:, 3] * order_std_price + order_mean_price).int()
        lob_prices = (self.lob[:, 0::2] * lob_std_price + lob_mean_price).int()
        lob_sizes = (self.lob[:, 1::2] * lob_std_size + lob_mean_size).int()
        for i in range(self.orders.shape[0]):
            if self.orders[i, 1] == 1:
                cancel_depth[i] = lob_prices[i].tolist().index(orders_price[i].item()) // 2
            if quantity[i] % 100 == 0:
                quantity_100[i] = quantity[i] // 100
                quantity_type[i] = -quantity_type[i]
        order_type = self.orders[:, 1] - 1
        depth = self.orders[:, -1]
        quantity = self.orders[:, 2]
        #check if direction are -1 and 1
        direction = self.orders[:, 4]
        self.orders = torch.cat((depth, cancel_depth, quantity, quantity_100, quantity_type, order_type, direction), dim=1)
        volume_imbalance_1 = lob_sizes[:, 1] / (lob_sizes[:, 1] + lob_sizes[:, 0])
        volume_imbalance_5 = (lob_sizes[:, 1] + lob_sizes[:, 3] + lob_sizes[:, 5] + lob_sizes[:, 7] + lob_sizes[:, 9]) / (lob_sizes[:, :10].sum(dim=1))
        absolute_volume_1 = lob_sizes[:, 1] + lob_sizes[:, 0]
        absolute_volume_5 = lob_sizes[:, :10].sum(dim=1)
        order_sign_imbalance_256 = torch.zeros((self.x_seq_size-256, 1))
        returns_50 = torch.zeros((self.x_seq_size-256, 1))
        returns_1 = torch.zeros((self.x_seq_size-256, 1))
        order_sign_imbalance_128 = torch.zeros((self.x_seq_size-256, 1))
        mid_price = (lob_prices[:, 0] + lob_prices[:, 1]) / 2
        for i in range(order_sign_imbalance_256.shape[0]):
            order_sign_imbalance_256[i] = self.orders[i:i+256, 4].sum()
            order_sign_imbalance_128[i] = self.orders[i+128:i+256, 4].sum()
            returns_1[i] = mid_price[i+255] / mid_price[i+254] - 1
            returns_50[i] = mid_price[i+255] / mid_price[i+205] - 1
        spread = lob_prices[:, 0] - lob_prices[:, 1]
        self.market_data = torch.cat((volume_imbalance_1, volume_imbalance_5, absolute_volume_1, absolute_volume_5, spread), dim=1)       
        self.market_data = self.market_data[255:]
        self.orders = self.orders[255:]
        self.market_data = torch.cat((self.market_data, order_sign_imbalance_256, order_sign_imbalance_128, returns_1, returns_50), dim=1)
        
        
        
        
        
        
            
        
        
        



        











