from torch.utils import data
import numpy as np
import torch
import constants as cst


class LOBDataset(data.Dataset):
    """ Characterizes a dataset for PyTorch. """
    def __init__(
            self,
            paths,
            seq_size,
            gen_seq_size,
            chosen_model,
            is_val=False,
            batch_size=None,
            limit_val_batches=None
    ):
        self.paths = paths
        self.seq_size = seq_size          #sequence length
        self.gen_seq_size = gen_seq_size      #sequence length of the input
        self.cond_seq_size = self.seq_size - self.gen_seq_size
        self.chosen_model = chosen_model
        self.is_val = is_val
        self.batch_size = batch_size
        self.limit_val_batches = limit_val_batches
        self._get_data()

    def __len__(self):
        """ Denotes the total number of samples. """
        return len(self.data)-self.seq_size+1

    def __getitem__(self, index):
        index_cond = self.cond_seq_size + index
        index_x = self.cond_seq_size + index + self.gen_seq_size
        if self.chosen_model == cst.Models.CGAN:
            orders = self.orders[index:index_x]
            market_state = self.market_data[index:index_x]
            return market_state, orders
        else:
            cond = self.orders[index:index_cond]
            x_0 = self.orders[index_cond:index_x]
            lob = self.lob[index:index_cond+1]
            return cond, x_0, lob

    def _get_data(self):
        """ Loads the data. """
        for i in range(len(self.paths)):
            if i == 0:
                path = self.paths[i]
                self.data = torch.from_numpy(np.load(path)).float().contiguous()
                if self.is_val:
                    self.data = self.data[:self.batch_size*self.limit_val_batches]
                if self.chosen_model == cst.Models.CGAN:
                    self.orders = self.data[:, :cst.LEN_ORDER_CGAN]
                    self.market_data = self.data[:, cst.LEN_ORDER_CGAN:]
                    # divisions to compute the imbalance of the order sign
                    self.market_data[:, -3] =  self.market_data[:, -3] / 128
                    self.market_data[:, -4] =  self.market_data[:, -4] / 256
                else:
                    self.orders = self.data[:, :cst.LEN_ORDER]
                    self.lob = self.data[:, cst.LEN_ORDER:]
                    self.lob = np.roll(self.lob, 1, axis=0)
                    self.lob[0, :] = 0
                    self.lob = torch.from_numpy(self.lob).float().contiguous()
            else:
                path = self.paths[i]
                data = torch.from_numpy(np.load(path)).float().contiguous()
                if self.is_val:
                    data = data[:self.batch_size*self.limit_val_batches]
                if self.chosen_model == cst.Models.CGAN:
                    orders = data[:, :cst.LEN_ORDER_CGAN]
                    market_data = data[:, cst.LEN_ORDER_CGAN:]
                    market_data[:, -3] =  market_data[:, -3] / 128
                    market_data[:, -4] =  market_data[:, -4] / 256
                    self.market_data = torch.cat((self.market_data, market_data), dim=0)
                else:
                    orders = data[:, :cst.LEN_ORDER]
                    lob = data[:, cst.LEN_ORDER:]
                    lob = np.roll(lob, 1, axis=0)
                    lob[0, :] = 0
                    lob = torch.from_numpy(lob).float().contiguous()
                    self.lob = torch.cat((self.lob, lob), dim=0)
                self.data = torch.cat((self.data, data), dim=0)
                self.orders = torch.cat((self.orders, orders), dim=0)
                    
        

    
        
        
        



        











