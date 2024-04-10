import configuration
from constants import LearningHyperParameter
import torch.nn as nn
import constants as cst


from models.feature_augmenters.AbstractAugmenter import AugmenterAB


class LSTMAugmenter(AugmenterAB, nn.Module):
    
    def __init__(self, input_size, augment_dim, cond_size, cond_type):
        super().__init__()
        augment_dim = augment_dim
        self.input_size = input_size
        self.fwd_lstm = nn.LSTM(input_size, augment_dim, num_layers=2, batch_first=True, device=cst.DEVICE)
        self.bck_lstm = nn.LSTM(augment_dim, input_size, num_layers=2, batch_first=True, device=cst.DEVICE)
        if cond_type == "full":
            self.fwd_cond_lstm = nn.LSTM(cond_size, augment_dim, num_layers=2, batch_first=True, device=cst.DEVICE)
        self.v_lstm = nn.LSTM(augment_dim, input_size, num_layers=2, batch_first=True, device=cst.DEVICE)
        self.cond_type = cond_type


    def forward(self, input, cond):
        out, (h_n, c_n) = self.fwd_lstm(input)
        if self.cond_type == "full":
            cond, (h_n, c_n) = self.fwd_cond_lstm(cond)
        return out, cond
     
    def augment(self, input, cond=None):
        return self.forward(input, cond)
    
    def deaugment(self, input, v=None):
        input, (_, _) = self.bck_lstm(input)
        if self.chosen_model == cst.Models.CDT.value:
            v, (_, _) = self.v_lstm(v)
        return input, v


        