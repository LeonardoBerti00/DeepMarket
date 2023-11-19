from config import Configuration
from constants import LearningHyperParameter
import torch.nn as nn
import constants as cst


from models.feature_augmenters.AbstractAugmenter import AugmenterAB


class LSTMAugmenter(AugmenterAB, nn.Module):
    
    def __init__(self, config: Configuration, input_size):
        super().__init__()
        dropout = config.HYPER_PARAMETERS[LearningHyperParameter.DROPOUT]
        augment_dim = config.HYPER_PARAMETERS[LearningHyperParameter.AUGMENT_DIM]
        self.input_size = input_size
        self.fwd_lstm = nn.LSTM(input_size, augment_dim, num_layers=1, batch_first=True, dropout=dropout)
        self.bck_lstm = nn.LSTM(augment_dim, input_size, num_layers=1, batch_first=True, dropout=dropout)
        self.chosen_model = config.CHOSEN_MODEL
        if config.CHOSEN_MODEL == cst.Models.CDT.value:
            self.v_lstm = nn.LSTM(augment_dim, input_size, num_layers=1, batch_first=True, dropout=dropout)


    def forward(self, input):
        out, (h_n, c_n) = self.fwd_lstm(input)
        return out
    
    def augment(self, input):
        return self.forward(input)
    
    def deaugment(self, input, v=None):
        input, (_, _) = self.bck_lstm(input)
        if self.chosen_model == cst.Models.CDT.value:
            v, (_, _) = self.v_lstm(v)
        return input, v


        