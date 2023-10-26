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
        print(f'augment_dim = {augment_dim}')
        print(f'input_size = {input_size}')
        self.bck_lstm = nn.LSTM(augment_dim, input_size, num_layers=1, batch_first=True, dropout=dropout)
        
    def forward(self, input):
        x, (h_n, c_n) = self.fwd_lstm(input)
        return x
    
    def augment(self, input):
        return self.forward(input)
    
    def deaugment(self, input):
        x, (h_n, c_n) = self.bck_lstm(input)
        return x

        