from config import Configuration
from constants import LearningHyperParameter
import torch.nn as nn
import constants as cst


from models.feature_augmenters.AbstractAugmenter import AugmenterAB


class LSTMAugmenter(AugmenterAB, nn.Module):
    
    def __init__(self, config: Configuration):
        super().__init__()
        dropout = config.HYPER_PARAMETERS[LearningHyperParameter.DROPOUT]
        AUGMENT_DIM = config.HYPER_PARAMETERS[LearningHyperParameter.AUGMENT_DIM]
        x_size = cst.LEN_EVENT + cst.LEN_LEVEL * config.N_LOB_LEVELS
        
        self.lstm = nn.LSTM(x_size, AUGMENT_DIM, num_layers=1, batch_first=True, dropout=dropout)
        
    def forward(self, input):
        x, (h_n, c_n) = self.lstm(input)
        return x
    
    def augment(self, input):
        return self.forward(input)

        