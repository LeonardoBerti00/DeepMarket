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
        self.lstm = nn.LSTM(input_size, augment_dim, num_layers=1, batch_first=True, dropout=dropout)
        
    def forward(self, input):
        x, (h_n, c_n) = self.lstm(input)
        return x
    
    def augment(self, input):
        return self.forward(input)

        