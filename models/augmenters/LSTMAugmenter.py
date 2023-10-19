from config import Configuration
from constants import LearningHyperParameter
import torch.nn as nn
import constants as cst


from models.augmenters.AbstractAugmenter import AugmenterAB


class LSTMAugmenter(AugmenterAB, nn.Module):
    
    def __init__(self, config: Configuration):
        super().__init__()
        dropout = config.HYPER_PARAMETERS[LearningHyperParameter.DROPOUT]
        latent_dim = config.HYPER_PARAMETERS[LearningHyperParameter.LATENT_DIM]
        x_size = cst.LEN_EVENT
        
        self.lstm = nn.LSTM(x_size, latent_dim, num_layers=1, batch_first=True, dropout=dropout)
        
    def forward(self, input):
        x, (h_n, c_n) = self.lstm(input)
        return x
    
    def augment(self, input):
        return self.forward(input)

        