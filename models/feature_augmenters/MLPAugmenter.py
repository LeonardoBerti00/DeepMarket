import configuration
from constants import LearningHyperParameter
import torch.nn as nn
import constants as cst
import torch

from models.feature_augmenters.AbstractAugmenter import AugmenterAB


class MLPAugmenter(AugmenterAB, nn.Module):
    
    def __init__(self, config, input_size):
        super().__init__()
        augment_dim = config.HYPER_PARAMETERS[LearningHyperParameter.AUGMENT_DIM]
        self.input_size = input_size
        self.fwd_fc = nn.Linear(input_size, augment_dim, dtype=torch.float32)
        self.bck_fc = nn.Linear(augment_dim, input_size, dtype=torch.float32)
        if config.CHOSEN_MODEL == cst.Models.CDT.value:
            self.v_fc = nn.Linear(augment_dim, input_size, dtype=torch.float32)

    def augment(self, input):
        return self.fwd_fc(input)
    
    def deaugment(self, input, v=None):
        if v is not None:
            return self.bck_fc(input), self.v_fc(v)
        else:
            return self.bck_fc(input)
