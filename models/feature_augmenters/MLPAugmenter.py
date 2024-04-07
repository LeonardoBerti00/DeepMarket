import configuration
from constants import LearningHyperParameter
import torch.nn as nn
import constants as cst
import torch

from models.feature_augmenters.AbstractAugmenter import AugmenterAB


class MLPAugmenter(AugmenterAB, nn.Module):
    
    def __init__(self, input_size, augment_dim, chosen_model):
        super().__init__()
        augment_dim = augment_dim
        self.input_size = input_size
        self.fwd_mlp = nn.Sequential(
            nn.Linear(input_size, augment_dim//2, dtype=torch.float32),
            nn.PReLU(init=0.01),
            nn.LayerNorm(augment_dim//2),
            nn.Linear(augment_dim//2, augment_dim, dtype=torch.float32),
        )
        self.bck_mlp = nn.Sequential(
            nn.Linear(augment_dim, augment_dim//2, dtype=torch.float32),
            nn.PReLU(init=0.01),
            nn.LayerNorm(augment_dim//2),
            nn.Linear(augment_dim//2, input_size, dtype=torch.float32),
        )
        self.v_mlp = nn.Sequential(
            nn.Linear(augment_dim, augment_dim//2, dtype=torch.float32),
            nn.PReLU(init=0.01),
            nn.LayerNorm(augment_dim//2),
            nn.Linear(augment_dim//2, input_size, dtype=torch.float32),
        )

    def augment(self, input):
        x = self.fwd_mlp(input)
        return x
    
    def deaugment(self, input, v=None):
        if v is not None:
            input = self.bck_mlp(input)
            v = self.v_mlp(v)
            return input, v
        else:
            return self.bck_mlp(input)
