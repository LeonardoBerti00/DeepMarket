import configuration
from constants import LearningHyperParameter
import torch.nn as nn
import constants as cst
import torch

from models.diffusers.CDT.Transformer import TransformerEncoder
from models.feature_augmenters.AbstractAugmenter import AugmenterAB


class MLPAugmenter(AugmenterAB, nn.Module):
    
    def __init__(self, input_size, augment_dim, cond_size, cond_type, cond_augmenter, cond_method):
        super().__init__()
        augment_dim = augment_dim
        self.input_size = input_size
        self.fwd_mlp = nn.Sequential(
            nn.Linear(input_size, augment_dim//2, dtype=torch.float32),
            nn.Linear(augment_dim//2, augment_dim, dtype=torch.float32),
        )
        self.cond_type = cond_type
        if cond_type == "full":
            if cond_augmenter == "MLP":
                self.fwd_cond_lob = nn.Sequential(
                    nn.Linear(cond_size, augment_dim, dtype=torch.float32),
                )
            elif cond_augmenter == "Transformer":
                self.fwd_cond_lob = nn.Sequential(
                    nn.Linear(cond_size, augment_dim, dtype=torch.float32),
                    TransformerEncoder(2, augment_dim, 4, 0.1, "only_event", ""))
        if cond_type == "full" and cond_method == "concatenation":
            augment_dim = augment_dim*2
        self.bck_mlp = nn.Sequential(
            nn.Linear(augment_dim, augment_dim//2, dtype=torch.float32),
            nn.Linear(augment_dim//2, input_size, dtype=torch.float32),
        )
        self.v_mlp = nn.Sequential(
            nn.Linear(augment_dim, augment_dim//2, dtype=torch.float32),
            nn.Linear(augment_dim//2, input_size, dtype=torch.float32),
        )
        

    def augment(self, input, cond=None):
        x = self.fwd_mlp(input)
        if self.cond_type == "full":
            cond = self.fwd_cond_lob(cond)
        return x, cond
    
    def deaugment(self, input, v=None):
        if v is not None:
            input = self.bck_mlp(input)
            v = self.v_mlp(v)
            return input, v
        else:
            return self.bck_mlp(input)
