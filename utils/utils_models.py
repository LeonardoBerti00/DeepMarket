from models.diffusers.GaussianDiffusion import GaussianDiffusion
from models.diffusers.CSDI.CSDI import CSDIDiffuser
import constants as cst
import torch.nn as nn

from models.feature_augmenters.MLPAugmenter import MLPAugmenter

def pick_diffuser(config, model_name, augmenter):
    if model_name == "CDT":
        return GaussianDiffusion(config, augmenter).to(cst.DEVICE, non_blocking=True)
    elif model_name == 'CSDI':
        return CSDIDiffuser(config, augmenter).to(cst.DEVICE, non_blocking=True)
    else:
        raise ValueError("Diffuser not found")

def pick_augmenter(augmenter_name, input_size, augment_dim, cond_size, cond_type, cond_augmenter, cond_method, chosen_model):
    if augmenter_name == 'MLP':
        return MLPAugmenter(input_size, augment_dim, cond_size, cond_type, cond_augmenter, cond_method, chosen_model).to(cst.DEVICE, non_blocking=True)
    else:
        raise ValueError("Augmenter not found")

