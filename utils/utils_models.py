from models.diffusers.gaussian_diffusion import GaussianDiffusion
import constants as cst
import torch.nn as nn

from models.feature_augmenters.MLPAugmenter import MLPAugmenter


def pick_augmenter(augmenter_name, input_size, augment_dim, cond_size, cond_type, cond_augmenter, cond_method, chosen_model):
    if augmenter_name == 'MLP':
        return MLPAugmenter(input_size, augment_dim, cond_size, cond_type, cond_augmenter, cond_method, chosen_model).to(cst.DEVICE, non_blocking=True)
    else:
        raise ValueError("Augmenter not found")

