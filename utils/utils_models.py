from models.diffusers.GaussianDiffusion import GaussianDiffusion
from models.diffusers.CSDI.CSDI import CSDIDiffuser
import constants as cst
import torch.nn as nn

from models.feature_augmenters.LSTMAugmenter import LSTMAugmenter
from models.feature_augmenters.MLPAugmenter import MLPAugmenter

def pick_diffuser(config, model_name, augmenter):
    if model_name == "CDT":
        return GaussianDiffusion(config, augmenter).to(cst.DEVICE, non_blocking=True)
    elif model_name == 'CSDI':
        return CSDIDiffuser(config, augmenter).to(cst.DEVICE, non_blocking=True)
    else:
        raise ValueError("Diffuser not found")
    return "fai un diffuser al posto di cazzeggiare"

def pick_augmenter(config, augmenter_name, input_size):
    if augmenter_name == "LSTM":
        return LSTMAugmenter(config, input_size).to(cst.DEVICE, non_blocking=True)
    elif augmenter_name == 'MLP':
        return MLPAugmenter(config, input_size).to(cst.DEVICE, non_blocking=True)
    else:
        raise ValueError("Diffuser not found")
    return "fai un diffuser al posto di cazzeggiare"