from models.diffusers.GaussianDiffusion import GaussianDiffusion
from models.diffusers.csdi.CSDI import CSDIDiffuser
import constants as cst

def pick_diffuser(config, model_name, augmenter):
    if model_name == "CDT":
        return GaussianDiffusion(config, augmenter).to(cst.DEVICE, non_blocking=True)
    elif model_name == 'CSDI':
        return CSDIDiffuser(config, augmenter).to(cst.DEVICE, non_blocking=True)
    else:
        raise ValueError("Diffuser not found")
    return "fai un diffuser al posto di cazzeggiare"