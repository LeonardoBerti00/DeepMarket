from models.diffusers.GaussianDiffusion import GaussianDiffusion
from models.diffusers.csdi.CSDI import CSDIDiffuser
import constants as cst

def pick_diffuser(config, model_name, augmenter):
    if model_name == "CDT":
        return GaussianDiffusion(config, augmenter).to(device=cst.DEVICE)
    elif model_name == 'CSDI':
        return CSDIDiffuser(config, augmenter).to(device=cst.DEVICE)
    else:
        raise ValueError("Diffuser not found")
    return "fai un diffuser al posto di cazzeggiare"