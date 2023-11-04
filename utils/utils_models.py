from models.diffusers.GaussianDiffusion import GaussianDiffusion
from models.diffusers.csdi.CSDI import CSDIDiffuser


def pick_diffuser(config, model_name):
    if model_name == "DiT":
        return GaussianDiffusion(config)
    elif model_name == 'CSDI':
        return CSDIDiffuser(config)
    else:
        raise ValueError("Diffuser not found")
    return "fai un diffuser al posto di cazzeggiare"