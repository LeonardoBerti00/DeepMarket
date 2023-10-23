from models.diffusers.StandardDiffusion import StandardDiffusion
from models.diffusers.csdi.CSDI import CSDIDiffuser


def pick_diffuser(config, model_name):
    if model_name == "DDPM":
        return StandardDiffusion(config)
    elif model_name == 'CSDI':
        return CSDIDiffuser(config)
    return "fai un diffuser al posto di cazzeggiare"