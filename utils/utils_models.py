from models.diffusers.StandardDiffusion import StandardDiffusion


def pick_diffuser(config, model_name):
    if model_name == "DDPM":
        return StandardDiffusion(config)
    return "fai un diffuser al posto di cazzeggiare"