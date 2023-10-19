import math
import torch.nn as nn


def pick_model(config, model_name):
    return "questo è un metodo che chiameremo fuori da NNEngine. NNEngin otterrà gli oggetti già fatti sennò è una merda. Ok"

#noise scheduler taken from "Improved Denoising Diffusion Probabilistic Models"
def noise_scheduler(diffusion_steps, s):
    alphas_dash = []
    f_0 = math.cos((s/(1+s) * (math.pi/2))**2)
    for t in range(1, diffusion_steps+1):
        f_t = math.cos(((t/diffusion_steps+s)/(s+1) * (math.pi/2))**2)
        alphas_dash.append(f_t / f_0)
    betas = [1 - (alphas_dash[i]/alphas_dash[i-1]) for i in range(1, len(alphas_dash))]
    betas.insert(0, 1)
    return alphas_dash, betas