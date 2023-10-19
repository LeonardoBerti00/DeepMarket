import math
import torch.nn as nn


def pick_diffuser(config, model_name):
    return "fai un diffuser al posto di cazzeggiare"

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

#formula taken from "Denoising Diffusion Probabilistic Models"
def compute_mean_tilde_t(x_0, x_T, alpha_dash_t, alpha_dash_t_1, beta_t, alpha_t):
    # alpha_dash_t_1 is alpha_dash(t-1) 
    return math.sqrt(alpha_dash_t_1)*beta_t*x_0 / (1-alpha_dash_t) + math.sqrt(alpha_t)*(1-alpha_dash_t_1)*x_T / (1-alpha_dash_t)