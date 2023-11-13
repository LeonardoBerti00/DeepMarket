import math
import torch
from matplotlib import pyplot as plt

import constants as cst

#noise scheduler taken from "Improved Denoising Diffusion Probabilistic Models"
def noise_scheduler(num_diffusionsteps, s):
    alphas_cumprod = []
    f_0 = math.cos((s/(1+s) * (math.pi/2)))**2
    for t in range(1, num_diffusionsteps+1):
        f_t = math.cos(((t/num_diffusionsteps+s)/(s+1) * (math.pi/2)))**2
        alphas_cumprod.append(f_t / f_0)
    betas = [1 - (alphas_cumprod[i]/alphas_cumprod[i-1]) for i in range(1, len(alphas_cumprod))]
    betas.insert(0, betas[0])
    #plot the alphas with respect to t
    plt.plot(alphas_cumprod)
    plt.show()
    return torch.Tensor(alphas_cumprod).to(cst.DEVICE), torch.clamp(torch.Tensor(betas).to(cst.DEVICE), max=0.999)

#formula taken from "Denoising Diffusion Probabilistic Models"
def compute_mean_tilde_t(x_0, x_T, alpha_cumprod_t, alpha_cumprod_t_1, beta_t, alpha_t):
    # alpha_cumprod_t_1 is alpha_cumprod(t-1)
    return torch.sqrt(alpha_cumprod_t_1)*beta_t*x_0 / (1-alpha_cumprod_t) + torch.sqrt(alpha_t)*(1-alpha_cumprod_t_1)*x_T / (1-alpha_cumprod_t)


def sinusoidal_positional_embedding(token_sequence_size, token_embedding_dim, n=10000.0):

    if token_embedding_dim % 2 != 0:
        raise ValueError("Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={:d})".format(token_embedding_dim))

    T = token_sequence_size
    d = token_embedding_dim

    positions = torch.arange(0, T).unsqueeze_(1)
    embeddings = torch.zeros(T, d)

    denominators = torch.pow(n, 2*torch.arange(0, d//2)/d) # 10000^(2i/d_model), i is the index of embedding
    embeddings[:, 0::2] = torch.sin(positions/denominators) # sin(pos/10000^(2i/d_model))
    embeddings[:, 1::2] = torch.cos(positions/denominators) # cos(pos/10000^(2i/d_model))

    return embeddings.to(cst.DEVICE)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = torch.nn.Conv1d(in_channels, out_channels, kernel_size)
    torch.nn.init.kaiming_normal_(layer.weight)
    return layer


def wandb_init():
    #wandb.login("d29d51017f4231b5149d36ad242526b374c9c60a")
    sweep_config = {
        'method': 'random',
        'metric': {
            'goal': 'minimize',
            'name': 'validation_loss'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 15,
            'eta': 2
        },
        'run_cap': 4
    }

    parameters_dict = {
        'epochs': {
            'value': 50
        },
        'optimizer': {
            'values': ['adam', 'sgd', 'lion']
        },
        'dropout': {
            'values': [0.1, 0.2]
        },
        'conditional_dropout': {
            'values': [0.1, 0.2]
        },
        'lr': {
            'distribution': 'uniform',
            'max': 0.01,
            'min': 0.0001,
        },
        'batch_size': {
            'values': [64, 128, 256]
        },
    }

    sweep_config['parameters'] = parameters_dict

    return sweep_config