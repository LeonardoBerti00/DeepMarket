import math
import torch
import constants as cst


#noise scheduler taken from "Improved Denoising Diffusion Probabilistic Models"
def noise_scheduler(num_diffusion_timesteps, max_beta=0.99):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    alpha_bar = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32).to(cst.DEVICE, non_blocking=True)


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

    return embeddings.to(cst.DEVICE, non_blocking=True)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = torch.nn.Conv1d(in_channels, out_channels, kernel_size)
    torch.nn.init.kaiming_normal_(layer.weight)
    return layer


