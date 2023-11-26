import math
import numpy as np
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





def unnormalize(x, mean, std):
    return x * std + mean

def to_original_lob(event_and_lob, seq_size):
    lob = event_and_lob[:, cst.LEN_EVENT:]

    lob[:, 0::2] = unnormalize(lob[:, 0::2], cst.TSLA_LOB_MEAN_PRICE_10, cst.TSLA_LOB_STD_PRICE_10)
    lob[:, 1::2] = unnormalize(lob[:, 1::2], cst.TSLA_LOB_MEAN_SIZE_10, cst.TSLA_LOB_STD_SIZE_10)
    lob = lob[seq_size - 2:, :]
    #assert (generated_events.shape[0]+1 == lob.shape[0])
    # round price and size

    lob[:, 0::2] = np.around(lob[:, 0::2], decimals=0)
    lob[:, 1::2] = np.around(lob[:, 1::2], decimals=0)

    return lob

def check_constraints(file_recon, file_lob, seq_size):
    generated_events = np.load(file_recon)
    event_and_lob = np.load(file_lob)
    lob = to_original_lob(event_and_lob, seq_size)
    print()
    print("numbers of lob ", lob.shape[0])
    print("numbers of gen events ", generated_events.shape[0])
    num_violations_price_del = 0
    num_violations_price_exec = 0
    num_violations_size = 0
    num_non_violations_price_del = 0
    num_non_violations_price_exec = 0
    num_add = 0
    for i in range(generated_events.shape[0]):
        price = generated_events[i, 3]
        size_event = generated_events[i, 2]
        type = generated_events[i, 1]
        if (type == 2 or type == 3 or type == 4):    #it is a cancellation
            #take the index of the lob with the same value of the price
            index = np.where(lob[i, :] == price)[0]
            if (index.shape[0] == 0):
                if (type == 2 or type == 3):
                    num_violations_price_del += 1
                else:
                    num_violations_price_exec += 1
            else:
                size_limit_order = lob[i, index[0] + 1]
                if (size_limit_order < size_event):
                    num_violations_size += 1
                else:
                    if (type == 2 or type == 3):
                        num_non_violations_price_del += 1
                    else:
                        num_non_violations_price_exec += 1
        else:
            num_add += 1
    print("number of violations for price deletion ", num_violations_price_del)
    print("number of violations for price execution ", num_violations_price_exec)
    print("number of violations for size ", num_violations_size)
    print("number of non violations for price deletion ", num_non_violations_price_del)
    print("number of non violations for price execution ", num_non_violations_price_exec)
    print("number of add orders ", num_add)
    print()


















