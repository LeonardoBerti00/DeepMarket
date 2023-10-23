import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Mlp
from utils.utils import sinusoidal_positional_embedding

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=1000):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: (N) Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ConditionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, cond_seq_len, cond_size, dropout_prob, cond_hidden_size=256):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.mlp = nn.Sequential(
            nn.Linear(cond_seq_len*cond_size, cond_hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(cond_hidden_size, cond_hidden_size, bias=True),
        )
        self.dropout_prob = dropout_prob

    def token_drop(self, cond, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(cond.shape[0], device=cond.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        for i in range(cond.shape[0]):
            if drop_ids[i]:
                cond[i] = torch.zeros((cond.shape[1], cond.shape[2]), device=cond.device)
        return cond

    def forward(self, cond, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            cond = self.token_drop(cond, force_drop_ids)
        cond = cond.flatten(1)
        embeddings = self.mlp(cond)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class adaLN_Zero(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, bias=True, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(               #it's the MLP in the figure 3 of the paper
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer_adaLN_Zero(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, input_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, 2 * input_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        # Zero-out output layers:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size,
        cond_seq_len,
        hidden_size,     #both of cond and x
        cond_size,
        diffusion_steps,
        depth,
        num_heads,
        token_sequence_size,
        mlp_ratio,
        cond_dropout_prob,
        cond_type
    ):
        super().__init__()
        self.num_heads = num_heads
        self.t_embedder = TimestepEmbedder(hidden_size, diffusion_steps)
        self.c_embedder = ConditionEmbedder(cond_seq_len, cond_size, cond_dropout_prob, hidden_size)

        self.pos_embed = sinusoidal_positional_embedding(token_sequence_size, hidden_size)
        if (cond_type == 'adaln_zero'):
            self.blocks = nn.ModuleList([
                adaLN_Zero(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
            ])
            self.final_layer = FinalLayer_adaLN_Zero(hidden_size, input_size)
            self.initialize_weights()
        elif (cond_type == 'crossattention'):
            pass
        elif (cond_type == 'concatenate'):
            pass
        else:
            raise ValueError(f'Unknown type: {cond_type}')


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)


    def forward(self, x, cond, t):
        """
        Forward pass of DiT.
        x: (N, K, F) tensor of time series
        t: (N,) tensor of diffusion timesteps
        cond: (N, P, C) tensor of past history
        """
        x[:] = x[:] + self.pos_embed
        t = self.t_embedder(t)
        cond = self.c_embedder(cond, self.training)
        #concatentate t and cond
        c = torch.cat([cond, t], dim=1)
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)



