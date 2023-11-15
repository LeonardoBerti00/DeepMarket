import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import Mlp, Attention
from models.diffusers.DiT.Embedders import TimestepEmbedder, ConditionEmbedder
from utils.utils import sinusoidal_positional_embedding
import constants as cst

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class adaLN_Zero(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, input_size, num_heads, cond_len, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(input_size, elementwise_affine=False, eps=1e-6)
        if input_size == cst.LEN_EVENT: num_heads = 1
        self.attn = nn.MultiheadAttention(input_size, num_heads=num_heads, bias=True, batch_first=True)
        self.norm2 = nn.LayerNorm(input_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(input_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=input_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(               #it's the MLP in the figure 3 of the paper
            nn.SiLU(),
            nn.Linear(input_size*cond_len, 6 * input_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        normalized_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_output, _ = self.attn(normalized_x, normalized_x, normalized_x)
        gate_msa = rearrange(gate_msa, 'n d -> n 1 d')
        gated_attn_output = torch.einsum('n l d, n l d -> n l d', attn_output, gate_msa)
        x = x + gated_attn_output
        normalized_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        gate_mlp = rearrange(gate_mlp, 'n d -> n 1 d')
        x = x + gate_mlp * self.mlp(normalized_x)
        return x


class FinalLayer_adaLN_Zero(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, input_size, cond_size, input_seq_len):
        super().__init__()
        self.input_size = input_size
        self.norm_final = nn.LayerNorm(input_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(input_size, 2 * input_size, bias=True)
        self.input_seq_len = input_seq_len
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(input_size*cond_size, 2 * input_size, bias=True)
        )
        # Zero-out output layers:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        # in the case of concatenated conditioning, we need to take only the last part of the sequence that was diffused
        # TODO reflect on the possibility to use an mlp instead of simply take the last part of the sequence
        if (x.shape[1] != self.input_seq_len):
            x = x[:, -self.input_seq_len:, :]
        out = self.linear(x)
        out = rearrange(out, 'n l (c m) -> n l c m', c=2, m=self.input_size)
        # it gives in output the noise and the variances
        noise, var = out[:, :, 0], out[:, :, 1]
        return noise, var


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size,
        cond_seq_len,
        cond_size,
        num_diffusionsteps,
        depth,
        num_heads,
        token_sequence_size,
        mlp_ratio,
        cond_dropout_prob,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.t_embedder = TimestepEmbedder(input_size, input_size//4, num_diffusionsteps)
        self.c_embedder = ConditionEmbedder(cond_seq_len, cond_size, cond_dropout_prob, input_size)
        if (token_sequence_size != 1):
            self.pos_embed = sinusoidal_positional_embedding(token_sequence_size, input_size)
        self.blocks = nn.ModuleList([
            adaLN_Zero(input_size, num_heads, cond_seq_len+1, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer_adaLN_Zero(input_size, cond_seq_len+1, token_sequence_size)
        self.initialize_weights()



    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

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

        if (x.shape[1] != 1):
            x = x + self.pos_embed
        t = self.t_embedder(t)
        t = rearrange(t, 'n d -> n 1 d')
        cond = self.c_embedder(cond, self.training)
        #concatentate t and cond
        c = torch.cat([cond, t], dim=1)
        # flatten c to give it in input to the mlp
        c = rearrange(c, 'n p c -> n (p c)')
        for block in self.blocks:
            x = block(x, c)
        noise, var = self.final_layer(x, c)
        return noise, var

    #TODO: implement forward_with_cfg
    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        noise, rest = model_out[:, :3], model_out[:, 3:]
        cond_noise, uncond_noise = torch.split(noise, len(noise) // 2, dim=0)
        half_noise = uncond_noise + cfg_scale * (cond_noise - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class CDT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size,
        cond_seq_len,
        cond_size,
        num_diffusionsteps,
        depth,
        num_heads,
        masked_sequence_size,
        mlp_ratio,
        cond_dropout_prob,
    ):
        super().__init__()
        assert cond_size == input_size
        self.cond_dropout_prob = cond_dropout_prob
        self.num_heads = num_heads
        self.t_embedder = TimestepEmbedder(input_size, input_size//4, num_diffusionsteps)
        self.seq_size = masked_sequence_size + cond_seq_len
        self.pos_embed = sinusoidal_positional_embedding(self.seq_size, input_size)
        self.blocks = nn.ModuleList([
            adaLN_Zero(input_size, num_heads, 1, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer_adaLN_Zero(input_size, 1, masked_sequence_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

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
        cond = self.token_drop(cond)
        full_input = torch.cat([cond, x], dim=1)
        full_input = full_input.add(self.pos_embed)
        t = self.t_embedder(t)
        for block in self.blocks:
            full_input = block(full_input, t)
        noise, var = self.final_layer(full_input, t)
        return noise, var

    def token_drop(self, cond, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(cond.shape[0], device=cond.device) < self.cond_dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        #create a mask of zeros for the rows to drop
        mask = torch.ones((cond.shape), device=cond.device)
        mask[drop_ids] = 0
        cond = torch.einsum('bld, bld -> bld', cond, mask)
        return cond

    #TODO: implement forward_with_cfg
    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        noise, rest = model_out[:, :3], model_out[:, 3:]
        cond_noise, uncond_noise = torch.split(noise, len(noise) // 2, dim=0)
        half_noise = uncond_noise + cfg_scale * (cond_noise - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
