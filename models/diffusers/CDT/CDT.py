import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import Mlp, Attention
from models.diffusers.CDT.Embedders import TimestepEmbedder, ConditionEmbedder
from utils.utils import sinusoidal_positional_embedding
import constants as cst
import random


class CDT(nn.Module):
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
        is_augmented,
        dropout
    ):
        super().__init__()
        assert cond_size == input_size
        self.cond_dropout_prob = cond_dropout_prob
        self.num_heads = num_heads
        self.t_embedder = sinusoidal_positional_embedding(num_diffusionsteps, input_size) #TimestepEmbedder(input_size, input_size//4, num_diffusionsteps)
        self.seq_size = masked_sequence_size + cond_seq_len
        self.pos_embed = sinusoidal_positional_embedding(self.seq_size, input_size)
        self.is_augmented = is_augmented
        if is_augmented:
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=input_size*mlp_ratio, dropout=dropout, batch_first=True) for _ in range(depth)
            ]) 
        else:
            self.layers = nn.ModuleList([
                nn.LSTM(input_size, input_size, 2, batch_first=True, dropout=dropout, bidirectional=False)
            ])  
        self.fc_noise = nn.Linear(input_size*self.seq_size, input_size, device=cst.DEVICE)
        self.fc_var = nn.Linear(input_size*self.seq_size, input_size, device=cst.DEVICE)


    def forward(self, x, cond, t):
        """
        Forward pass of CDT.
        x: (N, K, F) tensor of time series
        t: (N,) tensor of diffusion timesteps
        cond: (N, P, C) tensor of past history
        """
        cond = self.token_drop(cond)
        full_input = torch.cat([cond, x], dim=1)
        full_input = full_input.add(self.pos_embed)
        diff_time_emb = self.t_embedder[t]
        full_input = full_input.add(diff_time_emb.view(diff_time_emb.shape[0], 1, diff_time_emb.shape[1]))
        for layer in self.layers:
            if self.is_augmented:
                full_input = layer(full_input)
            else:
                full_input, _ = layer(full_input)
        full_input = rearrange(full_input, 'n l f -> n (l f)')
        noise = self.fc_noise(full_input)
        var = self.fc_var(full_input)
        noise = rearrange(noise, 'n l -> n 1 l')
        var = rearrange(var, 'n l -> n 1 l')
        return noise, var

    def token_drop(self, cond):
        rand = random.random()
        if rand < self.cond_dropout_prob:
            # create a mask of zeros for the rows to drop
            mask = torch.zeros((cond.shape), device=cond.device)
            cond = torch.einsum('bld, bld -> bld', cond, mask)
            return cond
        else:
            # no tokens are dropped
            return cond 
        
 