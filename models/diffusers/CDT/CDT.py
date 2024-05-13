import torch
import torch.nn as nn
from einops import rearrange
from utils.utils import sinusoidal_positional_embedding
import constants as cst
import random
from models.diffusers.CDT.Transformer import TransformerEncoder

class CDT(nn.Module):
    def __init__(
        self,
        input_size,
        cond_seq_len,
        num_diffusionsteps,
        depth,
        num_heads,
        masked_sequence_size,
        cond_dropout_prob,
        is_augmented,
        dropout,
        cond_type,
        cond_method
    ):
        super().__init__()
        self.cond_dropout_prob = cond_dropout_prob
        self.num_heads = num_heads
        if cond_method == 'concatenation' and cond_type == 'full' and is_augmented:
            input_size = input_size*2
            output_size = input_size
        elif cond_method == 'concatenation' and cond_type == 'full' and not is_augmented:
            output_size = input_size
            input_size = input_size + cst.N_LOB_LEVELS * cst.LEN_LEVEL
        elif cond_method == "crossattention":
            output_size = input_size
        self.t_embedder = sinusoidal_positional_embedding(num_diffusionsteps, input_size) #TimestepEmbedder(input_size, input_size//4, num_diffusionsteps)
        self.seq_size = masked_sequence_size + cond_seq_len
        self.pos_embed = sinusoidal_positional_embedding(self.seq_size, input_size)
        self.is_augmented = is_augmented
        self.cond_method = cond_method
        self.cond_type = cond_type
        self.layers = TransformerEncoder(num_heads, input_size, depth, dropout, cond_type, cond_method)
        #self.layers = nn.LSTM(input_size, input_size, 2, batch_first=True, dropout=dropout)
        self.fc_noise = nn.Linear(input_size*self.seq_size, output_size, device=cst.DEVICE)
        self.fc_var = nn.Linear(input_size*self.seq_size, output_size, device=cst.DEVICE)
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, x, cond_orders, t, cond_lob=None):
        """
        Forward pass of CDT.
        x: (N, K, F) tensor of time series
        t: (N,) tensor of diffusion timesteps
        cond_orders: (N, P, C) tensor of past history
        """
        cond_orders = self.token_drop(cond_orders)
        full_input = torch.cat([cond_orders, x], dim=1)
        if self.cond_method == 'concatenation' and self.cond_type == 'full':
            full_input = torch.cat([full_input, cond_lob], dim=-1)
        full_input = full_input.add(self.pos_embed)
        diff_time_emb = self.t_embedder[t]
        full_input = full_input.add(diff_time_emb.view(diff_time_emb.shape[0], 1, diff_time_emb.shape[1]))
        full_input = self.layer_norm(full_input)
        full_input = self.layers(full_input, mask=None, cond=cond_lob) 
        full_input = rearrange(full_input, 'n l f -> n (l f)')
        noise = self.fc_noise(full_input)
        var = self.fc_var(full_input)
        noise = rearrange(noise, 'n l -> n 1 l')
        var = rearrange(var, 'n l -> n 1 l')
        return noise, var

    def token_drop(self, cond_orders):
        rand = random.random()
        if rand < self.cond_dropout_prob:
            # create a mask of zeros for the rows to drop
            mask = torch.zeros((cond_orders.shape), device=cond_orders.device)
            cond_orders = torch.einsum('bld, bld -> bld', cond_orders, mask)
            return cond_orders
        else:
            # no tokens are dropped
            return cond_orders 
        
 