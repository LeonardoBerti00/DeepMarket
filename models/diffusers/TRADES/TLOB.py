from torch import nn
import torch
from einops import rearrange
import constants as cst
from models.diffusers.TRADES.bin import BiN
from models.diffusers.TRADES.mlp import MLP
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ComputeQKV(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.q = nn.Linear(hidden_dim, hidden_dim*num_heads)
        self.k = nn.Linear(hidden_dim, hidden_dim*num_heads)
        self.v = nn.Linear(hidden_dim, hidden_dim*num_heads)
        
    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        return q, k, v


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, final_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(hidden_dim)
        self.qkv = ComputeQKV(hidden_dim, num_heads)
        self.attention = nn.MultiheadAttention(hidden_dim*num_heads, num_heads, batch_first=True, device=cst.DEVICE)
        self.mlp = MLP(hidden_dim, hidden_dim*4, final_dim)
        self.w0 = nn.Linear(hidden_dim*num_heads, hidden_dim)
        
    def forward(self, x):
        res = x
        q, k, v = self.qkv(x)
        x, att = self.attention(q, k, v, average_attn_weights=False, need_weights=True)
        if torch.isnan(x).any():
            print("after att:", x.max())
        x = self.w0(x)
        x = x + res
        x = self.norm(x)
        x = self.mlp(x)
        if torch.isnan(x).any():
            print("after mlp:", x.max())
        if x.shape[-1] == res.shape[-1]:
            x = x + res
        return x, att


class TransformerLOB(nn.Module):
    def __init__(self, 
                 hidden_dim: int,
                 num_layers: int,
                 seq_size: int,
                 num_heads: int,
                 ) -> None:
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_size = seq_size
        self.num_heads = num_heads
        self.layers = nn.ModuleList()
        self.first_branch = nn.ModuleList()
        self.second_branch = nn.ModuleList()
        self.order_type_embedder = nn.Embedding(3, 1)
        for i in range(num_layers):
            if i != num_layers-1:
                self.layers.append(TransformerLayer(hidden_dim, num_heads, hidden_dim))
                self.layers.append(TransformerLayer(seq_size, num_heads, seq_size))
            else:
                self.layers.append(TransformerLayer(hidden_dim, num_heads, hidden_dim))
                self.layers.append(TransformerLayer(seq_size, num_heads, seq_size))
        
    
    def forward(self, x):
        for i in range(len(self.layers)):
            x, _ = self.layers[i](x)
            x = x.permute(0, 2, 1)
        return x

    
    
