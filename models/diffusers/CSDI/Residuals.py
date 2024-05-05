
from einops import rearrange
import torch.nn as nn
import torch
import math

from utils.utils import Conv1d_with_init

class ResidualBlock(nn.Module):
    
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super(ResidualBlock, self).__init__()
                
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=channels, nhead=nheads, dim_feedforward=64, activation="gelu", batch_first=True), num_layers=1)
        
        self.feature_layer = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=channels, nhead=nheads, dim_feedforward=64, activation="gelu", batch_first=True), num_layers=1)
        
    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.contiguous()
        y = rearrange(y, 'B C (K L) -> (B K) L C', K=K, L=L)
        y = self.time_layer(y)
        y = rearrange(y, '(B K) L C -> B K C L', B=B, K=K, L=L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = rearrange(y, 'B K C L -> (B L) K C', K=K, L=L)
        y = self.feature_layer(y)
        y = rearrange(y, '(B L) K C -> B C (K L)', B=B, L=L, K=K)
        return y

    def forward(self, x: torch.Tensor, cond_info: torch.Tensor, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)
        
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        C = cond_info.shape[-1]
        cond_info = cond_info.contiguous().view(B, C, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip