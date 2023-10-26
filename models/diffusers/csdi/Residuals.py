
import torch.nn as nn
import torch
import math

from utils.utils import Conv1d_with_init

class ResidualBlock(nn.Module):
    
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super(ResidualBlock, self).__init__()
        
        print(f'diffusion_embedding_dim = {diffusion_embedding_dim}')
        print(f'nheads = {nheads}')
        
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=channels, nhead=nheads, dim_feedforward=64, activation="gelu"), num_layers=1)
        
        self.feature_layer = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=channels, nhead=nheads, dim_feedforward=64, activation="gelu"), num_layers=1)
        
    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x: torch.Tensor, cond_info: torch.Tensor, diffusion_emb):
        B, channel, K, L = x.shape
        print(f'x.shape = {x.shape}')
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)
        
        print(f'cond_info.shape={cond_info.shape}')

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        cond_info = cond_info.permute(0,3,2,1)
        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, -1)
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