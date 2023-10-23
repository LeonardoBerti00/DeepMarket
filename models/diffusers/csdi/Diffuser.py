from models.diffusers.csdi.Embedder import CSDIEmbeddingDiffusionStep
from models.diffusers.csdi.Residuals import ResidualBlock
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class CSDIEpsilon(nn.Module):
    
    def __init__(self, num_steps=100, embedding_dim=128, side_dim=145, n_heads=2, input_dim=2, layers=1):
        super().__init__()
        
        # TODO: maybe make it a parameter
        self.channels = 1
        self.num_steps = num_steps
        self.embedding_dim = embedding_dim
        self.side_dim = side_dim
        self.n_heads = n_heads
        
        self.diffusion_embedding = CSDIEmbeddingDiffusionStep(
            num_steps=self.num_steps,
            embedding_dim=self.embedding_dim,
        )
        
        

        self.input_projection = self.Conv1d_with_init(input_dim, self.channels, 1)
        self.output_projection1 = self.Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = self.Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=self.side_dim,
                    channels=self.channels,
                    diffusion_embedding_dim=self.embedding_dim,
                    nheads=self.n_heads,
                )
                for _ in range(layers)
            ]
        )
        
    def Conv1d_with_init(self, in_channels, out_channels, kernel_size):
        layer = nn.Conv1d(in_channels, out_channels, kernel_size)
        nn.init.kaiming_normal_(layer.weight)
        return layer


    def forward(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x
