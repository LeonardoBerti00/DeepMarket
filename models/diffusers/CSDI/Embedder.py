from models.feature_augmenters.AbstractAugmenter import AugmenterAB
import torch.nn as nn
import torch
import torch.nn.functional as F

class CSDIEmbeddingDiffusionStep(AugmenterAB, nn.Module):
    
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super(CSDIEmbeddingDiffusionStep, self).__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table
    
    def augment(self, input: torch.Tensor):
        return super().augment(input)
    
    def deaugment(self, input: torch.Tensor):
        return input

