from config import Configuration
from constants import CSDIParameters, LearningHyperParameter
from models.diffusers.csdi.Embedder import CSDIEmbeddingDiffusionStep
from models.diffusers.csdi.Residuals import ResidualBlock
from models.diffusers.csdi.utils import Conv1d_with_init
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class DiffCSDI(nn.Module):
    
    def __init__(self, config: Configuration, inputdim=2):
        super().__init__()
        
        # TODO: maybe make it a parameter
        self.channels = 1
        self.num_steps = config.HYPER_PARAMETERS[LearningHyperParameter.DIFFUSION_STEPS]
        self.embedding_dim = config.CSDI_HYPERPARAMETERS[CSDIParameters.DIFFUSION_STEP_EMB_DIM]
        self.side_dim = config.CSDI_HYPERPARAMETERS[CSDIParameters.SIDE_DIM]
        self.n_heads = config.CSDI_HYPERPARAMETERS[CSDIParameters.N_HEADS]
        
        self.diffusion_embedding = CSDIEmbeddingDiffusionStep(
            num_steps=self.num_steps,
            embedding_dim=self.embedding_dim,
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=self.side_dim,
                    channels=self.channels,
                    diffusion_embedding_dim=self.embedding_dim,
                    nheads=self.n_heads,
                )
                for _ in range(config["layers"])
            ]
        )

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
