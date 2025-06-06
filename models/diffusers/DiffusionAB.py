from abc import ABC, abstractmethod
from typing import Dict, Tuple
import torch

import constants as cst


class DiffusionAB(ABC):
    """An abstract class for loss functions."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.betas = config.BETAS
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0, dtype=torch.float32)
        self.gen_seq_size = config.HYPER_PARAMETERS[cst.LearningHyperParameter.MASKED_SEQ_SIZE]
        self.SEQ_size = config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE]
        self.cond_seq_size = self.SEQ_size - self.gen_seq_size

    @abstractmethod
    def loss(self, true: torch.Tensor, recon: torch.Tensor, **kwargs):
        """Computes the loss given the true and predicted values."""
        pass

    def forward_reparametrized(self, x_0: torch.Tensor, t:  torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Reparametrized forward diffusion process, takes in input x_0 and returns x_t after t steps of noise
        x_t(x_0, ϵ) = √(α̅_t)x_0 + √(1 - α̅_t)ϵ
        """
        noise = torch.distributions.normal.Normal(0, 1).sample(x_0.shape).to(cst.DEVICE, non_blocking=True)
        first_term = torch.einsum('bld, b -> bld', x_0, torch.sqrt(self.alphas_cumprod[t]))
        second_term = torch.einsum('bld, b -> bld', noise, torch.sqrt(1 - self.alphas_cumprod[t]))
        x_t = first_term + second_term
        return x_t, noise

    def forward_process(self, x_0: torch.Tensor, t: torch.Tensor):
        # Standard forward process, takes in input x_0 and returns x_t after t steps of noise
        cov_matrix = torch.eye(x_0.shape)
        mean = torch.mul(x_0, torch.sqrt(self.alphas_cumprod[t]))
        std = torch.mul(cov_matrix, torch.sqrt(1 - self.alphas_cumprod[t]))
        x_T = torch.distributions.Normal(mean, std).rsample().to(cst.DEVICE, non_blocking=True)
        return x_T, {'mean': mean, 'std': std}