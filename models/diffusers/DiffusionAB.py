from abc import ABC, abstractmethod
from typing import Dict, Tuple
import torch

class DiffusionAB(ABC):
    """An abstract class for loss functions."""

    def __init__(self, config, alphas_dash, betas):
        self.config = config
        self.alphas_dash = alphas_dash
        self.betas = betas

    @abstractmethod
    def loss(self, true: torch.Tensor, recon: torch.Tensor, **kwargs) -> torch.Tensor:
        """Computes the loss given the true and predicted values."""
        pass
    
    def reparametrized_forward(self, x_0: torch.Tensor, diffusion_step: int, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Reparametrization trick for the diffusion process taken from DDPM paper
        eps = torch.distributions.normal.Normal(0, 1).sample(x_0.shape)
        first_term = torch.sqrt(self.alphas_dash[diffusion_step]) * x_0
        second_term = torch.sqrt(1 - self.alphas_dash[diffusion_step]) * eps
        x_t = first_term + second_term
        return x_t, {'eps': eps}

    def forward_process(self, x_0, diffusion_step: int):
        # Standard forward process, takes in input x_0 and returns x_t after t steps of noise
        cov_matrix = torch.eye(self.x_size)
        mean = torch.sqrt(self.alphas_dash[diffusion_step]) * x_0
        std = (1 - self.alphas_dash[diffusion_step]) * cov_matrix
        x_T = torch.distributions.Normal(mean, std).rsample()
        return x_T, {'mean': mean, 'std': std}
    