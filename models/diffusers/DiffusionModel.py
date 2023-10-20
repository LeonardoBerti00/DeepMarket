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
    
    def reparametrized_forward(self, input: torch.Tensor, diffusion_steps: int, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Reparametrization trick for the diffusion process taken from DDPM paper
        eps = torch.distributions.normal.Normal(0, 1).sample(input.shape)
        first_term = torch.sqrt(self.alphas_dash[diffusion_steps]) * input
        second_term = torch.sqrt(1 - self.alphas_dash[diffusion_steps]) * eps
        x_t = first_term + second_term
        return x_t, {'eps': eps}
    