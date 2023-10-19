from abc import ABC, abstractmethod
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
    