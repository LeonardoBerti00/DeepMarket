from abc import ABC, abstractmethod
import torch

class DiffusionAB(ABC):
    """An abstract class for loss functions."""

    @abstractmethod
    def loss(self, true: torch.Tensor, recon: torch.Tensor, **kwargs) -> torch.Tensor:
        """Computes the loss given the true and predicted values."""
        pass