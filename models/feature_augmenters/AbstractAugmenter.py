from abc import ABC, abstractmethod
import configuration
import torch


class AugmenterAB(ABC):
    
        
    @abstractmethod
    def augment(self, input: torch.Tensor):
        pass
    
    
    @abstractmethod
    def deaugment(self, input: torch.Tensor, context: dict):
        pass