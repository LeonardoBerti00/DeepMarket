from abc import ABC, abstractmethod
from config import Configuration
import torch


class AugmenterAB(ABC):
    
        
    @abstractmethod
    def augment(self, input: torch.Tensor):
        pass
    
    
    @abstractmethod
    def deaugment(self, input: torch.Tensor):
        pass