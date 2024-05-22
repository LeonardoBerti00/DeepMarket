from abc import ABC, abstractmethod
import importlib
from os.path import isfile, basename
import sys
from lightning import LightningModule
import torch

from configuration import Configuration
from constants import LearningHyperParameter


class NNEngine(LightningModule, ABC):
    
    
    def __init__(self, config: Configuration):
        super().__init__()
        self.config: Configuration = config
        self.IS_WANDB = config.IS_WANDB
        self.IS_DEBUG = config.IS_DEBUG
        self.lr = config.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE]
        self.optimizer = config.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER]
        self.training = config.IS_TRAINING
        self.test_batch_size = config.HYPER_PARAMETERS[LearningHyperParameter.TEST_BATCH_SIZE]
        self.epochs = config.HYPER_PARAMETERS[LearningHyperParameter.EPOCHS]
        self.chosen_stock = config.CHOSEN_STOCK.name
        self.chosen_stock = config.CHOSEN_STOCK.name
        self.train_losses, self.vlb_train_losses, self.simple_train_losses = [], [], []
        self.val_ema_losses, self.test_ema_losses = [], []
        self.min_loss_ema = 10000000
        self.filename_ckpt = config.FILENAME_CKPT
        self.num_violations_price = 0
        self.num_violations_size = 0
        self.chosen_model = config.CHOSEN_MODEL.name
        self.last_path_ckpt_ema = None
        
        
    @abstractmethod
    def sample(self, **kwargs) -> torch.Tensor:
        pass
        
    @abstractmethod
    def model_checkpointing(self, loss) -> None:
        pass
    
    @classmethod
    def factory(cls, class_path: str, config: Configuration):
        # Split the class_path into the module path and class name
        module_path, class_name = class_path.rsplit('.', 1)
        # Convert module path to file path if necessary
        module_file_path = module_path.replace('.', '/') + '.py'
        # Ensure the module file exists
        if not isfile(module_file_path):
            raise FileNotFoundError(f"Module file {module_file_path} does not exist")
        # Load the class from the module file path
        cls = NNEngine.load_class_from_path(module_file_path, class_name)
        # Ensure the loaded class is a subclass of AbstractClass
        if not issubclass(cls, NNEngine):
            raise TypeError(f"{class_name} is not a subclass of AbstractClass")
        
        return cls(config)
    
    @classmethod
    def load_class_from_path(cls, module_path: str, class_name: str):
        module_name = basename(module_path).replace('.py', '')
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return getattr(module, class_name)
