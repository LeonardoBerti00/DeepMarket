from config import Configuration
from models.augmenters.AbstractAugmenter import AugmenterAB
from models.diffusers.DiffusionModel import DiffusionAB
import torch
from einops import rearrange
import lightning as L
from constants import LearningHyperParameter
from torch import nn
from torch.nn import functional as F
import math
import time
import constants as cst
from utils import pick_model

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class NNEngine(L.LightningModule):
    
    def __init__(self, diffuser: DiffusionAB, config: Configuration, augmenter: AugmenterAB = None):
        super().__init__()
        """
        This is the skeleton of the diffusion models.

        Parameters:

        """
        self.diffuser = diffuser
        self.augmenter = augmenter
        self.lr = config.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE]
        self.optimizer = config.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER]
        self.momentum = config.HYPER_PARAMETERS[LearningHyperParameter.MOMENTUM]
        self.training = config.IS_TRAINING
        self.conditional_dropout = config.HYPER_PARAMETERS[LearningHyperParameter.CONDITIONAL_DROPOUT]
        self.batch_size = config.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE]
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.T = config.HYPER_PARAMETERS[LearningHyperParameter.COND_BACKWARD_WINDOW_SIZE]
        
        """IS_AUGMENTATION = config.IS_AUGMENTATION

        if (IS_AUGMENTATION and self.T!=0):
            self.lstm = nn.LSTM(x_size, latent_dim, num_layers=1, batch_first=True, dropout=dropout)
            self.diffusion_model = pick_model(config, config.CHOSEN_MODEL)"""

    def forward(self, x):
        if self.augmenter and self.T != 0:
            x = self.augmenter.augment(x)
        recon = self.diffuser(x)
        return recon

    def training_step(self, x, batch_idx):
        recon = self.forward(x)
        loss = self.loss(x, recon)
        self.log('train_loss', loss)
        self.train_losses.append(loss)
        return loss

    def validation_step(self, x, batch_idx):
        recon = self.forward(x)
        loss = self.loss(x, recon)
        self.log('val_loss', loss)
        self.val_losses.append(loss)
        self.val_snr.append(self.si_snr(x, recon))
        return loss

    def test_step(self, x, batch_idx):
        recon = self.forward(x)
        loss = self.loss(x, recon)
        self.log('test_loss', loss)
        self.test_losses.append(loss)
        return loss

    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        return self.optimizer

    def loss(self, input, recon, **kwargs):
        # Reconstruction loss is the mse between the input and the reconstruction
        return self.diffuser.loss(input, recon, **kwargs)

    def on_train_epoch_end(self) -> None:
        loss = sum(self.train_losses) / len(self.train_losses)
        print(f"\n train loss {loss}\n")

    def on_validation_epoch_end(self) -> None:
        loss = sum(self.val_losses) / len(self.val_losses)
        print(f"\n val loss {loss}")

    def on_test_epoch_end(self) -> None:
        loss = sum(self.test_losses) / len(self.test_losses)
        print(f"\n test loss {loss}")

    def inference_time(self, batch):
        x, _ = batch
        t0 = time.time()
        _ = self(x)
        torch.cuda.current_stream().synchronize()
        t1 = time.time()
        elapsed = t1 - t0
        # print("Inference for the model:", elapsed, "ms")
        return elapsed
