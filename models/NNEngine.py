from config import Configuration
from models.feature_augmenters.AbstractAugmenter import AugmenterAB
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
from utils import pick_diffuser, noise_scheduler
from models.feature_augmenters.LSTMAugmenter import LSTMAugmenter

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
    
    def __init__(self, config: Configuration):
        super().__init__()
        """
        This is the skeleton of the diffusion models.

        Parameters:

        """
        self.diffuser: DiffusionAB = pick_diffuser(config, config.CHOSEN_MODEL)
        self.lr = config.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE]
        self.optimizer = config.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER]
        self.momentum = config.HYPER_PARAMETERS[LearningHyperParameter.MOMENTUM]
        self.training = config.IS_TRAINING
        self.conditional_dropout = config.HYPER_PARAMETERS[LearningHyperParameter.CONDITIONAL_DROPOUT]
        self.batch_size = config.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE]
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.diffusion_steps = config.HYPER_PARAMETERS[LearningHyperParameter.DIFFUSION_STEPS]
        self.L = config.HYPER_PARAMETERS[LearningHyperParameter.WINDOW_SIZE]
        self.K = config.HYPER_PARAMETERS[LearningHyperParameter.MASKED_WINDOW_SIZE]
        self.len_cond = self.L - self.K
        self.alphas_dash, self.betas = noise_scheduler(self.diffusion_steps, config.HYPER_PARAMETERS[LearningHyperParameter.S])

        self.IS_AUGMENTATION = config.IS_AUGMENTATION

        if (self.IS_AUGMENTATION):
            self.augmenter = LSTMAugmenter(config)
        
        

    def forward(self, input):
        if self.IS_AUGMENTATION:
            input = self.augmenter.augment(input)

        #divide input into x and y
        cond, x_0 = input[:, :self.len_cond, :], input[:, self.len_cond:, :]
        # forward
        x_T, eps = self.diffuser.reparametrized_forward(x_0, cond, self.diffusion_steps-1)
        # reverse
        recon = self.diffuser(x_T, cond, eps)
        
        return recon


    def forward_process(self, x_0, t):
        # Standard forward process, takaes in input x_0 and returns x_t after t steps of noise
        cov_matrix = torch.eye(self.x_size)
        mean = math.sqrt(self.alphas_dash[t]) * x_0
        std = (1 - self.alphas_dash[t]) * cov_matrix
        x_T = torch.distributions.Normal(mean, std).rsample()
        return x_T


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
