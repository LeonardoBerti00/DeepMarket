import math
from typing import Dict
import torch
from models.SinusoidalPosEmb import SinusoidalPosEmb
from models.diffusers.DiffusionAB import DiffusionAB
import constants as cst
from constants import LearningHyperParameter
from torch import nn


class StandardDiffusion(nn.Module, DiffusionAB):
    """An abstract class for loss functions."""

    def __init__(self, config):
        super().__init__()
        self.dropout = config.HYPER_PARAMETERS[LearningHyperParameter.DROPOUT]
        self.batch_size = config.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE]
        self.diffusion_steps = config.HYPER_PARAMETERS[LearningHyperParameter.DIFFUSION_STEPS]
        self.lambda_ = config.HYPER_PARAMETERS[LearningHyperParameter.LAMBDA]
        self.K = config.HYPER_PARAMETERS[LearningHyperParameter.MASKED_WINDOW_SIZE]
        if config.IS_AUGMENTATION:
            self.features_size = config.HYPER_PARAMETERS[LearningHyperParameter.AUGMENT_DIM]
        else:
            self.features_size = cst.LEN_EVENT
        self.NN = nn.Transformer(d_model=self.features_size, nhead=8, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=2048, dropout=self.dropout, activation='relu', batch_first=True)
        self.alphas_dash = config.ALPHAS_DASH
        self.betas = config.BETAS
        self.losses = []
        self.v = nn.Parameter(torch.randn(self.features_size))
        self.SinusoidalPosEmb = SinusoidalPosEmb(self.diffusion_steps)

    def forward(self, x_T: torch.Tensor, context: Dict):
        assert 'eps' in context
        eps = context['eps']
        return self.reverse(x_T, eps)


    def reverse_reparameterized(self, x_T, eps_true):
        x_t = x_T
        losses = []
        for t in range(1, self.diffusion_steps):
            beta_t = self.betas[t]
            alpha_t = 1 - beta_t
            alpha_dash_t = self.alphas_dash[t]
            beta_tilde_t = (1 - self.alphas_dash[t-1]) / (1 - self.alphas_dash[t]) * beta_t
            emb_t = self.SinusoidalPosEmb(t)
            eps_t = self.NN(x_t, emb_t)
            z = torch.distributions.Normal(0, 1).sample(x_t.shape)
            sigma_t = math.exp(self.v*math.log(beta_t) + (1-self.v)*math.log(beta_tilde_t))       #formula taken from IDDPM paper
            x_t = 1 / math.sqrt(alpha_t) * (x_t[:] - (beta_t / math.sqrt(1 - alpha_dash_t)) * eps_t) + sigma_t * z
            L_t = self.loss_step(eps_t, eps_true)
            losses.append(L_t)
        self.losses.append(losses)
        return x_t


    def loss_step_reparameterized(self, eps_t, eps_true):
        return torch.norm(eps_t - eps_true, p=2)


    def loss(self, true: torch.Tensor, recon: torch.Tensor, **kwargs) -> torch.Tensor:
        """Computes the loss taken from IDDPM."""
        L_simple = self.losses[-1]
        L_vlb = ...
        L_hybrid = L_simple + self.lambda_*L_vlb
        return L_hybrid