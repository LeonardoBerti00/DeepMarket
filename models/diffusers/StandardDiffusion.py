import math
from typing import Dict
import torch
from models.diffusers.DiffusionAB import DiffusionAB
import constants as cst
from constants import LearningHyperParameter
from torch import nn
from utils.utils import sinusoidal_positional_embedding
from models.DiT import DiT, ConditionEmbedder

class StandardDiffusion(nn.Module, DiffusionAB):
    """An abstract class for loss functions."""

    def __init__(self, config):
        super().__init__()
        self.dropout = config.HYPER_PARAMETERS[LearningHyperParameter.DROPOUT]
        self.cond_droput = config.HYPER_PARAMETERS[LearningHyperParameter.CONDITIONAL_DROPOUT]
        self.diffusion_steps = config.HYPER_PARAMETERS[LearningHyperParameter.DIFFUSION_STEPS]
        self.lambda_ = config.HYPER_PARAMETERS[LearningHyperParameter.LAMBDA]
        self.x_window_size = config.HYPER_PARAMETERS[LearningHyperParameter.MASKED_WINDOW_SIZE]
        self.window_size = config.HYPER_PARAMETERS[LearningHyperParameter.WINDOW_SIZE]
        self.cond_seq_size = self.window_size - self.x_window_size
        self.emb_t_dim = config.HYPER_PARAMETERS[LearningHyperParameter.EMB_T_DIM]
        if config.IS_AUGMENTATION:
            self.input_size = config.HYPER_PARAMETERS[LearningHyperParameter.AUGMENT_DIM]
        else:
            self.input_size = cst.LEN_EVENT
        self.cond_size = cst.COND_SIZE    #TODO change this, because cond size can be equal to x size if the conditoning is concatenation
        self.NN = DiT(
            self.input_size,
            self.cond_seq_size,
            self.hidden_size,
            self.cond_size,
            self.diffusion_steps,
            self.depth,
            self.num_heads,
            self.x_window_size,
            self.mlp_ratio,
            self.cond_dropout_prob,
            self.type
        )
        self.alphas_dash = config.ALPHAS_DASH
        self.betas = config.BETAS
        self.losses = []
        self.v = nn.Parameter(torch.randn(self.features_size))

    def forward(self, x_T: torch.Tensor, context: Dict, cond):
        assert 'eps' in context
        eps = context['eps']
        return self.reverse_reparameterized(x_T, cond, eps)


    def reverse_reparameterized(self, x_T, cond, eps_true):
        x_t = x_T
        step_losses = []
        print("x_t shape: ", x_t.shape)
        for t in range(1, self.diffusion_steps):
            beta_t = self.betas[t]
            alpha_t = 1 - beta_t
            alpha_dash_t = self.alphas_dash[t]
            beta_tilde_t = (1 - self.alphas_dash[t-1]) / (1 - self.alphas_dash[t]) * beta_t
            eps_t = self.NN(x_t, cond, t)
            z = torch.distributions.Normal(0, 1).sample(x_t.shape)
            sigma_t = math.exp(self.v*math.log(beta_t) + (1-self.v)*math.log(beta_tilde_t))       #formula taken from IDDPM paper
            x_t = 1 / math.sqrt(alpha_t) * (x_t[:] - (beta_t / math.sqrt(1 - alpha_dash_t)) * eps_t) + sigma_t * z
            L_t = self.loss_step(eps_t, eps_true)
            step_losses.append(L_t)
        self.losses.append(step_losses)
        return x_t


    def loss_step_reparameterized(self, eps_t, eps_true):
        return torch.norm(eps_t - eps_true, p=2)


    def loss(self, true: torch.Tensor, recon: torch.Tensor, **kwargs) -> torch.Tensor:
        """Computes the loss taken from IDDPM."""
        L_simple = self.losses[-1]
        L_vlb = ...
        L_hybrid = L_simple + self.lambda_*L_vlb
        return L_hybrid