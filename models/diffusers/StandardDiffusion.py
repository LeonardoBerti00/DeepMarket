import math
from typing import Dict, Tuple
import torch
from models.diffusers.DiffusionAB import DiffusionAB
import constants as cst
from constants import LearningHyperParameter
from torch import nn
from models.diffusers.DiT.DiT import DiT


class StandardDiffusion(nn.Module, DiffusionAB):
    """An abstract class for loss functions."""

    def __init__(self, config):
        super().__init__()
        self.dropout = config.HYPER_PARAMETERS[LearningHyperParameter.DROPOUT]
        self.cond_dropout = config.HYPER_PARAMETERS[LearningHyperParameter.CONDITIONAL_DROPOUT]
        self.diffusion_steps = config.HYPER_PARAMETERS[LearningHyperParameter.DIFFUSION_STEPS]
        self.lambda_ = config.HYPER_PARAMETERS[LearningHyperParameter.LAMBDA]
        self.x_seq_size = config.HYPER_PARAMETERS[LearningHyperParameter.MASKED_SEQ_SIZE]
        self.seq_size = config.HYPER_PARAMETERS[LearningHyperParameter.SEQ_SIZE]
        self.cond_seq_size = self.seq_size - self.x_seq_size
        self.depth = config.HYPER_PARAMETERS[LearningHyperParameter.DiT_DEPTH]
        self.num_heads = config.HYPER_PARAMETERS[LearningHyperParameter.DiT_NUM_HEADS]
        self.mlp_ratio = config.HYPER_PARAMETERS[LearningHyperParameter.DiT_MLP_RATIO]
        self.cond_dropout_prob = config.HYPER_PARAMETERS[LearningHyperParameter.CONDITIONAL_DROPOUT]
        self.type = config.HYPER_PARAMETERS[LearningHyperParameter.DiT_TYPE]
        self.input_size = cst.LEN_EVENT
        batch_size = config.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE]
        if config.IS_AUGMENTATION_X:
            self.hidden_size = config.HYPER_PARAMETERS[LearningHyperParameter.AUGMENT_DIM]
        else:
            self.hidden_size = self.input_size

        if config.IS_AUGMENTATION_COND:
            self.cond_size = config.HYPER_PARAMETERS[LearningHyperParameter.AUGMENT_DIM]
        else:
            self.cond_size = config.COND_SIZE

        self.NN = DiT(
            self.input_size,
            self.cond_seq_size,
            self.hidden_size,
            self.cond_size,
            self.diffusion_steps,
            self.depth,
            self.num_heads,
            self.x_seq_size,
            self.mlp_ratio,
            self.cond_dropout_prob,
            self.type
        )
        self.alphas_cumprod = config.ALPHAS_CUMPROD
        self.betas = config.BETAS
        self.losses = []
        self.v = nn.Parameter(torch.randn(batch_size, self.x_seq_size, self.hidden_size))
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        #calculation for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_var = (1.0 - self.alphas_cumprod) / (1.0 - self.alphas_cumprod) * self.betas

    def forward(self, x_T: torch.Tensor, context: Dict):
        assert 'eps' in context
        eps = context['eps']
        assert 'condiitoning' in context
        cond = context['conditioning']
        return self.reverse_reparametrized(x_T, cond, eps)

    def forward_reparametrized(self, x_0: torch.Tensor, diffusion_step: int, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert 'conditioning' in kwargs
        cond: torch.Tensor = kwargs['conditioning']
        x_t, eps = super().forward_reparametrized(x_0, diffusion_step)
        return x_t, {'eps': eps, 'conditioning': cond}

    def reverse_reparametrized(self, x_t, cond, eps_true):
        step_losses = []
        for t in range(1, self.diffusion_steps):
            beta_t = self.betas[t]
            alpha_t = 1 - beta_t
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_tilde_t = self.posterior_var[t]          #it's the posterior variance of q(x_{t-1} | x_t, x_0)
            times = torch.full((x_t.shape[0],), t, device=cst.DEVICE)
            eps_t, var_t = self.NN(x_t, cond, times)
            assert eps_t.shape == x_t.shape
            z = torch.distributions.Normal(0, 1).sample(x_t.shape)
            sigma_t = math.exp(self.v*math.log(beta_t) + (1-self.v)*math.log(beta_tilde_t))       #formula taken from IDDPM paper
            x_t = 1 / math.sqrt(alpha_t) * (x_t - (beta_t / math.sqrt(1 - alpha_cumprod_t) * eps_t)) + (sigma_t * z)
            L_t = self.loss_step(eps_t, eps_true)
            step_losses.append(L_t)
        self.losses.append(step_losses)
        return x_t, {}


    def loss_step(self, eps_t, eps_true):
        return torch.norm(eps_t - eps_true, p=2)


    def loss(self, true: torch.Tensor, recon: torch.Tensor, **kwargs) -> torch.Tensor:
        """Computes the loss taken from IDDPM."""
        L_simple = self.losses[-1]
        L_vlb = ...
        L_hybrid = L_simple + self.lambda_*L_vlb
        return L_hybrid