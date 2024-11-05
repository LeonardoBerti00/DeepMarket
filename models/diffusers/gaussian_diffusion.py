from typing import Dict, Tuple
import numpy as np
import torch
from einops import rearrange, repeat, reduce

from models.diffusers.diffusionAB import DiffusionAB
import constants as cst
from constants import LearningHyperParameter
from torch import nn
from models.diffusers.TRADES.TRADES import TRADES

"""
Functions -> _vlb_loss, _p_mean, _q_posterior_mean_var, _normal_kl, _gaussian_log_likelihood, _approx_standard_normal_cdf 
are ported from the original Ho et al. diffusion models codebase: https://github.com/hojonathanho/diffusion
"""

class GaussianDiffusion(nn.Module, DiffusionAB):
    """A diffusion model that uses Gaussian noise inspired from the IDDPM paper."""
    def __init__(self, config, feature_augmenter, bin):
        super().__init__()
        self.dropout = config.HYPER_PARAMETERS[LearningHyperParameter.DROPOUT]
        self.batch_size = config.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE]
        self.num_diffusionsteps = config.HYPER_PARAMETERS[LearningHyperParameter.NUM_DIFFUSIONSTEPS]
        self.lambda_ = config.HYPER_PARAMETERS[LearningHyperParameter.LAMBDA]
        self.x_seq_size = config.HYPER_PARAMETERS[LearningHyperParameter.MASKED_SEQ_SIZE]
        self.seq_size = config.HYPER_PARAMETERS[LearningHyperParameter.SEQ_SIZE]
        self.cond_seq_size = self.seq_size - self.x_seq_size
        self.depth = config.HYPER_PARAMETERS[LearningHyperParameter.TRADES_DEPTH]
        self.num_heads = config.HYPER_PARAMETERS[LearningHyperParameter.TRADES_NUM_HEADS]
        self.mlp_ratio = config.HYPER_PARAMETERS[LearningHyperParameter.TRADES_MLP_RATIO]
        #self.depth = config.HYPER_PARAMETERS[LearningHyperParameter.CDT_DEPTH]
        #self.num_heads = config.HYPER_PARAMETERS[LearningHyperParameter.CDT_NUM_HEADS]
        #self.mlp_ratio = config.HYPER_PARAMETERS[LearningHyperParameter.CDT_MLP_RATIO]
        self.cond_dropout_prob = config.HYPER_PARAMETERS[LearningHyperParameter.CONDITIONAL_DROPOUT]
        self.sampling_type = config.SAMPLING_TYPE
        self.IS_AUGMENTATION = config.IS_AUGMENTATION
        self.init_losses()
        self.cond_method = config.COND_METHOD
        self.bin = bin
        if config.IS_AUGMENTATION:
            self.input_size = config.HYPER_PARAMETERS[LearningHyperParameter.AUGMENT_DIM]
            self.feature_augmenter = feature_augmenter
        else:
            self.input_size = config.HYPER_PARAMETERS[LearningHyperParameter.SIZE_ORDER_EMB]
        self.NN = TRADES(
            self.input_size,
            self.cond_seq_size,
            self.num_diffusionsteps,
            self.depth,
            self.num_heads,
            self.x_seq_size,
            self.cond_dropout_prob,
            self.IS_AUGMENTATION,
            self.dropout,
            config.COND_TYPE,
            self.cond_method
        )

        self.betas = config.BETAS
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0, dtype=torch.float32)
        self.alphas_cumprod_prev = torch.cat([torch.Tensor([self.alphas_cumprod[0]]).to(cst.DEVICE), self.alphas_cumprod[:-1]])
        # calculation for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_var = (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod) * self.betas
        self.posterior_log_var_clipped = torch.log(self.posterior_var)
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )
            
        if self.sampling_type == "DDIM":
            self.ddim_eta = config.HYPER_PARAMETERS[LearningHyperParameter.DDIM_ETA]
            self.ddim_nsteps = config.HYPER_PARAMETERS[LearningHyperParameter.DDIM_NSTEPS]
            tmp = self.num_diffusionsteps / self.ddim_nsteps
            self.t = torch.arange(0, self.num_diffusionsteps, tmp).long() + 1
            self.ddim_alpha = self.alphas_cumprod[self.t].clone()
            self.ddim_alpha_sqrt = torch.sqrt(self.ddim_alpha)
            self.ddim_alpha_prev = torch.cat([torch.Tensor([self.alphas_cumprod[0]]), self.alphas_cumprod[self.t[:-1]]])
            self.ddim_sqrt_one_minus_alpha = (1. - self.ddim_alpha) ** .5
            self.ddim_sigma = (self.ddim_eta *
                               ((1 - self.ddim_alpha_prev) / (1 - self.ddim_alpha) *
                                (1 - self.ddim_alpha / self.ddim_alpha_prev)) ** .5)
            
        
    def sample(self, x_0, real_cond_orders, real_cond_lob, weights):
        if self.sampling_type == "DDIM":
            return self.ddim_sample(x_0, real_cond_orders, real_cond_lob)
        elif self.sampling_type == "DDPM":
            return self.ddpm_sample(x_0, real_cond_orders, real_cond_lob, weights)
        
        
    def ddim_sample(self, x_0, cond_orders, cond_lob):
        orig_cond_orders = cond_orders.detach().clone()
        cond_lob = rearrange(cond_lob, 'b l f -> b f l')
        cond_lob = self.bin(cond_lob)
        cond_lob = rearrange(cond_lob, 'b f l -> b l f')
        if cond_lob is not None:
            orig_cond_lob = cond_lob.detach().clone()
        else:
            orig_cond_lob = None
        tmp = torch.full(size=(x_0.shape[0],), fill_value=self.num_diffusionsteps-1, device=cst.DEVICE, dtype=torch.int64)
        x_t, context = self.forward_reparametrized(x_0, tmp, **{"conditioning": cond_orders})
        context.update({'x_t': x_t.detach().clone()})        
        time_steps = torch.flip(self.t, dims=(0,))
        for i, step in enumerate(time_steps):
            # augment
            x_t_aug, cond_orders, cond_lob = self.augment(x_t, orig_cond_orders, orig_cond_lob)
            index = len(time_steps) - i - 1
            ts = x_t.new_full((x_0.shape[0],), step, dtype=torch.long)
            x_t = self.ddim_single_step(x_t_aug, cond_lob, cond_orders, ts, index, x_t)
        return x_t
        
    def ddim_single_step(self, x_t_aug, cond_lob, cond_orders, ts, index, x_t):
        noise_t, v = self.NN(x_t_aug, cond_orders, ts, cond_lob)
        #check for nan in x_t_aug and cond and noise_t
        if self.IS_AUGMENTATION:
            e_t, v = self.deaugment(noise_t, v)
        alpha = self.ddim_alpha[index]
        alpha_prev = self.ddim_alpha_prev[index]
        sigma = self.ddim_sigma[index]
        sqrt_one_minus_alpha = self.ddim_sqrt_one_minus_alpha[index]
        # Current prediction for x_0 
        pred_x0 = (x_t - sqrt_one_minus_alpha * e_t) / (alpha ** 0.5)
        dir_xt = (1. - alpha_prev - sigma ** 2).sqrt() * e_t
        # no noise is added, when η=0
        if sigma == 0.:
            noise = 0.
        else:
            noise = torch.randn(x_t.shape, device=x_t.device)
        x_prev = (alpha_prev ** 0.5) * pred_x0 + dir_xt + sigma * noise
        return x_prev
    
    def ddpm_sample(self, x_0, cond_orders, cond_lob, weights):
        orig_cond_orders = cond_orders.detach().clone()
        cond_lob = rearrange(cond_lob, 'b l f -> b f l')
        cond_lob = self.bin(cond_lob)
        cond_lob = rearrange(cond_lob, 'b f l -> b l f')
        if cond_lob is not None:
            orig_cond_lob = cond_lob.detach().clone()
        else:
            orig_cond_lob = None
        t = torch.full(size=(x_0.shape[0],), fill_value=self.num_diffusionsteps-1, device=cst.DEVICE, dtype=torch.int64)
        x_t, noise = self.forward_reparametrized(x_0, t)
        for i in range(self.num_diffusionsteps-1, -1, -1):
            # augment
            x_t_aug, cond_orders, cond_lob = self.augment(x_t, orig_cond_orders, orig_cond_lob)
            x_t = self.ddpm_single_step(x_0, x_t_aug, x_t, t, cond_orders, noise, weights, cond_lob)
            t -= 1
        return x_t


    def forward_reparametrized(self, x_0: torch.Tensor, t: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_t, noise = super().forward_reparametrized(x_0, t)
        return x_t, noise

    def ddpm_single_step(self, x_0, x_t_aug, x_t, t, cond_orders, noise_true, weights, cond_lob, batch_idx=None):
        '''
        Compute the reverse diffusion process for the current time step
        '''
        # Get the beta and alpha values for the current time step
        beta_t = self.betas[t]
        alpha_t = 1 - beta_t
        alpha_cumprod_t = self.alphas_cumprod[t]
        beta_t = repeat(beta_t, 'b -> b 1 d', d=x_0.shape[-1])
        alpha_t = repeat(alpha_t, 'b -> b 1 d', d=x_0.shape[-1])
        alpha_cumprod_t = repeat(alpha_cumprod_t, 'b -> b 1 d', d=x_0.shape[-1])
        # Get the noise and v outputs from the neural network for the current time step
        noise_t, v = self.NN(x_t_aug, cond_orders, t, cond_lob)
        #check for nan in x_t_aug and cond and noise_t
        if torch.isnan(v).any():
            print("v", v.max())
        if torch.isnan(noise_t).any():
            print("noise_t:", noise_t.max())
        if self.IS_AUGMENTATION:
            noise_t, v = self.deaugment(noise_t, v)
        # Compute the variance for the current time step using the formula from the IDDPM paper
        frac = (v + 1) / 2
        max_log = torch.log(beta_t)
        min_log = self.posterior_log_var_clipped[t]
        min_log = repeat(min_log, 'b -> b 1 d', d=x_0.shape[-1])
        log_var_t = frac * max_log + (1 - frac) * min_log
        var_t = torch.exp(log_var_t)
        std_t = torch.sqrt(var_t)
        # Sample a standard normal random variable z
        z = torch.distributions.normal.Normal(0, 1).sample(x_t.shape).to(cst.DEVICE, non_blocking=True)

        #take the indexes for which t = 1
        indexes = torch.where(t == 0)
        z[indexes] = 0.0

        # Compute x_{t-1} from x_t through the reverse diffusion process for the current time step
        x_recon = 1 / torch.sqrt(alpha_t) * (x_t - (beta_t / torch.sqrt(1 - alpha_cumprod_t) * noise_t)) + (std_t * z)
        # Compute the mean squared error loss between the noise and the true noise
        L_mse = self._mse_loss(noise_t, noise_true)
        # Append the loss to the mse_losses list
        self.mse_losses.append(L_mse)
        # Compute the variational lower bound loss for the current time step
        L_vlb = self._vlb_loss(
            noise_t=noise_t.detach(),
            pred_log_var=log_var_t,
            x_0=x_0,
            x_t=x_t,
            t=t,
            beta_t=beta_t,
            alpha_t=alpha_t,
            alpha_cumprod_t=alpha_cumprod_t,
            clip_denoised=False,
            weights=weights
        )
        #check if there are nan in L_vlb
        if torch.isnan(L_vlb).any():
            print("L_vlb:", L_vlb.max())
        # Append the loss to the vbl_losses list
        self.vlb_losses.append(L_vlb)
        return x_recon

    def augment(self, x_t: torch.Tensor, cond_orders: torch.Tensor, cond_lob: torch.Tensor):
        if self.IS_AUGMENTATION:
            full_orders = torch.cat([cond_orders, x_t], dim=1)
            full_orders_aug, cond_lob = self.feature_augmenter.augment(full_orders, cond_lob)
            cond_orders = full_orders_aug[:, :self.cond_seq_size, :]
            x_t = full_orders_aug[:, self.cond_seq_size:, :]
        return x_t, cond_orders, cond_lob

    def deaugment(self, noise: torch.Tensor, v: torch.Tensor):
        noise, v = self.feature_augmenter.deaugment(noise, v)
        return noise, v

    def _mse_loss(self, noise_t, noise_true):
        return torch.norm(noise_t - noise_true, p=2, dim=[1, 2])

    def loss(self):
        """Computes the loss taken from DDPM."""
        L_simple = torch.stack(self.mse_losses)
        L_vlb = torch.stack(self.vlb_losses)
        #print("L_simple:", torch.mean(L_simple).item(), "L_vlb:", torch.mean(L_vlb).item())
        L_hybrid = L_simple + self.lambda_*L_vlb
        return L_hybrid, L_simple, L_vlb


    #ported from https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
    def _vlb_loss(
        self, noise_t, pred_log_var, x_0, x_t, t, beta_t, alpha_t, alpha_cumprod_t, clip_denoised=False, model_kwargs=None, weights=None
    ):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits.
        This allows for comparison to other papers.
        """
        true_mean, true_log_variance_clipped = self._q_posterior_mean_var(x_0=x_0, x_t=x_t, t=t)
        pred_mean = self._p_mean(
            noise_t, x_t, t, beta_t, alpha_t, alpha_cumprod_t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = self._normal_kl(
            true_mean, true_log_variance_clipped, pred_mean, pred_log_var
        )
        #check for nan in kl and print
        if torch.isnan(kl).any():
            print("kl:")
        kl = self._mean_flat(kl) / np.log(2.0)
        decoder_nll = -self._gaussian_log_likelihood(
            x_0, means=pred_mean, log_scales=pred_log_var*0.5
        )
        if torch.isnan(decoder_nll).any():
            print("decoder_nll:")
        assert decoder_nll.shape == x_0.shape
        decoder_nll = self._mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return output / torch.from_numpy(weights).to(cst.DEVICE)[t]

    def _p_mean(self, noise_t, x_t, t,  beta_t, alpha_t, alpha_cumprod_t, clip_denoised=True, model_kwargs=None):
        '''
        Get the mean of the prior p(x_{t-1} | x_t).
        '''
        pred_mean = 1/torch.sqrt(alpha_t) * (x_t - (beta_t*noise_t/torch.sqrt(1-alpha_cumprod_t)))
        return pred_mean

    def _q_posterior_mean_var(self, x_0, x_t, t):
        """
        Get the mean and variance of the posterior q(x_{t-1} | x_t, x_0).

        :param x_0: the initial image.
        :param x_t: the image at timestep t.
        :param t: the timestep.
        :return: a tuple (mean, variance).
        """
        posterior_mean_coef1 = repeat(self.posterior_mean_coef1[t], 'b -> b 1 d', d=x_0.shape[-1])
        posterior_mean_coef2 = repeat(self.posterior_mean_coef2[t], 'b -> b 1 d', d=x_0.shape[-1])
        true_mean = (
                posterior_mean_coef1 * x_0
                + posterior_mean_coef2 * x_t
        )
        true_log_var_clipped = repeat(self.posterior_log_var_clipped[t], 'b -> b 1 d', d=x_0.shape[-1])
        return true_mean, true_log_var_clipped


    def _normal_kl(self, mean1, logvar1, mean2, logvar2):
        """
        Compute the KL divergence between two gaussians.

        Shapes are automatically broadcasted, so batches can be compared to
        scalars, among other use cases.
        """
        output = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))
        return output

    def _gaussian_log_likelihood(self, x, means, log_scales):
        """
        It computes the log-likelihood log(p(x_0)) that is the probability that x was generated by the predicted distribution.
        We need it when t = 0.
        :param x: the target.
        :param means: the Gaussian mean Tensor.
        :param log_scales: the Gaussian log var Tensor.
        :return: a tensor like x of log probabilities (in nats).
        """
        assert x.shape == means.shape == log_scales.shape
        centered_x = x - means
        inv_stdv = torch.exp(log_scales)
        plus_in = inv_stdv * (centered_x + 1.0)
        cdf_plus = self._approx_standard_normal_cdf(plus_in)
        min_in = inv_stdv * (centered_x - 1.0)
        cdf_min = self._approx_standard_normal_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-6))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-6))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-6))),
        )
        return log_probs


    def _approx_standard_normal_cdf(self, x):
        """
        A fast approximation of the cumulative distribution function of the
        standard normal.
        """
        return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def _mean_flat(self, tensor):
        """
        Take the mean over all non-batch dimensions.
        """
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    def init_losses(self):
        self.mse_losses = []
        self.vlb_losses = []