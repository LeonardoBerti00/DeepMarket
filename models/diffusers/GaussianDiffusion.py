import math
from typing import Dict, Tuple
import numpy as np
import torch
from einops import rearrange, repeat, reduce

from config import Configuration
from models.diffusers.DiffusionAB import DiffusionAB
import constants as cst
from constants import LearningHyperParameter
from torch import nn
from models.diffusers.CDT.CDT import CDT, CDT, DiT


class GaussianDiffusion(nn.Module, DiffusionAB):
    """A diffusion model that uses Gaussian noise inspired from the IDDPM paper."""
    def __init__(self, config: Configuration, feature_augmenter):
        super().__init__()
        self.dropout = config.HYPER_PARAMETERS[LearningHyperParameter.DROPOUT]
        self.cond_dropout = config.HYPER_PARAMETERS[LearningHyperParameter.CONDITIONAL_DROPOUT]
        self.num_diffusionsteps = config.HYPER_PARAMETERS[LearningHyperParameter.NUM_DIFFUSIONSTEPS]
        self.lambda_ = config.HYPER_PARAMETERS[LearningHyperParameter.LAMBDA]
        self.x_seq_size = config.HYPER_PARAMETERS[LearningHyperParameter.MASKED_SEQ_SIZE]
        self.seq_size = config.HYPER_PARAMETERS[LearningHyperParameter.SEQ_SIZE]
        self.cond_seq_size = self.seq_size - self.x_seq_size
        self.depth = config.HYPER_PARAMETERS[LearningHyperParameter.CDT_DEPTH]
        self.num_heads = config.HYPER_PARAMETERS[LearningHyperParameter.CDT_NUM_HEADS]
        self.mlp_ratio = config.HYPER_PARAMETERS[LearningHyperParameter.CDT_MLP_RATIO]
        self.cond_dropout_prob = config.HYPER_PARAMETERS[LearningHyperParameter.CONDITIONAL_DROPOUT]
        self.type = config.COND_METHOD
        self.IS_AUGMENTATION = config.IS_AUGMENTATION
        self.init_losses()
        if config.IS_AUGMENTATION:
            self.input_size = config.HYPER_PARAMETERS[LearningHyperParameter.AUGMENT_DIM]
            self.cond_size = config.HYPER_PARAMETERS[LearningHyperParameter.AUGMENT_DIM]
            self.feature_augmenter = feature_augmenter
        else:
            self.input_size = cst.LEN_EVENT_ONE_HOT
            self.cond_size = config.COND_SIZE
        if (self.type == 'adaln_zero'):
            self.NN = DiT(
                self.input_size,
                self.cond_seq_size,
                self.cond_size,
                self.num_diffusionsteps,
                self.depth,
                self.num_heads,
                self.x_seq_size,
                self.mlp_ratio,
                self.cond_dropout_prob,
            )
        elif (self.type == 'concatenation'):
            self.NN = CDT(
                self.input_size,
                self.cond_seq_size,
                self.cond_size,
                self.num_diffusionsteps,
                self.depth,
                self.num_heads,
                self.x_seq_size,
                self.mlp_ratio,
                self.cond_dropout_prob,
            )
        elif (self.type == 'crossattention'):
            pass
        else:
            raise ValueError("Invalid conditioning type")

        self.betas = config.BETAS
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0, dtype=torch.float32)
        self.alphas_cumprod_prev = torch.cat([torch.Tensor([self.alphas_cumprod[0]]).to(cst.DEVICE, non_blocking=True), self.alphas_cumprod[:-1]])

        #calculation for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_var = (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod) * self.betas
        self.posterior_log_var_clipped = torch.log(
            torch.cat([self.posterior_var[:1], self.posterior_var[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )
        # if we use concatenation or crossattention, we need to use the feature augmentation or cond type full
        assert config.COND_TYPE == 'only_event' or self.IS_AUGMENTATION or self.type == 'adaln_zero'


    def forward(self, x_t_aug: torch.Tensor, context: Dict):
        assert 'x_t' in context
        x_t = context['x_t']
        assert 'noise_true' in context
        noise_true = context['noise_true']
        assert 'conditioning' in context
        cond = context['conditioning_aug']
        assert 't' in context
        t = context['t']
        assert 'x_0' in context
        x_0 = context['x_0']
        return self.reverse_reparametrized(x_0, x_t_aug, x_t, t, cond, noise_true)


    def forward_reparametrized(self, x_0: torch.Tensor, t: int, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert 'conditioning' in kwargs
        cond: torch.Tensor = kwargs['conditioning']
        x_t, noise = super().forward_reparametrized(x_0, t)
        return x_t, {'noise_true': noise, 'conditioning': cond}

    def reverse_reparametrized(self, x_0, x_t_aug, x_t, t, cond, noise_true):
        '''
        Compute the reverse diffusion process for the current time step
        '''
        # Get the beta and alpha values for the current time step
        beta_t = self.betas[t]
        alpha_t = 1 - beta_t
        alpha_cumprod_t = self.alphas_cumprod[t]
        beta_t = repeat(beta_t, 'b -> b 1 d', d=cst.LEN_EVENT_ONE_HOT)
        alpha_t = repeat(alpha_t, 'b -> b 1 d', d=cst.LEN_EVENT_ONE_HOT)
        alpha_cumprod_t = repeat(alpha_cumprod_t, 'b -> b 1 d', d=cst.LEN_EVENT_ONE_HOT)
        # Get the noise and v outputs from the neural network for the current time step
        noise_t, v = self.NN(x_t_aug, cond, t)
        if self.IS_AUGMENTATION:
            noise_t, v = self.deaugment(noise_t, v)
        # Compute the variance for the current time step using the formula from the IDDPM paper
        frac = (v + 1) / 2
        max_log = torch.log(beta_t)
        min_log = self.posterior_log_var_clipped[t]
        min_log = repeat(min_log, 'b -> b 1 d', d=cst.LEN_EVENT_ONE_HOT)
        log_var_t = frac * max_log + (1 - frac) * min_log
        var_t = torch.exp(log_var_t)

        # Sample a standard normal random variable z
        z = torch.distributions.normal.Normal(0, 1).sample(x_t.shape).to(cst.DEVICE, non_blocking=True)
        # Compute x_{t-1} from x_t the reverse diffusion process for the current time step
        x_recon = 1 / torch.sqrt(alpha_t) * (x_t - (beta_t / torch.sqrt(1 - alpha_cumprod_t) * noise_t)) + (var_t * z)

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
        )
        # Append the loss to the vbl_losses list
        self.vlb_losses.append(L_vlb)
        # Return the reverse diffusion output and an empty dictionary
        return x_recon, {}

    def deaugment(self, noise: torch.Tensor, v: torch.Tensor):
        noise, v = self.feature_augmenter.deaugment(noise, v)
        return noise, v

    def _mse_loss(self, noise_t, noise_true):
        return torch.norm(noise_t - noise_true, p=2, dim=[1, 2])

    def loss(self, true: torch.Tensor, recon: torch.Tensor, **kwargs) -> torch.Tensor:
        """Computes the loss taken from IDDPM."""
        L_simple = torch.stack(self.mse_losses)
        L_vlb = torch.stack(self.vlb_losses)
        #compute average differences in orders of magnitude between L_simple e L_vlb
        L_hybrid = L_simple + L_vlb
        return L_hybrid


    #ported from https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
    def _vlb_loss(
        self, noise_t, pred_log_var, x_0, x_t, t, beta_t, alpha_t, alpha_cumprod_t, clip_denoised=False, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits.
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, true_log_variance_clipped = self._q_posterior_mean_var(x_0=x_0, x_t=x_t, t=t)
        pred_mean = self._p_mean(
            noise_t, x_t, t, beta_t, alpha_t, alpha_cumprod_t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )

        kl = self._normal_kl(
            true_mean, true_log_variance_clipped, pred_mean, pred_log_var
        )
        kl = self._mean_flat(kl) / np.log(2.0)
        decoder_nll = -self._gaussian_log_likelihood(
            x_0, means=pred_mean, log_scales=pred_log_var*0.5
        )
        assert decoder_nll.shape == x_0.shape
        decoder_nll = self._mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return output

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
        posterior_mean_coef1 = repeat(self.posterior_mean_coef1[t], 'b -> b 1 d', d=cst.LEN_EVENT_ONE_HOT)
        posterior_mean_coef2 = repeat(self.posterior_mean_coef2[t], 'b -> b 1 d', d=cst.LEN_EVENT_ONE_HOT)
        true_mean = (
                posterior_mean_coef1 * x_0
                + posterior_mean_coef2 * x_t
        )
        true_log_var_clipped = repeat(self.posterior_log_var_clipped[t], 'b -> b 1 d', d=cst.LEN_EVENT_ONE_HOT)
        return true_mean, true_log_var_clipped

    """
    Ported from the original Ho et al. diffusion models codebase:
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
    """
    def _normal_kl(self, mean1, logvar1, mean2, logvar2):
        """
        Compute the KL divergence between two gaussians.

        Shapes are automatically broadcasted, so batches can be compared to
        scalars, among other use cases.
        """
        output = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))
        '''
        k = torch.full((mean1.shape), cst.LEN_EVENT_ONE_HOT, dtype=torch.float32).to(cst.DEVICE, non_blocking=True)
        #create a tensor matrices with the value of logvar1 as the diagonal
        diag_ar1 = torch.diag_embed(torch.exp(logvar1))
        diag_ar2 = torch.diag_embed(torch.exp(logvar2))
        tmp = (torch.einsum('blmn, blgh -> blmh', torch.inverse(diag_ar1), diag_ar2))
        mask = torch.zeros(diag_ar1.shape).to(cst.DEVICE, non_blocking=True)
        mask[:, :, torch.arange(0, diag_ar1.shape[2]), torch.arange(0, diag_ar1.shape[3])] = 1.0  # This will mask all non-diagonal values.
        tmp_masked = tmp * mask
        trace_term = reduce(tmp_masked, 'b l m n -> b l m', 'sum')  # output will will contain the trace of each batch matrices.
        output3 = 0.5 * ((logvar2 - logvar1) - k + (mean1 - mean2) * torch.exp(logvar2) * (mean1 - mean2) + trace_term)
        '''
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
        '''
        assert x.shape == means.shape == log_scales.shape
        centered_x = x - means
        var = torch.exp(log_scales)
        first_term = - ((centered_x ** 2) / (2*var))
        second_term = - log_scales*0.5
        third_term = - (math.log(2*math.pi)/2)
        log_probs = -((centered_x ** 2) / (2*var)) - (0.5 * log_scales) - (math.log(2 * math.pi)/2)
        assert log_probs.shape == x.shape
        '''
        assert x.shape == means.shape == log_scales.shape
        if torch.isnan(x).any():  # check for nan values in x
            print("x contains nan:", torch.isnan(x).any())
        centered_x = x - means
        if torch.isnan(centered_x).any():  # check for nan values in centered_x
            print("centered_x contains nan:", torch.isnan(centered_x).any())
        inv_stdv = torch.exp(log_scales)
        if torch.isnan(inv_stdv).any():  # check for nan values in inv_stdv
            print("inv_stdv contains nan:", torch.isnan(inv_stdv).any())
        plus_in = inv_stdv * (centered_x + 1.0)
        if torch.isnan(plus_in).any():  # check for nan values in plus_in
            print("plus_in contains nan:", torch.isnan(plus_in).any())
        cdf_plus = self._approx_standard_normal_cdf(plus_in)
        if torch.isnan(cdf_plus).any():  # check for nan values in cdf_plus
            print("cdf_plus contains nan:", torch.isnan(cdf_plus).any())
        min_in = inv_stdv * (centered_x - 1.0)
        if torch.isnan(min_in).any():  # check for nan values in min_in
            print("min_in contains nan:", torch.isnan(min_in).any())
        cdf_min = self._approx_standard_normal_cdf(min_in)
        if torch.isnan(cdf_min).any():  # check for nan values in cdf_min
            print("cdf_min contains nan:", torch.isnan(cdf_min).any())
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-6))
        if torch.isnan(log_cdf_plus).any():  # check for nan values in log_cdf_plus
            print("log_cdf_plus contains nan:", torch.isnan(log_cdf_plus).any())
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-6))
        if torch.isnan(log_one_minus_cdf_min).any():  # check for nan values in log_one_minus_cdf_min
            print("log_one_minus_cdf_min contains nan:", torch.isnan(log_one_minus_cdf_min).any())
        cdf_delta = cdf_plus - cdf_min
        if torch.isnan(cdf_delta).any():  # check for nan values in cdf_delta
            print("cdf_delta contains nan:", torch.isnan(cdf_delta).any())
        log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-6))),
        )
        if torch.isnan(log_probs).any():  # check for nan values in log_probs
            print("log_probs contains nan:", torch.isnan(log_probs).any())
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