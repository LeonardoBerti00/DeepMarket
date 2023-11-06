from typing import Dict, Tuple
import numpy as np
import torch
from models.diffusers.DiffusionAB import DiffusionAB
import constants as cst
from constants import LearningHyperParameter
from torch import nn
from models.diffusers.DiT.DiT import DiT

class GaussianDiffusion(nn.Module, DiffusionAB):
    """A diffusion model that uses Gaussian noise inspired from the IDDPM paper."""
    def __init__(self, config):
        super().__init__()
        self.dropout = config.HYPER_PARAMETERS[LearningHyperParameter.DROPOUT]
        self.cond_dropout = config.HYPER_PARAMETERS[LearningHyperParameter.CONDITIONAL_DROPOUT]
        self.num_timesteps = config.HYPER_PARAMETERS[LearningHyperParameter.NUM_TIMESTEPS]
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
        self.mse_losses = []
        self.vlb_losses = []
        if config.IS_AUGMENTATION_X:
            self.hidden_size = config.HYPER_PARAMETERS[LearningHyperParameter.AUGMENT_DIM]
            self.input_size = self.hidden_size
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
            self.num_timesteps,
            self.depth,
            self.num_heads,
            self.x_seq_size,
            self.mlp_ratio,
            self.cond_dropout_prob,
            self.type
        )
        self.alphas_cumprod = config.ALPHAS_CUMPROD
        self.betas = config.BETAS
        self.alphas = 1 - self.betas
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=cst.DEVICE), self.alphas_cumprod[:-1]])

        #calculation for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_var = (1.0 - self.alphas_cumprod) / (1.0 - self.alphas_cumprod) * self.betas
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

    def forward(self, x_t: torch.Tensor, context: Dict):
        assert 'noise' in context
        noise = context['noise']
        assert 'conditioning' in context
        cond = context['conditioning']
        assert 't' in context
        t = context['t']
        assert 'is_train' in context
        is_train = context['is_train']
        assert 'x_0' in context
        x_0 = context['x_0']
        return self.reverse_reparametrized(x_0, x_t, t, cond, noise, is_train)

    def forward_reparametrized(self, x_0: torch.Tensor, t: int, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert 'conditioning' in kwargs
        cond: torch.Tensor = kwargs['conditioning']
        x_t, noise = super().forward_reparametrized(x_0, t)
        return x_t, {'noise': noise, 'conditioning': cond}

    def reverse_reparametrized(self, x_0, x_t, t, cond, noise_true, is_train):
        '''
        Compute the reverse diffusion process for the current time step
        '''
        # Get the beta and alpha values for the current time step
        beta_t = self.betas[t]
        alpha_t = 1 - beta_t
        alpha_cumprod_t = self.alphas_cumprod[t]
        # Get the posterior variance of q(x_{t-1} | x_t, x_0) for the current time step
        beta_tilde_t = self.posterior_var[t]

        # Get the noise and v outputs from the neural network for the current time step
        noise_t, v = self.NN(x_t, cond, t)
        assert v.shape == noise_t.shape == x_t.shape

        # Compute the variance for the current time step using the formula from the IDDPM paper
        frac = (v + 1) / 2
        max_log = torch.log(beta_t)
        min_log = self.posterior_log_var_clipped[t]
        log_var_t = frac * max_log + (1 - frac) * min_log
        var_t = torch.exp(log_var_t)
        # Sample a standard normal random variable z
        z = torch.distributions.Normal(0, 1).sample(x_t.shape)
        # Compute x_{t-1} from x_t the reverse diffusion process for the current time step
        x_t = 1 / torch.sqrt(alpha_t) * (x_t - (beta_t / torch.sqrt(1 - alpha_cumprod_t) * noise_t)) + (var_t * z)

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
            clip_denoised=False,
        )
        # Append the loss to the vbl_losses list
        self.vlb_losses.append(L_vlb)
        # Return the reverse diffusion output and an empty dictionary
        return x_t, {}


    def _mse_loss(self, noise_t, noise_true):
        return torch.norm(noise_t - noise_true, p=2)


    def loss(self, true: torch.Tensor, recon: torch.Tensor, **kwargs) -> torch.Tensor:
        """Computes the loss taken from IDDPM."""
        L_simple = self.simple_losses[-1]
        L_vlb = self.vbl_losses[-1]
        #compute in media la differenza di ordini di grandezza tra L_simple e L_vlb
        L_hybrid = L_simple + self.lambda_*L_vlb
        return L_hybrid


    #ported from https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
    def _vlb_loss(
        self, noise_t, pred_log_var, x_0, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, true_log_variance_clipped = self._q_posterior_mean_var(x_0=x_0, x_t=x_t, t=t)
        pred_mean = self._p_mean(
            noise_t, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = self._normal_kl(
            true_mean, true_log_variance_clipped, pred_mean, pred_log_var
        )
        kl = self._mean_flat(kl) / np.log(2.0)

        '''
        #TODO change from image to time series
        L_0 = -self._discretized_gaussian_log_likelihood(
            x_0, means=pred_mean, log_scales=0.5 * pred_log_var
        )
        assert L_0.shape == x_0.shape
        L_0 = self._mean_flat(L_0) / np.log(2.0)

        # At the first timestep return the NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), L_0, kl)
        '''
        return kl

    def _p_mean(self, noise_t, x_t, t, clip_denoised=True, model_kwargs=None):
        '''
        Get the mean of the prior p(x_{t-1} | x_t).
        '''
        pred_mean = 1/torch.sqrt(self.alphas[t]) * (x_t - self.betas[t]*noise_t/torch.sqrt(1-self.alphas_cumprod[t]))
        return pred_mean

    def _q_posterior_mean_var(self, x_0, x_t, t):
        """
        Get the mean and variance of the posterior q(x_{t-1} | x_t, x_0).

        :param x_0: the initial image.
        :param x_t: the image at timestep t.
        :param t: the timestep.
        :return: a tuple (mean, variance).
        """
        true_mean = (
                self.posterior_mean_coef1[t] * x_0
                + self.posterior_mean_coef2[t] * x_t
        )
        #print check if requires grad of every term is false or true
        print("true_mean.requires_grad", true_mean.requires_grad)
        print("self.posterior_log_variance_clipped.requires_grad", self.posterior_log_variance_clipped.requires_grad)
        print("self.posterior_mean.requires_grad", self.posterior_mean_coef1.requires_grad)
        true_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return true_mean, true_log_variance_clipped

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
        tensor = None
        for obj in (mean1, logvar1, mean2, logvar2):
            if isinstance(obj, torch.Tensor):
                tensor = obj
                break
        assert tensor is not None, "at least one argument must be a Tensor"

        # Force variances to be Tensors. Broadcasting helps convert scalars to
        # Tensors, but it does not work for th.exp().
        logvar1, logvar2 = [
            x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
            for x in (logvar1, logvar2)
        ]

        return 0.5 * (
                -1.0
                + logvar2
                - logvar1
                + torch.exp(logvar1 - logvar2)
                + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
        )

    def _approx_standard_normal_cdf(self, x):
        """
        A fast approximation of the cumulative distribution function of the
        standard normal.
        """
        return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def _discretized_gaussian_log_likelihood(self, x, means, log_scales):
        """
        Compute the log-likelihood of a Gaussian distribution discretizing to a
        given image.

        :param x: the target images. It is assumed that this was uint8 values,
                  rescaled to the range [-1, 1].
        :param means: the Gaussian mean Tensor.
        :param log_scales: the Gaussian log stddev Tensor.
        :return: a tensor like x of log probabilities (in nats).
        """
        assert x.shape == means.shape == log_scales.shape
        centered_x = x - means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
        cdf_plus = self._approx_standard_normal_cdf(plus_in)
        min_in = inv_stdv * (centered_x - 1.0 / 255.0)
        cdf_min = self._approx_standard_normal_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
        )
        assert log_probs.shape == x.shape
        return log_probs

    def _mean_flat(self, tensor):
        """
        Take the mean over all non-batch dimensions.
        """
        return tensor.mean(dim=list(range(1, len(tensor.shape))))