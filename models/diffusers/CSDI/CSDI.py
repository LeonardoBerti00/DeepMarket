from typing import Dict
import einops

import numpy as np
import torch
import torch.nn as nn

import constants as cst
import configuration
from models.diffusers.CSDI.Diffuser import CSDIEpsilon
from models.diffusers.DiffusionAB import DiffusionAB
from models.feature_augmenters.AbstractAugmenter import AugmenterAB

"""
    Adapted from https://github.com/ermongroup/CSDI/tree/main
"""
class CSDIDiffuser(nn.Module, DiffusionAB):
    
    def __init__(self, config, feature_augmenter):
        DiffusionAB.__init__(self, config)
        super(CSDIDiffuser, self).__init__()
        self.IS_AUGMENTATION = config.IS_AUGMENTATION
        self.device = cst.DEVICE
        self.target_dim = config.HYPER_PARAMETERS[cst.LearningHyperParameter.SIZE_ORDER_EMB]
        self.feature_augmenter = feature_augmenter
        self.num_steps = config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_DIFFUSIONSTEPS]
        self.embedding_dim = config.HYPER_PARAMETERS[cst.LearningHyperParameter.CSDI_DIFFUSION_STEP_EMB_DIM]
        self.n_heads = config.HYPER_PARAMETERS[cst.LearningHyperParameter.CSDI_N_HEADS]
        self.embedding_time_dim = config.HYPER_PARAMETERS[cst.LearningHyperParameter.CSDI_EMBEDDING_TIME_DIM]
        self.embedding_feature_dim = config.HYPER_PARAMETERS[cst.LearningHyperParameter.CSDI_EMBEDDING_FEATURE_DIM]
        self.layers = config.HYPER_PARAMETERS[cst.LearningHyperParameter.CSDI_LAYERS]
        self.side_dim = self.embedding_time_dim + 1
        
        self.betas = config.BETAS
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0, dtype=torch.float32)
        
        # TODO: change into dynamic input dim
        self.input_dim = 2
        self.mse_losses = []
        self.diffuser = CSDIEpsilon(self.num_steps, self.embedding_dim, self.side_dim, self.n_heads, self.input_dim, self.layers)
        
    def forward_reparametrized(self, input: torch.Tensor, diffusion_step: int, **kwargs):
        # here the conditioning and input are merged together again
        # because CSDI needs the mask on the entire input
        assert 'conditioning' in kwargs
        cond: torch.Tensor = kwargs['conditioning']
        # both conditioning and input need to have the same number of features
        assert cond.shape[-1] == input.shape[-1]
    
        whole_input = torch.cat([cond, input], dim=1)
                        
        x_t, noise = DiffusionAB.forward_reparametrized(self, whole_input, diffusion_step)
        x_t = x_t[:, self.cond_seq_size:, :]
        
        return x_t, {'noise_true': noise, 'conditioning': kwargs['conditioning']}
        
    def forward(self, x_T: torch.Tensor, context: Dict[str, torch.Tensor]):
        #print(context)
        assert 'cond_orders_aug' in context
        assert 'cond_augmenter' in context
        
        cond = context['cond_orders_aug']
        noise_true = context['noise_true']
        assert cond.shape[-1] == x_T.shape[-1]
        
        # condition mask
        whole_input = torch.cat([cond, x_T], dim=1)
        cond_mask = torch.zeros(whole_input.shape, device=cst.DEVICE)
        cond_mask[:, :self.cond_seq_size + 1, :] = 1
        
        # calculate conditioning and x_T
        cond = whole_input * cond_mask
        x_T = whole_input * (1 - cond_mask)
        
        # whole_input[:,:,0] is the timestamp of the data 
        side_info = self.get_side_info(whole_input[:,:,0], cond_mask).permute(0,2,3,1)
        
        cond = cond.unsqueeze(-1)
        x_T = x_T.unsqueeze(-1)
        total_input = torch.cat([cond, x_T], dim=-1) # (B,L,K,2)
        total_input = total_input.permute(0, 3, 2, 1)  # (B,2,K,L)
        B, _, K, L = total_input.shape
        '''
        if is_train:
            t = torch.randint(0, self.num_steps, [B]).to(self.device, non_blocking=True)
            recon = self.diffuser(total_input, side_info, t).permute(0,2,1).unsqueeze(0)
        else:
            recon = torch.zeros((self.num_steps, B, L, K))
            for set_t in range(self.num_steps):
                t = (torch.ones(B) * set_t).long().to(self.device, non_blocking=True)
                recon[set_t] = self.diffuser(total_input, side_info, t).permute(0,2,1)
        '''
        t = context["t"]
        noise_t = self.diffuser(total_input, side_info, t)
        # de-augment the conditioning and the input
        cond_mask = self.feature_augmenter.deaugment(cond_mask)
        context.update({'cond_mask': cond_mask})
        if (self.IS_AUGMENTATION):
            noise_t = self.feature_augmenter.deaugment(noise_t)
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        beta_t = einops.repeat(beta_t, 'b -> b l k', l=L, k=K)
        alpha_t = einops.repeat(alpha_t, 'b -> b l k', l=L, k=K)
        alpha_cumprod_t = einops.repeat(alpha_cumprod_t, 'b -> b l k', l=L, k=K)
        # Sample a standard normal random variable z
        z = torch.distributions.normal.Normal(0, 1).sample(x_T.shape).to(cst.DEVICE, non_blocking=True)
        #take the indexes for which t = 1
        indexes = torch.where(t == 0)
        z[indexes] = 0.0
        std_t = torch.sqrt(beta_t)
        x_T = x_T.squeeze(-1)
        # Compute x_{t-1} from x_t through the reverse diffusion process for the current time step
        x_recon = 1 / torch.sqrt(alpha_t) * (x_T - (beta_t / torch.sqrt(1 - alpha_cumprod_t) * noise_t)) + (std_t * z)
        # Compute the mean squared error loss between the noise and the true noise
        target_mask = torch.ones(cond_mask.shape, device=cst.DEVICE) - cond_mask
        residual = ((noise_t - noise_true) * target_mask)**2
        self.mse_losses.append(torch.mean(residual, dim=(2,3)))
        return x_recon, context

    def deaugment(self, input: torch.Tensor):
        final_output = torch.zeros((input.shape[0], input.shape[1], input.shape[2], self.target_dim), device=cst.DEVICE)
        if self.IS_AUGMENTATION and isinstance(self.diffuser, CSDIDiffuser):
            for i in range(input.shape[0]):
                final_output[i] = self.feature_augmenter.deaugment(input[i], {})
        return final_output

    def time_embedding(self, pos: torch.Tensor, d_model=128):
        """
            Time embedding as Eq. 13 in the paper
        """
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device, non_blocking=True)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device, non_blocking=True) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp: torch.Tensor, cond_mask: torch.Tensor):
        # we don't have discrete features, so we don't use them
        B, L, K = cond_mask.shape
        time_embed = self.time_embedding(observed_tp, self.embedding_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        time_embed = time_embed.permute(0, 3, 2, 1) # (B, *, K, L)
        side_mask = cond_mask.permute(0,2,1).unsqueeze(1) # (B, 1, K, L)
        side_info = torch.cat([time_embed, side_mask], dim=1)
        return side_info

    def loss(self, true: torch.Tensor, recon: torch.Tensor, **kwargs) -> torch.Tensor:
        assert 'cond_mask' in kwargs
        assert 'noise' in kwargs
        # return the mean for every instance in the batch B        
        L_simple = torch.stack(self.mse_losses).mean(dim=0)
        return L_simple

    def init_losses(self):
        self.mse_losses = []