from typing import Dict
import einops

import numpy as np
import torch
import torch.nn as nn

import constants as cst
import configuration
from models.diffusers.csdi.Diffuser import CSDIEpsilon
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
        self.embedding_dim = config.CSDI_HYPERPARAMETERS[cst.CSDIParameters.DIFFUSION_STEP_EMB_DIM]
        self.n_heads = config.CSDI_HYPERPARAMETERS[cst.CSDIParameters.N_HEADS]
        self.embedding_time_dim = config.CSDI_HYPERPARAMETERS[cst.CSDIParameters.EMBEDDING_TIME_DIM]
        self.embedding_feature_dim = config.CSDI_HYPERPARAMETERS[cst.CSDIParameters.EMBEDDING_FEATURE_DIM]
        self.layers = config.CSDI_HYPERPARAMETERS[cst.CSDIParameters.LAYERS]
        self.side_dim = self.embedding_time_dim + 1
        
        self.betas = config.BETAS
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0, dtype=torch.float32)
        
        # TODO: change into dynamic input dim
        self.input_dim = 2
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
        
        return x_t, {'noise': noise, 'conditioning': kwargs['conditioning']}
        
    def forward(self, x_T: torch.Tensor, context: Dict[str, torch.Tensor]):
        assert 'conditioning_aug' in context
        assert 'cond_augmenter' in context
        
        cond = context['conditioning_aug']
        
        assert cond.shape[-1] == x_T.shape[-1]
        
        is_train = context.get('is_train', True)
        cond_augmenter: AugmenterAB = context['cond_augmenter']
        
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
        if is_train:
            t = torch.randint(0, self.num_steps, [B]).to(self.device, non_blocking=True)
            recon = self.diffuser(total_input, side_info, t).permute(0,2,1).unsqueeze(0)
        else:
            recon = torch.zeros((self.num_steps, B, L, K))
            for set_t in range(self.num_steps):
                t = (torch.ones(B) * set_t).long().to(self.device, non_blocking=True)
                recon[set_t] = self.diffuser(total_input, side_info, t).permute(0,2,1)
        # de-augment the conditioning and the input
        cond_mask, _ = cond_augmenter.deaugment(cond_mask)
        context.update({'cond_mask': cond_mask})
        if (self.IS_AUGMENTATION):
            recon = self.deaugment(recon)
        return recon, context

    def deaugment(self, input: torch.Tensor):
        final_output = torch.zeros((input.shape[0], input.shape[1], input.shape[2], self.target_dim), device=cst.DEVICE)
        if self.IS_AUGMENTATION and isinstance(self.diffuser, CSDIDiffuser):
            for i in range(input.shape[0]):
                final_output[i], _ = self.feature_augmenter.deaugment(input[i], {})
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
        
        cond_mask: torch.Tensor = kwargs['cond_mask']
        noise: torch.Tensor = kwargs['noise']
        
        target_mask = torch.ones(cond_mask.shape, device=cst.DEVICE) - cond_mask
        noise = einops.repeat(noise, 'm n o -> k m n o', k=recon.shape[0])
        target_mask =  einops.repeat(target_mask, 'm n o -> k m n o', k=recon.shape[0])
        residual = ((noise - recon) * target_mask)**2
        # return the mean for every instance in the batch B
        return torch.mean(residual, dim=(2,3))