from typing import Dict

import numpy as np
import torch
import torch.nn as nn

import constants as cst
from config import Configuration
from models.diffusers.csdi.Diffuser import CSDIEpsilon
from models.diffusers.DiffusionAB import DiffusionAB
from models.feature_augmenters.AbstractAugmenter import AugmenterAB

"""
    Adapted from https://github.com/ermongroup/CSDI/tree/main
"""
class CSDIDiffuser(nn.Module, DiffusionAB):
    
    def __init__(self, config: Configuration):
        DiffusionAB.__init__(self, config)
        super(CSDIDiffuser, self).__init__()
        
        self.device = cst.DEVICE
        self.target_dim = cst.LEN_EVENT
        
        self.num_steps = config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_TIMESTEPS]
        self.embedding_dim = config.CSDI_HYPERPARAMETERS[cst.CSDIParameters.DIFFUSION_STEP_EMB_DIM]
        self.n_heads = config.CSDI_HYPERPARAMETERS[cst.CSDIParameters.N_HEADS]
        self.embedding_time_dim = config.CSDI_HYPERPARAMETERS[cst.CSDIParameters.EMBEDDING_TIME_DIM]
        self.embedding_feature_dim = config.CSDI_HYPERPARAMETERS[cst.CSDIParameters.EMBEDDING_FEATURE_DIM]
        self.layers = config.CSDI_HYPERPARAMETERS[cst.CSDIParameters.LAYERS]
        self.side_dim = self.embedding_time_dim + 1
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
        
        cond_mask = torch.zeros(whole_input.shape, device=cst.DEVICE)
        cond_mask[:, :self.cond_seq_size + 1, :] = 1
                        
        x_t, noise = DiffusionAB.forward_reparametrized(self, whole_input, diffusion_step)
        x_t = x_t * (1 - cond_mask)

        cond = whole_input * cond_mask
        return x_t, {'noise': noise, 'conditioning': cond, 'cond_mask': cond_mask, 'whole_input': whole_input}
        
        
    def forward(self, x_T: torch.Tensor, context: Dict[str, torch.Tensor]):
        assert 'conditioning' in context
        assert 'cond_mask' in context
        assert 'whole_input' in context
        assert 'cond_augmenter' in context
        
        cond = context['conditioning']
        whole_input = context['whole_input']
        cond_mask = context['cond_mask']
        is_train = context.get('is_train', True)
        cond_augmenter: AugmenterAB = context['cond_augmenter']

        # whole_input[:,:,0] is the timestamp of the data 
        side_info = self.get_side_info(whole_input[:,:,0], cond_mask).permute(0,2,3,1)
                
        cond = cond.unsqueeze(-1)
        x_T = x_T.unsqueeze(-1)
        total_input = torch.cat([cond, x_T], dim=-1) # (B,L,K,2)
        total_input = total_input.permute(0, 3, 2, 1)  # (B,2,K,L)

        B, _, K, L = total_input.shape
        if is_train:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
            recon = self.diffuser(total_input, side_info, t).permute(0,2,1).unsqueeze(0)
        else:
            recon = torch.zeros((self.num_steps, B, L, K))
            for set_t in range(self.num_steps):
                t = (torch.ones(B) * set_t).long().to(self.device)
                recon[set_t] = self.diffuser(total_input, side_info, t).permute(0,2,1)
        context.update({'cond_mask': cond_augmenter.deaugment(cond_mask)})
        return recon, context
        
    def time_embedding(self, pos: torch.Tensor, d_model=128):
        """
            Time embedding as Eq. 13 in the paper
        """
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
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
        cond_mask: torch.Tensor = kwargs['cond_mask']
        noise: torch.Tensor = kwargs['noise']
        target_mask = torch.ones(cond_mask.shape) - cond_mask
        loss_sum = 0
        for t in range(recon.shape[0]):
            residual = (noise - recon[t]) * target_mask
            num_eval = target_mask.sum()
            loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            loss_sum += loss
        return loss_sum / recon.shape[0]