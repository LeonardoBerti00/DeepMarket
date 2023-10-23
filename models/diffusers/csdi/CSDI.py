from typing import Dict
from config import Configuration
from models.diffusers.DiffusionAB import DiffusionAB
from models.diffusers.csdi.Diffuser import CSDIEpsilon
import torch.nn as nn
import torch
import constants as cst

"""
    Adapted from https://github.com/ermongroup/CSDI/tree/main
"""
class CSDIDiffuser(nn.Module, DiffusionAB):
    
    def __init__(self, config: Configuration):
        DiffusionAB.__init__(self, config)
        super(CSDIDiffuser, self).__init__()
        
        self.device = cst.DEVICE
        self.target_dim = cst.LEN_EVENT
        
        self.num_steps = config.HYPER_PARAMETERS[cst.LearningHyperParameter.DIFFUSION_STEPS]
        self.embedding_dim = config.CSDI_HYPERPARAMETERS[cst.CSDIParameters.DIFFUSION_STEP_EMB_DIM]
        self.n_heads = config.CSDI_HYPERPARAMETERS[cst.CSDIParameters.N_HEADS]
        self.embedding_time_dim = config.CSDI_HYPERPARAMETERS[cst.CSDIParameters.EMBEDDING_TIME_DIM]
        self.embedding_feature_dim = config.CSDI_HYPERPARAMETERS[cst.CSDIParameters.EMBEDDING_FEATURE_DIM]
        self.layers = config.CSDI_HYPERPARAMETERS[cst.CSDIParameters.LAYERS]
        self.side_dim = self.embedding_time_dim + self.embedding_feature_dim + 1
        # TODO: change into dynamic input dim
        self.input_dim = 2
        self.diffuser = CSDIEpsilon(self.num_steps, self.embedding_dim, self.side_dim, self.n_heads, self.input_dim, self.layers)
        
    def reparametrized_forward(self, input: torch.Tensor, diffusion_steps: int, **kwargs):
        # here the conditioning and input are merged together again
        # because CSDI needs the mask on the entire input
        assert 'conditioning' in kwargs
        
        cond = kwargs['conditioning']
        whole_input = torch.cat([cond, input])
        
        cond_mask = torch.zeros(whole_input.shape)
        cond_mask[:, :len(cond), :] = 1
        
        x_t, eps = super().reparameterized_forward(whole_input, diffusion_steps)
        x_t = x_t * (1 - cond_mask)

        cond = whole_input * cond_mask
        return x_t, {'eps': eps, 'conditioning': cond, 'cond_mask': cond_mask }
        
        
    def forward(self, x_T: torch.Tensor, context: Dict[str, torch.Tensor]):
        assert 'conditioning' in context
        assert 'eps' in context
        assert 'cond_mask' in context
        
        cond = context['conditioning']
        eps = context['eps']
        cond_mask = context['cond_mask']
        is_train = context.get('is_train', True)
        set_t = context.get('set_t', -1)
        
        features = torch.cat([cond, x_T])
        # features[:,:,0] is the timestamp of the data            
        side_info = self.get_side_info(features[:,:,0], cond_mask, features)
        
        B = features.shape[0]
        if not is_train:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)

        total_input = torch.cat([cond, x_T], dim=1)

        return self.diffuser(total_input, side_info, t)  # (B,K,L)
        
    def time_embedding(self, pos, d_model=128):
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


    def get_side_info(self, observed_tp, cond_mask, features):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        
        features = features.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side_info = torch.cat([time_embed, features], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
        side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def loss(self, true: torch.Tensor, recon: torch.Tensor, **kwargs) -> torch.Tensor:
        if kwargs['is_train']:
            return self.calc_loss(true, recon, observed_mask=kwargs['observed_mask'],
                             side_info=kwargs['side_info'],
                             is_train=kwargs['is_train'],
                             set_t=kwargs['set_t'])
        else:
            return self.calc_loss_valid(true, recon, observed_mask=kwargs['observed_mask'],
                                        side_info=kwargs['side_info'],
                                        is_train=kwargs['is_train'])
    
    
    
    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps
    
    def calc_loss(self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1):
        B, K, L = observed_data.shape
        if not is_train:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        cond_obs  = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1-cond_mask) * noisy_data).unsqueeze(1)
        total_input = torch.cat([cond_obs, noisy_target], dim=1)
        
        predicted = self.diffuser(total_input, side_info, t)  # (B,K,L)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss
    
    
    


