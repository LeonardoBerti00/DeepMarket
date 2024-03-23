import copy
import numpy as np
from torch import nn
import os

import configuration
import torch
import lightning as L
import constants as cst
from constants import LearningHyperParameter
import time
import matplotlib.pyplot as plt

import wandb
from models.diffusers.GaussianDiffusion import GaussianDiffusion
from models.diffusers.CSDI.CSDI import CSDIDiffuser
from utils.utils_data import unnormalize
from utils.utils_models import pick_diffuser, pick_augmenter
from models.feature_augmenters.LSTMAugmenter import LSTMAugmenter
from lion_pytorch import Lion
from torch_ema import ExponentialMovingAverage
from models.diffusers.CDT.Sampler import LossSecondMomentResampler


class NNEngine(L.LightningModule):
    
    def __init__(self, config):
        super().__init__()
        """
        This is the skeleton of the diffusion models.
        """
        self.IS_WANDB = config.IS_WANDB
        self.lr = config.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE]
        self.optimizer = config.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER]
        self.training = config.IS_TRAINING
        self.conditional_dropout = config.HYPER_PARAMETERS[LearningHyperParameter.CONDITIONAL_DROPOUT]
        self.test_batch_size = config.HYPER_PARAMETERS[LearningHyperParameter.TEST_BATCH_SIZE]
        self.IS_AUGMENTATION = config.IS_AUGMENTATION
        self.augment_dim = config.HYPER_PARAMETERS[LearningHyperParameter.AUGMENT_DIM]
        self.cond_type = config.COND_TYPE
        self.cond_size = config.COND_SIZE
        self.epochs = config.HYPER_PARAMETERS[LearningHyperParameter.EPOCHS]
        self.cond_seq_size = config.COND_SEQ_SIZE
        self.reg_term_weight = config.HYPER_PARAMETERS[LearningHyperParameter.REG_TERM_WEIGHT]
        self.train_losses, self.vlb_train_losses, self.simple_train_losses = [], [], []
        self.val_ema_losses, self.test_ema_losses = [], []
        self.min_loss_ema = 10000000
        self.filename_ckpt = config.FILENAME_CKPT
        self.save_hyperparameters()
        self.num_diffusionsteps = config.HYPER_PARAMETERS[LearningHyperParameter.NUM_DIFFUSIONSTEPS]
        self.size_type_emb = config.HYPER_PARAMETERS[LearningHyperParameter.SIZE_TYPE_EMB]
        self.size_order_emb = config.HYPER_PARAMETERS[LearningHyperParameter.SIZE_ORDER_EMB]
        self.betas = config.BETAS
        self.num_violations_price = 0
        self.num_violations_size = 0
        self.chosen_model = config.CHOSEN_MODEL.name
        self.last_path_ckpt_ema = None
        # TODO: Why not choose this augmenter from the config?
        # TODO: make both conditioning as default to switch to nn.Identity
        if self.IS_AUGMENTATION:
            self.feature_augmenter = pick_augmenter(config.CHOSEN_AUGMENTER, self.size_order_emb, self.augment_dim, self.chosen_model)
            self.diffuser = pick_diffuser(config, config.CHOSEN_MODEL, self.feature_augmenter)
        else:
            self.diffuser = pick_diffuser(config, config.CHOSEN_MODEL, None)
            
        if self.IS_AUGMENTATION and self.cond_type in ['full', 'only_lob']:
            self.conditioning_augmenter = pick_augmenter(config.CHOSEN_AUGMENTER, config.COND_SIZE, self.augment_dim, self.chosen_model)

        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.999)
        self.ema.to(cst.DEVICE)
        self.sampler = LossSecondMomentResampler(self.num_diffusionsteps)
        self.type_embedder = nn.Embedding(3, self.size_type_emb, dtype=torch.float32)
        self.vlb_sampler = LossSecondMomentResampler(self.num_diffusionsteps)
        self.simple_sampler = LossSecondMomentResampler(self.num_diffusionsteps)
        

    def forward(self, cond, x_0, is_train, batch_idx=None):
        # x_0 shape is (batch_size, seq_size=1, cst.LEN_EVENT_ONE_HOT=8)
        #x_0, cond = self.type_embedding(x_0, cond)
        real_input, real_cond = x_0.detach().clone(), cond.detach().clone()
        if is_train:
            if isinstance(self.diffuser, CSDIDiffuser) and is_train:
                self.t = torch.randint(low=1, high=self.num_diffusionsteps, size=(x_0.shape[0],), device=cst.DEVICE, dtype=torch.int64)
            elif isinstance(self.diffuser, GaussianDiffusion) and is_train:
                self.t, _ = self.sampler.sample(x_0.shape[0])
            recon, context = self.single_step(cond, x_0, is_train, real_cond)
        else:
            self.t = torch.full(size=(x_0.shape[0],), fill_value=self.num_diffusionsteps-1, device=cst.DEVICE, dtype=torch.int64)
            if self.IS_AUGMENTATION and self.cond_type == 'full':
                cond = self.conditioning_augmenter.augment(cond)
            for i in range(self.num_diffusionsteps-1, -1, -1):
                recon, context = self.single_step(cond, x_0, is_train, real_cond)
                self.t -= 1
        return recon, context

    def sampling(self, cond, x_0):
        self.t = torch.full(size=(x_0.shape[0],), fill_value=self.num_diffusionsteps-1, device=cst.DEVICE, dtype=torch.int64)
        #x_0, cond = self.type_embedding(x_0, cond)
        real_cond = cond.detach().clone()
        x_t, context = self.diffuser.forward_reparametrized(x_0, self.t, **{"conditioning": cond})
        context.update({'x_t': x_t.detach().clone()})
        for i in range(self.num_diffusionsteps-1, -1, -1):
            # augment
            x_t, cond = self.augment(x_t, real_cond, False)

            context.update({
                'is_train': False,
                't': self.t,
                'x_0': x_0,
                'conditioning_aug': cond,
                'weights': self.sampler.weights()
            })

            x_t, reverse_context = self.diffuser(x_t, context)

            reverse_context.update({'conditioning': real_cond})
            # return the deaugmented denoised input and the reverse context
            self.t -= 1
        return x_t

    def single_step(self, cond, x_0, is_train, real_cond):
        # forward process
        x_t, context = self.diffuser.forward_reparametrized(x_0, self.t, **{"conditioning": cond})
        context.update({'x_t': x_t.detach().clone()})
        # augment
        x_t, cond = self.augment(x_t, context['conditioning'], is_train)
        context.update({
            'is_train': is_train,
            't': self.t,
            'x_0': x_0,
            'conditioning_aug': cond,
            'weights': self.sampler.weights()
        })

        x_recon, reverse_context = self.diffuser(x_t, context)

        reverse_context.update({'conditioning': real_cond})
        # return the deaugmented denoised input and the reverse context
        return x_recon, reverse_context


    def augment(self, x_t: torch.Tensor, cond: torch.Tensor, is_train: bool):
        if self.IS_AUGMENTATION and self.cond_type == 'only_event':
            full_input = torch.cat([cond, x_t], dim=1)
            full_input_aug = self.feature_augmenter.augment(full_input)
            cond = full_input_aug[:, :self.cond_seq_size, :]
            x_t = full_input_aug[:, self.cond_seq_size:, :]
        elif self.IS_AUGMENTATION and (self.cond_type == 'only_lob' or self.cond_type == 'full'):
            x_t = self.feature_augmenter.augment(x_t)
            cond = self.conditioning_augmenter.augment(cond)
            assert self.augment_dim == self.cond_size
        return x_t, cond

    def type_embedding(self, x_0, cond):
        order_type = x_0[:, :, 1]
        type_emb = self.type_embedder(order_type.long())
        x_0 = torch.cat((x_0[:, :, :1], type_emb, x_0[:, :, 2:]), dim=2)

        if self.cond_type == 'only_event' or self.cond_type == 'full':
            cond_type = cond[:, :, 1]
            cond_depth_emb = self.type_embedder(cond_type.long())
            cond = torch.cat((cond[:, :, :1], cond_depth_emb, cond[:, :, 2:]), dim=2)
        return x_0, cond

    def loss(self, real, recon, **kwargs):
        regularization_term = torch.norm(recon[:, 0, 2], p=3) / recon.shape[0]
        L_hybrid, L_simple, L_vlb = self.diffuser.loss(real, recon, **kwargs)
        return L_hybrid + self.reg_term_weight * regularization_term, L_simple, L_vlb

    def training_step(self, input, batch_idx):
        if self.global_step == 0 and self.IS_WANDB:
            self._define_log_metrics()
        x_0 = input[1]
        full_cond = input[0]
        cond = self._select_cond(full_cond, self.cond_type)
        recon, reverse_context = self.forward(cond, x_0, is_train=True, batch_idx=batch_idx)
        reverse_context.update({'is_train': True})
        if isinstance(self.diffuser, GaussianDiffusion):
            batch_loss, L_simple, L_vlb = self.loss(x_0, recon, **reverse_context)
            self.simple_train_losses.append(torch.mean(L_simple).item())
            self.vlb_train_losses.append(torch.mean(L_vlb).item())
        else:
            batch_loss = self.loss(x_0, recon, **reverse_context)
        batch_loss_mean = torch.mean(batch_loss)
        self.train_losses.append(batch_loss_mean.item())
        self.sampler.update_losses(self.t, batch_loss[0])
        self.vlb_sampler.update_losses(self.t, L_vlb[0])
        self.simple_sampler.update_losses(self.t, L_simple[0])
        if isinstance(self.diffuser, GaussianDiffusion):
            self.diffuser.init_losses()
        self.ema.update()
        return batch_loss_mean

    def on_train_epoch_start(self) -> None:
        print(f'learning rate: {self.optimizer.param_groups[0]["lr"]}')

    def on_validation_start(self) -> None:
        loss = sum(self.train_losses) / len(self.train_losses)
        if isinstance(self.diffuser, GaussianDiffusion):
            L_simple = sum(self.simple_train_losses) / len(self.simple_train_losses)
            L_vlb = sum(self.vlb_train_losses) / len(self.vlb_train_losses)
            plt.plot(range(self.num_diffusionsteps), np.mean(self.simple_sampler._loss_history, axis=-1))
            plt.xlabel('num_diffusionsteps')
            plt.ylabel('Simple')
            plt.savefig(f'data/plot/simple_loss{self.current_epoch}.png')
            plt.close()  
            plt.plot(range(self.num_diffusionsteps), np.mean(self.vlb_sampler._loss_history, axis=-1))
            plt.xlabel('num_diffusionsteps')
            plt.ylabel('VLB')
            plt.savefig(f'data/plot/vlb_loss{self.current_epoch}.png')
            plt.close() 
            if self.IS_WANDB:
                wandb.log({'train loss simple': L_simple}, step=self.current_epoch + 1)
                wandb.log({'train loss vlb': L_vlb}, step=self.current_epoch + 1)
                wandb.log({'train_loss': loss}, step=self.current_epoch + 1)
            print(f'\ntrain loss simple on epoch {self.current_epoch} is {round(L_simple, 3)}')
            print(f'\ntrain loss vlb on epoch {self.current_epoch} is {round(L_vlb, 3)}')
            print(f'\ntrain loss on epoch {self.current_epoch} is {round(loss, 3)}')
        self.train_losses = []
        self.simple_train_losses = []
        self.vlb_train_losses = []
        self.val_ema_losses = []
        self.simple_val_losses = []
        self.vlb_val_losses = []
    

    def validation_step(self, input, batch_idx):
        x_0 = input[1]
        full_cond = input[0]
        cond = self._select_cond(full_cond, self.cond_type)
        # Validation: with EMA
        with self.ema.average_parameters():
            current_time = time.time()
            recon, reverse_context = self.forward(cond, x_0, is_train=False)
            reverse_context.update({'is_train': False})
            if isinstance(self.diffuser, GaussianDiffusion):
                batch_loss, L_simple, L_vlb = self.loss(x_0, recon, **reverse_context)
                self.simple_val_losses.append(torch.mean(L_simple).item())
                self.vlb_val_losses.append(torch.mean(L_vlb).item())
            else:
                batch_loss = self.loss(x_0, recon, **reverse_context)
            batch_loss_mean = torch.mean(batch_loss)
            self.val_ema_losses.append(batch_loss_mean.item())
        if isinstance(self.diffuser, GaussianDiffusion):
            self.diffuser.init_losses()
        return batch_loss_mean


    def on_validation_epoch_end(self) -> None:
        loss_ema = sum(self.val_ema_losses) / len(self.val_ema_losses)

        # model checkpointing
        if loss_ema < self.min_loss_ema:
            # if the improvement is less than 0.01, we halve the learning rate
            if loss_ema - self.min_loss_ema > -0.005:
                self.optimizer.param_groups[0]["lr"] /= 2  
            self.min_loss_ema = loss_ema
            self._model_checkpointing(loss_ema)

        if isinstance(self.diffuser, GaussianDiffusion):
            L_simple = sum(self.simple_val_losses) / len(self.simple_val_losses)
            L_vlb = sum(self.vlb_val_losses) / len(self.vlb_val_losses)
            if self.IS_WANDB:
                wandb.log({'val_loss_simple': L_simple}, step=self.current_epoch + 1)
                wandb.log({'val_loss_vlb': L_vlb}, step=self.current_epoch + 1)
            print(f'\nval loss simple on epoch {self.current_epoch} is {round(L_simple, 3)}')
            print(f'\nval loss vlb on epoch {self.current_epoch} is {round(L_vlb, 3)}')

        self.log('val_ema_loss', loss_ema)
        print(f"\n val ema loss on epoch {self.current_epoch} is {round(loss_ema, 3)}")
        

    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer == 'LION':
            self.optimizer = Lion(self.parameters(), lr=self.lr)
        return self.optimizer

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update()

    def _define_log_metrics(self):
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("val_ema_loss", summary="min")

    def _select_cond(self, cond, cond_type):
        if cond_type == 'only_event':
            cond = cond[:, :, :cst.LEN_EVENT]
        elif cond_type == 'only_lob':
            cond = cond[:, :, cst.LEN_EVENT:]
        return cond

    def _model_checkpointing(self, loss):
        if self.last_path_ckpt_ema is not None:
            os.remove(self.last_path_ckpt_ema)
        filename_ckpt_ema = ("val_ema=" + str(round(loss, 3)) +
                             "_epoch=" + str(self.current_epoch) +
                             "_" + self.filename_ckpt +
                             "_ema.ckpt"
                             )
        path_ckpt_ema = cst.DIR_SAVED_MODEL + "/" + str(self.chosen_model) + "/" + filename_ckpt_ema
        with self.ema.average_parameters():
            self.trainer.save_checkpoint(path_ckpt_ema)
        self.last_path_ckpt_ema = path_ckpt_ema


