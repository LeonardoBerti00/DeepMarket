from einops import rearrange
import numpy as np
from torch import nn
import os
from lightning import LightningModule
import torch
import constants as cst
from constants import LearningHyperParameter
import matplotlib.pyplot as plt

import wandb
from models.diffusers.gaussian_diffusion import GaussianDiffusion
from utils.utils_models import pick_augmenter
from lion_pytorch import Lion
from torch_ema import ExponentialMovingAverage
from models.diffusers.TRADES.Sampler import LossSecondMomentResampler


class DiffusionEngine(LightningModule):
    
    def __init__(self, config):
        super().__init__()
        self.conditional_dropout = config.HYPER_PARAMETERS[LearningHyperParameter.CONDITIONAL_DROPOUT]
        self.IS_AUGMENTATION = config.IS_AUGMENTATION
        self.IS_WANDB = config.IS_WANDB
        self.augment_dim = config.HYPER_PARAMETERS[LearningHyperParameter.AUGMENT_DIM]
        self.cond_type = config.COND_TYPE
        self.cond_method = config.COND_METHOD
        self.cond_seq_size = config.HYPER_PARAMETERS[LearningHyperParameter.SEQ_SIZE] - config.HYPER_PARAMETERS[LearningHyperParameter.MASKED_SEQ_SIZE]
        self.reg_term_weight = config.HYPER_PARAMETERS[LearningHyperParameter.REG_TERM_WEIGHT]
        self.num_diffusionsteps = config.HYPER_PARAMETERS[LearningHyperParameter.NUM_DIFFUSIONSTEPS]
        self.size_type_emb = config.HYPER_PARAMETERS[LearningHyperParameter.SIZE_TYPE_EMB]
        self.size_order_emb = config.HYPER_PARAMETERS[LearningHyperParameter.SIZE_ORDER_EMB]
        self.chosen_model = config.CHOSEN_MODEL.value
        self.betas = config.BETAS
        self.training = config.IS_TRAINING
        self.test_batch_size = config.HYPER_PARAMETERS[LearningHyperParameter.TEST_BATCH_SIZE]
        self.epochs = config.HYPER_PARAMETERS[LearningHyperParameter.EPOCHS]
        self.seq_size = config.HYPER_PARAMETERS[LearningHyperParameter.SEQ_SIZE]
        self.train_losses, self.vlb_train_losses, self.simple_train_losses = [], [], []
        self.val_ema_losses, self.test_ema_losses = [], []
        self.min_loss_ema = np.inf
        self.min_train_loss = np.inf
        self.filename_ckpt = config.FILENAME_CKPT
        self.last_path_ckpt_ema = None
        self.optimizer = config.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER]
        self.lr = config.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE]
        self.cond_size = config.COND_SIZE
        if self.IS_AUGMENTATION:
            self.feature_augmenter = pick_augmenter(config.CHOSEN_AUGMENTER, self.size_order_emb, self.augment_dim, self.cond_size, self.cond_type, config.CHOSEN_COND_AUGMENTER, self.cond_method, self.chosen_model)
            self.diffuser = GaussianDiffusion(config, self.feature_augmenter).to(cst.DEVICE, non_blocking=True)
        else:
            self.diffuser = GaussianDiffusion(config, None).to(cst.DEVICE, non_blocking=True)
            
        self.type_embedder = nn.Embedding(3, self.size_type_emb, dtype=torch.float32)
        self.type_embedder.requires_grad_(False)
        self.type_embedder.weight.data = torch.tensor([[ 0.4438, -0.2984,  0.2888], [ 0.8249,  0.5847,  0.1448], [ 1.5600, -1.2847,  1.0294]], device=cst.DEVICE, dtype=torch.float32)
        if self.IS_WANDB:
            wandb.log({"type_embedder": self.type_embedder.weight.data}, step=0)
            
        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.999)
        self.ema.to(cst.DEVICE)
        self.sampler = LossSecondMomentResampler(self.num_diffusionsteps)
        self.vlb_sampler = LossSecondMomentResampler(self.num_diffusionsteps)
        self.simple_sampler = LossSecondMomentResampler(self.num_diffusionsteps)
        self.save_hyperparameters()
        

    def forward(self, cond_orders, x_0, cond_lob, is_train, batch_idx=None):
        # x_0 shape is (batch_size, seq_size=1, cst.LEN_ORDER=8)
        x_0, cond_orders = self.type_embedding(x_0, cond_orders)
        if is_train:
            self.t, _ = self.sampler.sample(x_0.shape[0])
            recon = self.single_step(cond_orders, x_0, cond_lob, batch_idx)
        else:
            self.t = torch.full(size=(x_0.shape[0],), fill_value=self.num_diffusionsteps-1, device=cst.DEVICE, dtype=torch.int64)
            for i in range(self.num_diffusionsteps-1, -1, -1):
                recon = self.single_step(cond_orders, x_0, cond_lob)
                self.t -= 1
        return recon

    def sample(self, **kwargs) -> torch.Tensor:
        cond_orders: torch.Tensor = kwargs['cond_orders']
        x_0: torch.Tensor = kwargs['x']
        cond_lob: torch.Tensor = kwargs['cond_lob']
        x_0, cond_orders = self.type_embedding(x_0, cond_orders)
        x_0 = torch.zeros_like(x_0)
        weights = self.sampler.weights()
        x_t = self.diffuser.sample(x_0, cond_orders, cond_lob, weights)
        return x_t


    def single_step(self, cond_orders, x_0, cond_lob, batch_idx=None):
        # forward process
        x_t, noise = self.diffuser.forward_reparametrized(x_0, self.t)
        if torch.isnan(x_t).any():
            print("before aug:", x_t.max())
        # augment
        x_t_aug, cond_orders, cond_lob = self.diffuser.augment(x_t, cond_orders, cond_lob)
        if torch.isnan(x_t_aug).any():
            print("after aug:", x_t_aug.max())
        weights = self.sampler.weights()
        x_recon = self.diffuser.ddpm_single_step(x_0, x_t_aug, x_t, self.t, cond_orders, noise, weights, cond_lob, batch_idx)
        # return the deaugmented denoised input and the reverse context
        return x_recon
    

    def type_embedding(self, x_0, cond):
        order_type = x_0[:, :, 1]
        order_type_emb = self.type_embedder(order_type.long())
        x_0 = torch.cat((x_0[:, :, :1], order_type_emb, x_0[:, :, 2:]), dim=2)
        cond_type = cond[:, :, 1]
        cond_depth_emb = self.type_embedder(cond_type.long())
        cond = torch.cat((cond[:, :, :1], cond_depth_emb, cond[:, :, 2:]), dim=2)
        return x_0, cond
    
    
    def loss(self):
        # regularization term to avoid order with negative size
        L_hybrid, L_simple, L_vlb = self.diffuser.loss()
        #print(f"hybrid loss: {L_hybrid.mean()}")
        #print(f"simple loss: {L_simple.mean()}")
        #print(f"vlb loss: {L_vlb.mean()}")
        return L_hybrid, L_simple, L_vlb


    def training_step(self, input, batch_idx):
        #print(batch_idx)
        #if batch_idx == 5:
        #    print("stop")
        if self.global_step == 0 and self.IS_WANDB:
            self._define_log_metrics()
        x_0 = input[1].contiguous()
        cond_orders = input[0].contiguous()
        cond_lob = input[2].contiguous()
        x_0.requires_grad_(True)
        cond_orders.requires_grad_(True)
        cond_lob.requires_grad_(True)
        if self.cond_type != 'full':
            cond_lob = None
        recon = self.forward(cond_orders, x_0, cond_lob, is_train=True, batch_idx=batch_idx)
        batch_loss, L_simple, L_vlb = self.loss()
        self.simple_train_losses.append(torch.mean(L_simple).item())
        self.vlb_train_losses.append(torch.mean(L_vlb).item())
        batch_loss_mean = torch.mean(batch_loss)
        self.train_losses.append(batch_loss_mean.item())
        self.sampler.update_losses(self.t, batch_loss[0])
        self.vlb_sampler.update_losses(self.t, L_vlb[0])
        self.simple_sampler.update_losses(self.t, L_simple[0])
        self.diffuser.init_losses()
        self.ema.update()
        if batch_idx % 1000 == 0:
            print(f"batch loss: {batch_loss_mean}")
        return batch_loss_mean

    def on_train_epoch_start(self) -> None:
        print(f'learning rate: {self.optimizer.param_groups[0]["lr"]}')

    def on_validation_start(self) -> None:
        loss = sum(self.train_losses) / len(self.train_losses)
        if isinstance(self.diffuser, GaussianDiffusion):
            L_simple = sum(self.simple_train_losses) / len(self.simple_train_losses)
            L_vlb = sum(self.vlb_train_losses) / len(self.vlb_train_losses)
            if self.IS_WANDB:
                wandb.log({'train loss simple': L_simple}, step=self.current_epoch + 1)
                wandb.log({'train loss vlb': L_vlb}, step=self.current_epoch + 1)
                wandb.log({'train_loss': loss}, step=self.current_epoch + 1)
                #Simple loss plot
                plt.plot(range(self.num_diffusionsteps), np.mean(self.simple_sampler._loss_history, axis=-1))
                plt.xlabel('num_diffusionsteps')
                plt.ylabel('Simple')
                wandb.log({"simple_loss": wandb.Image(plt)}, step=self.current_epoch + 1)
                plt.close()
                # VLB loss plot
                plt.plot(range(self.num_diffusionsteps), np.mean(self.vlb_sampler._loss_history, axis=-1))
                plt.xlabel('num_diffusionsteps')
                plt.ylabel('VLB')
                wandb.log({"vlb_loss": wandb.Image(plt)}, step=self.current_epoch + 1)
                plt.close()
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
        cond_orders = input[0]
        cond_lob = input[2]
        if self.cond_type != 'full':
            cond_lob = None
        # Validation: with EMA
        with self.ema.average_parameters():
            recon = self.forward(cond_orders, x_0, cond_lob, is_train=False)
            batch_loss, L_simple, L_vlb = self.loss()
            self.simple_val_losses.append(torch.mean(L_simple).item())
            self.vlb_val_losses.append(torch.mean(L_vlb).item())
            batch_loss_mean = torch.mean(batch_loss)
            self.val_ema_losses.append(batch_loss_mean.item())
        self.diffuser.init_losses()
        return batch_loss_mean


    def on_validation_epoch_end(self) -> None:
        loss_ema = sum(self.val_ema_losses) / len(self.val_ema_losses)

        # model checkpointing
        if loss_ema < self.min_loss_ema:
            # if the improvement is less than 0.01, we halve the learning rate
            if loss_ema - self.min_loss_ema > -0.002:
                self.optimizer.param_groups[0]["lr"] /= 2  
            self.min_loss_ema = loss_ema
            self.model_checkpointing(loss_ema)
        else:
            self.optimizer.param_groups[0]["lr"] /= 2

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
            self.optimizer = torch.optim.Adam(
                [
                    {'params': self.diffuser.parameters()},
                    {'params': self.type_embedder.parameters(), "lr": 0.01},
                ], 
                lr=self.lr
                )
        elif self.optimizer == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer == 'LION':
            self.optimizer = Lion(self.parameters(), lr=self.lr)
        return self.optimizer

    def _define_log_metrics(self):
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("val_ema_loss", summary="min")

    def model_checkpointing(self, loss):
        if self.last_path_ckpt_ema is not None:
            os.remove(self.last_path_ckpt_ema)
        filename_ckpt_ema = ("val_ema=" + str(round(loss, 3)) +
                             "_epoch=" + str(self.current_epoch) +
                             "_" + self.filename_ckpt +
                             ".ckpt"
                             )
        path_ckpt_ema = cst.DIR_SAVED_MODEL + "/" + str(self.chosen_model) + "/" + filename_ckpt_ema
        with self.ema.average_parameters():
            self.trainer.save_checkpoint(path_ckpt_ema)
        self.last_path_ckpt_ema = path_ckpt_ema


