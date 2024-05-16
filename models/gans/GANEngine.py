import numpy as np
from torch import nn
import os

import torch
import lightning as L
from configuration import Configuration
import constants as cst
from constants import LearningHyperParameter, GANHyperParameters
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
import wandb
from models.diffusers.GaussianDiffusion import GaussianDiffusion
from lion_pytorch import Lion
from torch_ema import ExponentialMovingAverage
from models.gans.wgan import Generator, Discriminator


class GANEngine(L.LightningModule):
    
    def __init__(self, config: Configuration):
        super().__init__()
        """
        This is the skeleton of the gan models.
        """
        self.IS_WANDB = config.IS_WANDB
        self.IS_DEBUG = config.IS_DEBUG
        self.lr = config.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE]
        self.optimizer = config.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER]
        self.training = config.IS_TRAINING
        self.test_batch_size = config.HYPER_PARAMETERS[LearningHyperParameter.TEST_BATCH_SIZE]
        self.epochs = config.HYPER_PARAMETERS[LearningHyperParameter.EPOCHS]
        #self.p_norm = config.HYPER_PARAMETERS[LearningHyperParameter.P_NORM]
        self.train_losses, self.vlb_train_losses, self.simple_train_losses = [], [], []
        self.val_ema_losses, self.test_ema_losses = [], []
        self.min_loss_ema = 10000000
        self.filename_ckpt = config.FILENAME_CKPT
        # hyperparameters for the GAN
        self.seq_len = config.HYPER_PARAMETERS[GANHyperParameters.SEQ_LEN]
        self.order_feature_dim=config.HYPER_PARAMETERS[GANHyperParameters.ORDER_FEATURES_DIM],
        # generator's hyperparameters
        self.generator_lstm_input_dim = config.HYPER_PARAMETERS[GANHyperParameters.ORDER_FEATURES_DIM]
        self.generator_lstm_hidden_state_dim=config.HYPER_PARAMETERS[GANHyperParameters.GENERATOR_LSTM_HIDDEN_STATE_DIM]
        self.generator_hidden_fc_dim=config.HYPER_PARAMETERS[GANHyperParameters.GENERATOR_FC_HIDDEN_DIM]
        self.generator_kernel_conv=config.HYPER_PARAMETERS[GANHyperParameters.GENERATOR_KERNEL_SIZE]
        self.generator_num_fc_layers=config.HYPER_PARAMETERS[GANHyperParameters.GENERATOR_NUM_FC_LAYERS]
        self.generator_num_conv_layers=config.HYPER_PARAMETERS[GANHyperParameters.GENERATOR_NUM_CONV_LAYERS]
        self.generator_stride=config.HYPER_PARAMETERS[GANHyperParameters.GENERATOR_STRIDE]
        # discriminator's hyperparameters
        self.discriminator_lstm_input_dim=self.generator_lstm_input_dim + self.order_feature_dim,
        self.discriminator_lstm_hidden_state_dim=config.HYPER_PARAMETERS[GANHyperParameters.DISCRIMINATOR_LSTM_HIDDEN_STATE_DIM]
        self.discriminator_hidden_fc_dim=config.HYPER_PARAMETERS[GANHyperParameters.DISCRIMINATOR_FC_HIDDEN_DIM]
        self.discriminator_kernel_conv=config.HYPER_PARAMETERS[GANHyperParameters.DISCRIMINATOR_KERNEL_SIZE]
        self.discriminator_num_fc_layers=config.HYPER_PARAMETERS[GANHyperParameters.DISCRIMINATOR_NUM_FC_LAYERS]
        self.discriminator_num_conv_layers=config.HYPER_PARAMETERS[GANHyperParameters.DISCRIMINATOR_NUM_CONV_LAYERS]
        self.discriminator_stride=config.HYPER_PARAMETERS[GANHyperParameters.DISCRIMINATOR_STRIDE]
        # wasserstein gan c parameter as in Arjovsky et al. “Wasserstein generative adversarial networks.” ICML 2017
        self.c = 1e-2
        self.save_hyperparameters()
        self.num_violations_price = 0
        self.num_violations_size = 0
        self.chosen_model = config.CHOSEN_MODEL.name
        self.last_path_ckpt_ema = None
        
        self.generator: Generator = Generator(seq_len=self.seq_len,
                                              lstm_input_dim=self.generator_lstm_input_dim,
                                              lstm_hidden_state_dim=self.generator_lstm_hidden_state_dim,
                                              order_feature_dim=self.order_feature_dim,
                                              hidden_fc_dim=self.generator_hidden_fc_dim,
                                              kernel_conv=self.generator_kernel_conv,
                                              num_fc_layers=self.generator_num_fc_layers,
                                              num_conv_layers=self.generator_num_conv_layers,
                                              stride=self.generator_stride,
                                              device=cst.DEVICE)
        
        self.discriminator: Discriminator = Discriminator(seq_len=self.seq_len,
                                                          lstm_input_dim=self.discriminator_lstm_input_dim,
                                                          lstm_hidden_state_dim=self.discriminator_lstm_hidden_state_dim,
                                                          hidden_fc_dim=self.discriminator_hidden_fc_dim,
                                                          kernel_conv=self.discriminator_kernel_conv,
                                                          num_fc_layers=self.discriminator_num_fc_layers,
                                                          num_conv_layers=self.discriminator_num_conv_layers,
                                                          stride=self.discriminator_stride,
                                                          device=cst.DEVICE)
            
        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.999)
        self.ema.to(cst.DEVICE)
        

    def forward(self, noise: torch.Tensor, y: torch.Tensor):
        return self.generator(noise, y)     
        
    def training_step(self, batch):
        # y.shape -> (batch,  seq_len, num_features)
        # market_orders.shape -> (batch, seq_len, market_orders_features)
        y, market_orders = batch

        optimizer_g, optimizer_d = self.optimizers()
        # critic step
        d_loss = self.__critic_step(y, market_orders, optimizer_d)
        g_loss = self.__generator_step(y, market_orders, optimizer_g)
        
        self.ema.update()
        return g_loss + d_loss

    def __generator_step(self, y: torch.Tensor, market_orders: torch.Tensor, optimizer: torch.optim.Optimizer | Lion):
        noise = torch.randn(y.shape[0], 1, self.hparams.order_feature_dim).type_as(y)
        
        self.toggle_optimizer(optimizer)
        # generate a fake order from pure noise based on the conditioning of real y
        fake_order = self(noise, y)
        market_orders[:,-1,:] = fake_order
        fake_logits = self.discriminator(y, market_orders)
       
        # * min E_{x~P_X}[C(x)] - E_{Z~P_Z}[C(g(z))]
        loss = -fake_logits.mean().view(-1)

        self.log("g_loss", loss, prog_bar=True)
        self.manual_backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        self.untoggle_optimizer(optimizer)
        
        return loss
        
    def __critic_step(self, y: torch.Tensor, market_orders: torch.Tensor, optimizer: torch.optim.Optimizer | Lion):
        noise = torch.randn(y.shape[0], 1, self.hparams.order_feature_dim).type_as(y)
        generated_order = self(noise, y)
        # get the critic score on the real market
        real_logits = self.discriminator(y, market_orders)
        # replace the last order in the market with the generated one
        market_orders[:,-1,:] = generated_order
        # get the critic score on the market with the last order as fake
        fake_logits = self.discriminator(y, market_orders)
        # * max E_{x~P_X}[C(x)] - E_{Z~P_Z}[C(g(z))]
        loss = -(real_logits.mean() - fake_logits.mean())
        self.log("d_loss", loss, prog_bar=True)
        self.manual_backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        # * Gradient clippling
        for p in self.discriminator.parameters():
            p.data.clamp_(-self.hparams.c, self.hparams.c)
            
        self.untoggle_optimizer(optimizer)
        
        return loss
        
    def on_train_epoch_start(self) -> None:
        print(f'learning rate: {self.optimizer.param_groups[0]["lr"]}')  

    def validation_step(self, batch):
        y, market_orders = batch
        # Validation: with EMA
        with self.ema.average_parameters():
            x = torch.randn(y.shape[0], 1, self.hparams.order_feature_dim).type_as(y)
            generated = self(x, y)
            batch_loss = self.discriminator(generated, market_orders)
            batch_loss_mean = torch.mean(batch_loss)
            self.val_ema_losses.append(batch_loss_mean.item())
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
        else:
            self.optimizer.param_groups[0]["lr"] /= 2

        """if isinstance(self.diffuser, GaussianDiffusion):
            L_simple = sum(self.simple_val_losses) / len(self.simple_val_losses)
            L_vlb = sum(self.vlb_val_losses) / len(self.vlb_val_losses)
            if self.IS_WANDB:
                wandb.log({'val_loss_simple': L_simple}, step=self.current_epoch + 1)
                wandb.log({'val_loss_vlb': L_vlb}, step=self.current_epoch + 1)
            print(f'\nval loss simple on epoch {self.current_epoch} is {round(L_simple, 3)}')
            print(f'\nval loss vlb on epoch {self.current_epoch} is {round(L_vlb, 3)}')"""

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

    def _define_log_metrics(self):
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("val_ema_loss", summary="min")

    def _model_checkpointing(self, loss):
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


