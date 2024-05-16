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
        self.conditional_dropout = config.HYPER_PARAMETERS[LearningHyperParameter.CONDITIONAL_DROPOUT]
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
        self.generator_lstm_input_dim = config.HYPER_PARAMETERS[GANHyperParameters.GENERATOR_LSTM_INPUT_DIM]
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
        

    def forward(self, x):
        return self.generator(x)
    
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)        
        
    def training_step(self, batch):
        input, _ = batch

        optimizer_g, optimizer_d = self.optimizers()

        # sample noise
        z = torch.randn(input.shape[0], self.hparams.generator_lstm_hidden_state_dim)
        z = z.type_as(input)

        # train generator
        # generate images
        self.toggle_optimizer(optimizer_g)
        self.generated_input = self(z)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(input.size(0), 1)
        valid = valid.type_as(input)

        # adversarial loss is binary cross-entropy
        g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        self.toggle_optimizer(optimizer_d)

        # how well can it label as real?
        valid = torch.ones(input.size(0), 1)
        valid = valid.type_as(input)

        real_loss = self.adversarial_loss(self.discriminator(input), valid)

        # how well can it label as fake?
        fake = torch.zeros(input.size(0), 1)
        fake = fake.type_as(input)

        fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)
        
        self.ema.update()
        return g_loss + d_loss

    def on_train_epoch_start(self) -> None:
        print(f'learning rate: {self.optimizer.param_groups[0]["lr"]}')  

    def validation_step(self, input, batch_idx):
        # Validation: with EMA
        with self.ema.average_parameters():
            generated = self.forward(input)
            fake = torch.zeros(generated.size(0), 1)
            fake = fake.type_as(generated)
            y_hat = self.discriminator(generated)
            batch_loss = self.adversarial_loss(y_hat, fake)
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

    #def on_before_zero_grad(self, *args, **kwargs):
    #    self.ema.update()

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


