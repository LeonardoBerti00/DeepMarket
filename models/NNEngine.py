import copy
import numpy
from torch import nn

from config import Configuration
import torch
import lightning as L
import constants as cst
from constants import LearningHyperParameter
import time

import wandb
from evaluation.evaluation_utils import JSD
from models.diffusers.GaussianDiffusion import GaussianDiffusion
from models.diffusers.csdi.CSDI import CSDIDiffuser
from utils.utils_models import pick_diffuser
from models.feature_augmenters.LSTMAugmenter import LSTMAugmenter
from lion_pytorch import Lion
from torch_ema import ExponentialMovingAverage
from models.diffusers.DiT.Sampler import LossSecondMomentResampler


class NNEngine(L.LightningModule):
    
    def __init__(self, config: Configuration, val_num_steps: int, test_num_steps: int, val_data, test_data, trainer):
        super().__init__()
        """
        This is the skeleton of the diffusion models.
        """
        self.trainer = trainer
        self.IS_WANDB = config.IS_WANDB
        self.lr = config.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE]
        self.optimizer = config.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER]
        self.momentum = config.HYPER_PARAMETERS[LearningHyperParameter.MOMENTUM]
        self.training = config.IS_TRAINING
        self.conditional_dropout = config.HYPER_PARAMETERS[LearningHyperParameter.CONDITIONAL_DROPOUT]
        self.batch_size = config.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE]
        self.IS_AUGMENTATION = config.IS_AUGMENTATION
        self.cond_type = config.COND_TYPE
        self.epochs = config.HYPER_PARAMETERS[LearningHyperParameter.EPOCHS]
        self.cond_seq_size = config.COND_SEQ_SIZE
        self.val_data, self.test_data = val_data, test_data
        self.val_num_batches = int(val_num_steps / self.batch_size) + 1
        self.test_num_batches = int(test_num_steps / self.batch_size) + 1
        self.train_losses, self.val_losses, self.test_losses = [], [], []
        self.val_ema_losses, self.test_ema_losses = [], []
        self.val_reconstructions, self.test_reconstructions = numpy.zeros((val_num_steps, cst.LEN_EVENT)), numpy.zeros((test_num_steps, cst.LEN_EVENT))
        self.val_ema_reconstructions, self.test_ema_reconstructions = numpy.zeros((val_num_steps, cst.LEN_EVENT)), numpy.zeros((test_num_steps, cst.LEN_EVENT))
        self.num_diffusionsteps = config.HYPER_PARAMETERS[LearningHyperParameter.NUM_DIFFUSIONSTEPS]
        self.alphas_cumprod, self.betas = config.ALPHAS_CUMPROD, config.BETAS


        # TODO: Why not choose this augmenter from the config?
        # TODO: make both conditioning as default to switch to nn.Identity
        if (self.IS_AUGMENTATION):
            self.feature_augmenter = LSTMAugmenter(config, cst.LEN_EVENT).to(device=cst.DEVICE)
            self.diffuser = pick_diffuser(config, config.CHOSEN_MODEL, self.feature_augmenter)
        else:
            self.diffuser = pick_diffuser(config, config.CHOSEN_MODEL, None)
        if (self.IS_AUGMENTATION and self.cond_type == 'full'):
            self.conditioning_augmenter = LSTMAugmenter(config, config.COND_SIZE).to(device=cst.DEVICE)
        elif (self.IS_AUGMENTATION and self.cond_type == 'only_event'):
            self.conditioning_augmenter = self.feature_augmenter

        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.999)
        self.sampler = LossSecondMomentResampler(self.num_diffusionsteps)
        self.lstm = nn.LSTM(config.COND_SIZE, config.HYPER_PARAMETERS[LearningHyperParameter.AUGMENT_DIM], num_layers=1, batch_first=True, dropout=0.1)


    def forward(self, cond, x_0, is_train):
        # save the real input for future
        real_input, real_cond = copy.deepcopy(x_0), copy.deepcopy(cond)

        self.t = torch.full(size=(x_0.shape[0],), fill_value=self.num_diffusionsteps - 1, device=cst.DEVICE)
        if isinstance(self.diffuser, CSDIDiffuser) and is_train:
            self.t = torch.randint(low=0, high=self.num_diffusionsteps - 1, size=(x_0.shape[0],), device=cst.DEVICE)
        elif isinstance(self.diffuser, GaussianDiffusion) and is_train:
            self.t, _ = self.sampler.sample(self.batch_size)

        if is_train:
            recon, context = self.single_step(cond, x_0, is_train, real_cond)
        else:
            for i in range(self.num_diffusionsteps):
                if i == 0 and self.IS_AUGMENTATION and self.cond_type == 'full':
                    cond = self.conditioning_augmenter.augment(cond)
                if i == self.num_diffusionsteps - 1:
                    pass
                recon, context = self.single_step(cond, x_0, is_train, real_cond)
                self.t -= 1

        return recon, context


    def single_step(self, cond, x_0, is_train, real_cond):
        # forward process
        x_t, context = self.diffuser.forward_reparametrized(x_0, self.t, **{"conditioning": cond})
        context.update({'x_t': copy.deepcopy(x_t)})
        x_t.requires_grad = True

        # augment
        x_t, cond = self.augment(x_t, cond, is_train)

        context.update({
            'is_train': is_train,
            't': self.t,
            'x_0': x_0,
            'conditioning_aug': cond,
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
        if self.IS_AUGMENTATION and self.cond_type == 'full' and is_train:
            cond = self.conditioning_augmenter.augment(cond)
            x_t = self.feature_augmenter.augment(x_t)
        elif self.IS_AUGMENTATION and self.cond_type == 'full' and not is_train:
            # if we are in test mode and cond type is full, we augment the conditioning only for the first step directly in forward
            x_t = self.feature_augmenter.augment(x_t)
        return x_t, cond

    def loss(self, real, recon, **kwargs):
        return self.diffuser.loss(real, recon, **kwargs)

    def training_step(self, input, batch_idx):
        if self.global_step == 0 and self.IS_WANDB:
            self._define_log_metrics()
        x_0 = input[1]
        cond = input[0]
        recon, reverse_context = self.forward(cond, x_0, is_train=True)
        reverse_context.update({'is_train': True})
        batch_losses = self.loss(x_0, recon, **reverse_context)
        batch_loss_mean = torch.mean(batch_losses)
        self.train_losses.append(batch_loss_mean.item())
        self.sampler.update_losses(self.t, batch_losses)
        if isinstance(self.diffuser, GaussianDiffusion):
            self.diffuser.init_losses()
        return batch_loss_mean

    def on_train_epoch_end(self) -> None:
        loss = sum(self.train_losses) / len(self.train_losses)
        self.log('train_loss', loss)
        print(f'train loss on epoch {self.current_epoch} is {loss}')



    def validation_step(self, input, batch_idx):
        x_0 = input[1]
        cond = input[0]
        recon, reverse_context = self.forward(cond, x_0, is_train=False)
        reverse_context.update({'is_train': False})
        batch_losses = self.loss(x_0, recon, **reverse_context)
        batch_loss_mean = torch.mean(batch_losses)
        self.val_losses.append(batch_loss_mean.item())
        if isinstance(self.diffuser, GaussianDiffusion):
            self.diffuser.init_losses()
        # Validation: with EMA
        with self.ema.average_parameters():
            recon, reverse_context = self.forward(cond, x_0, is_train=False)
            reverse_context.update({'is_train': False})
            batch_ema_losses = self.loss(x_0, recon, **reverse_context)
            ema_loss = torch.mean(batch_ema_losses)
            self.val_ema_losses.append(ema_loss)
        if isinstance(self.diffuser, GaussianDiffusion):
            self.diffuser.init_losses()
        return batch_loss_mean

    def on_validation_epoch_end(self) -> None:
        loss = sum(self.val_losses) / len(self.val_losses)
        loss_ema = sum(self.val_ema_losses) / len(self.val_ema_losses)
        self.log('val_loss', loss)
        self.log('val_ema_loss', loss_ema)
        print(f"\n val loss on epoch {self.current_epoch} is {loss}")
        print(f"\n val ema loss on epoch {self.current_epoch} is {loss_ema}")


    def test_step(self, input, batch_idx):
        x_0 = input[1]
        cond = input[0]
        recon, reverse_context = self.forward(cond, x_0, is_train=False)
        reverse_context.update({'is_train': False})
        batch_losses = self.loss(x_0, recon, **reverse_context)
        batch_loss_mean = torch.mean(batch_losses)
        self.test_losses.append(batch_loss_mean.item())
        recon = self._to_original_dim(recon)
        if batch_idx != self.test_num_batches - 1:
            self.test_reconstructions[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size] = recon.cpu().detach().numpy()
        else:
            self.test_reconstructions[batch_idx*self.batch_size:] = recon.cpu().detach().numpy()
        if isinstance(self.diffuser, GaussianDiffusion):
            self.diffuser.init_losses()
        # Testing: with EMA
        with self.ema.average_parameters():
            recon, reverse_context = self.forward(cond, x_0, is_train=False)
            reverse_context.update({'is_train': False})
            batch_ema_losses = self.loss(x_0, recon, **reverse_context)
            ema_loss = torch.mean(batch_ema_losses)
            self.test_ema_losses.append(ema_loss.item())
            recon = self._to_original_dim(recon)
            if batch_idx != self.test_num_batches - 1:
                self.test_ema_reconstructions[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size] = recon.cpu().detach().numpy()
            else:
                self.test_ema_reconstructions[batch_idx * self.batch_size:] = recon.cpu().detach().numpy()
        if isinstance(self.diffuser, GaussianDiffusion):
            self.diffuser.init_losses()
        return batch_loss_mean

    def on_test_end(self) -> None:
        loss = sum(self.test_losses) / len(self.test_losses)
        loss_ema = sum(self.test_ema_losses) / len(self.test_ema_losses)
        jsd_test = JSD(self.test_data, self.test_ema_reconstructions)
        jsd_test_ema = JSD(self.test_data, self.test_reconstructions)
        self.log('test_jsd_ema', jsd_test_ema)
        self.log('test_jsd', jsd_test.jsd_for_each_feature())
        self.log('test_loss', loss)
        self.log('test_ema_loss', loss_ema)
        print(f"\n test loss on epoch {self.current_epoch} is {loss}")
        print(f"\n test ema loss on epoch {self.current_epoch} is {loss_ema}")


    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        elif self.optimizer == 'LION':
            self.optimizer = Lion(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler}

    def inference_time(self, cond, x):
        t0 = time.time()
        _ = self(cond, x)
        torch.cuda.current_stream().synchronize()
        t1 = time.time()
        elapsed = t1 - t0
        # print("Inference for the model:", elapsed, "ms")
        return elapsed

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update()

    def _define_log_metrics(self):
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("val_ema_loss", summary="min")
        wandb.define_metric("test_loss", summary="min")
        wandb.define_metric("test_ema_loss", summary="min")

    def _to_original_dim(self, recon):
        out = torch.zeros((recon.shape[0], recon.shape[1], 5), device=cst.DEVICE)
        type = torch.argmax(recon[:, :, 1:5], dim=2)
        # type == 0 is add, type == 1 is cancel, type == 2 is deletion, type == 3 is execution
        direction = torch.argmax(recon[:, :, 7:9], dim=2)
        # direction == 1 is buy, direction == 0 is sell
        out[:, :, 0] = recon[:, :, 0]
        out[:, :, 1] = type
        out[:, :, 2] = recon[:, :, 5]
        out[:, :, 3] = recon[:, :, 6]
        out[:, :, 4] = direction
        return out