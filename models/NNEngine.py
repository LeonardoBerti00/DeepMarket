import copy
import numpy

from config import Configuration
import torch
import lightning as L
import constants as cst
from constants import LearningHyperParameter
import time

import wandb
from evaluation.evaluation_utils import JSDCalculator, KSCalculator
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
        self.diffuser = pick_diffuser(config, config.CHOSEN_MODEL)
        self.lr = config.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE]
        self.optimizer = config.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER]
        self.momentum = config.HYPER_PARAMETERS[LearningHyperParameter.MOMENTUM]
        self.training = config.IS_TRAINING
        self.conditional_dropout = config.HYPER_PARAMETERS[LearningHyperParameter.CONDITIONAL_DROPOUT]
        self.batch_size = config.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE]
        self.cond_type = config.HYPER_PARAMETERS[LearningHyperParameter.COND_TYPE]
        self.epochs = config.HYPER_PARAMETERS[LearningHyperParameter.EPOCHS]
        self.cond_seq_size = config.COND_SEQ_SIZE
        self.val_data, self.test_data = val_data, test_data
        self.val_num_batches = int(val_num_steps / self.batch_size) + 1
        self.test_num_batches = int(test_num_steps / self.batch_size) + 1
        self.train_losses, self.val_losses, self.test_losses = [], [], []
        self.val_ema_losses, self.test_ema_losses = [], []
        self.val_reconstructions, self.test_reconstructions = numpy.zeros((val_num_steps, cst.LEN_EVENT)), numpy.zeros((test_num_steps, cst.LEN_EVENT))
        self.val_ema_reconstructions, self.test_ema_reconstructions = numpy.zeros((val_num_steps, cst.LEN_EVENT)), numpy.zeros((test_num_steps, cst.LEN_EVENT))
        self.num_timesteps = config.HYPER_PARAMETERS[LearningHyperParameter.NUM_TIMESTEPS]
        self.alphas_cumprod, self.betas = config.ALPHAS_CUMPROD, config.BETAS
        self.IS_AUGMENTATION = config.IS_AUGMENTATION

        # TODO: Why not choose this augmenter from the config?
        # TODO: make both conditioning as default to switch to nn.Identity
        if (self.IS_AUGMENTATION):
            self.feature_augmenter = LSTMAugmenter(config, cst.LEN_EVENT)
        if (self.IS_AUGMENTATION and self.cond_type == 'full'):
            self.conditioning_augmenter = LSTMAugmenter(config, config.COND_SIZE)
        elif (self.IS_AUGMENTATION and self.cond_type == 'only_event'):
            self.conditioning_augmenter = self.feature_augmenter

        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.999)
        self.sampler = LossSecondMomentResampler(self.num_timesteps)


    def forward(self, cond, x_0, is_train=True):
        # save the real input for future
        real_input, real_cond = copy.deepcopy(x_0), copy.deepcopy(cond)

        # augment
        x_0, cond = self.augment(x_0, cond)

        self.t = self.num_timesteps - 1
        if isinstance(self.diffuser, CSDIDiffuser):
            self.t = torch.randint(low=0, high=self.num_timesteps - 1, size=(x_0.shape[0],), device=cst.DEVICE) if is_train else self.t
        elif isinstance(self.diffuser, GaussianDiffusion):
            self.t, _ = self.sampler.sample(self.batch_size) if is_train else self.t

        # forward process
        x_t, context = self.diffuser.forward_reparametrized(x_0, self.t, **{"conditioning": cond})

        # augment
        #x_0_aug, x_t_aug, cond, = self.augment(x_0, x_t, cond)

        # reverse
        context.update({
            'is_train': is_train,
            'cond_augmenter': self.conditioning_augmenter,
            't': self.t,
            'x_0': x_0,
        })

        noise_recon, reverse_context = self.diffuser(x_t, context)


        # de-augment the denoised input (recon)
        noise_true = reverse_context['noise']
        noise_recon, noise_true = self.deaugment(noise_recon, noise_true)

        reverse_context.update({'conditioning': real_cond})
        # return the deaugmented denoised input and the reverse context
        return noise_recon, reverse_context

    '''
    def deaugment(self, input: torch.Tensor, noise: torch.Tensor):
        if self.IS_AUGMENTATION_X:
            input = self.feature_augmenter.deaugment(input)
            noise = self.feature_augmenter.deaugment(noise)
        return input, noise
        
    def augment(self, x_0: torch.Tensor, x_t: torch.Tensor, cond: torch.Tensor):
        if self.IS_AUGMENTATION_X:
            x_0 = self.feature_augmenter.augment(x_0)
            x_t = self.feature_augmenter.augment(x_t)
        # x_0.shape = (batch_size, K, latent_dim)
        if self.IS_AUGMENTATION_COND and self.cond_type == 'full':
            cond = self.conditioning_augmenter.augment(cond)
        # cond.shape = (batch_size, cond_size, latent_dim)
        if self.IS_AUGMENTATION_COND and self.cond_type == 'only_event':
            cond = self.feature_augmenter.augment(cond)
        return x_0, x_t, cond
    '''


    def deaugment(self, input: torch.Tensor):
        final_output = torch.zeros((input.shape[0], input.shape[1], input.shape[2], cst.LEN_EVENT), device=cst.DEVICE)
        if self.IS_AUGMENTATION and isinstance(self.diffuser, CSDIDiffuser):
            for i in range(input.shape[0]):
                final_output[i] = self.feature_augmenter.deaugment(input[i])
        return final_output

    def augment(self, x_0: torch.Tensor, cond: torch.Tensor):
        if self.IS_AUGMENTATION and self.cond_type == 'only_event':
            full_input = torch.cat([cond, x_0], dim=1)
            full_input_aug = self.feature_augmenter.augment(full_input)
            cond = full_input_aug[:, :self.cond_seq_size, :]
            x_0 = full_input_aug[:, self.cond_seq_size:, :]
        # x_0.shape = (batch_size, K, latent_dim)
        if self.IS_AUGMENTATION and self.cond_type == 'full':
            cond = self.conditioning_augmenter.augment(cond)
            x_0 = self.feature_augmenter.augment(x_0)
        # cond.shape = (batch_size, cond_size, latent_dim)
        return x_0, cond

    def _define_log_metrics(self):
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("val_ema_loss", summary="min")
        wandb.define_metric("test_loss", summary="min")
        wandb.define_metric("test_ema_loss", summary="min")

    def training_step(self, input, batch_idx):
        if self.global_step == 0 and self.IS_WANDB:
            self._define_log_metrics()
        x_0 = input[1]
        cond = input[0]
        recon, reverse_context = self.forward(cond, x_0, is_train=True)
        reverse_context.update({'is_train': True})
        loss = self.loss(x_0, recon, **reverse_context)
        self.train_losses.append(loss)
        self.sampler.update_losses(self.t, loss)
        return loss

    def validation_step(self, input, batch_idx):
        x_0 = input[1]
        cond = input[0]
        recon, reverse_context = self.forward(cond, x_0, is_train=False)
        reverse_context.update({'is_train': False})
        loss = self.loss(x_0, recon, **reverse_context)
        self.val_losses.append(loss)
        if batch_idx != self.val_num_batches - 1:
            self.val_reconstructions[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size] = recon.cpu().detach().numpy()
        else:
            self.val_reconstructions[batch_idx * self.batch_size:] = recon.cpu().detach().numpy()
        # Validation: with EMA
        with self.ema.average_parameters():
            recon, reverse_context = self.forward(cond, x_0, is_train=False)
            reverse_context.update({'is_train': False})
            ema_loss = self.loss(x_0, recon, **reverse_context)
            self.val_ema_losses.append(ema_loss)
            if batch_idx != self.val_num_batches - 1:
                self.val_ema_reconstructions[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size] = recon.cpu().detach().numpy()
            else:
                self.val_ema_reconstructions[batch_idx * self.batch_size:] = recon.cpu().detach().numpy()
        return loss

    def test_step(self, input, batch_idx):
        x_0 = input[1]
        cond = input[0]
        recon, reverse_context = self.forward(cond, x_0, is_train=False)
        reverse_context.update({'is_train': False})
        loss = self.loss(x_0, recon, **reverse_context)
        self.test_losses.append(loss)
        if batch_idx != self.test_num_batches - 1:
            self.test_reconstructions[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size] = recon.cpu().detach().numpy()
        else:
            self.test_reconstructions[batch_idx*self.batch_size:] = recon.cpu().detach().numpy()
        # Testing: with EMA
        with self.ema.average_parameters():
            recon, reverse_context = self.forward(cond, x_0, is_train=False)
            reverse_context.update({'is_train': False})
            ema_loss = self.loss(x_0, recon, **reverse_context)
            self.val_ema_losses.append(ema_loss)
            if batch_idx != self.test_num_batches - 1:
                self.test_ema_reconstructions[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size] = recon.cpu().detach().numpy()
            else:
                self.test_ema_reconstructions[batch_idx * self.batch_size:] = recon.cpu().detach().numpy()
        return loss

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

    def loss(self, input, recon, **kwargs):
        # Reconstruction loss is the mse between the input and the reconstruction
        return self.diffuser.loss(input, recon, **kwargs)

    def on_train_epoch_end(self) -> None:
        loss = sum(self.train_losses) / len(self.train_losses)
        self.log('train_loss', loss)
        print(f"\n train loss {loss}\n")

    def on_validation_epoch_end(self) -> None:


        loss = sum(self.val_losses) / len(self.val_losses)
        loss_ema = sum(self.val_ema_losses) / len(self.val_ema_losses)
        jsd_val = JSDCalculator(self.val_data, self.val_ema_reconstructions)
        ks_val = KSCalculator(self.val_data, self.val_ema_reconstructions)
        self.log('val_jsd', jsd_val.jsd_for_each_feature())
        self.log('val_ks', ks_val.calculate_ks())
        self.log('val_loss', loss)
        self.log('val_ema_loss', loss_ema)
        print(f"\n val loss {loss}")
        print(f"\n val ema loss {loss_ema}")
        print(f"\n val jsd for offset: {jsd_val.jsd_for_each_feature()[0]}")
        print(f"\n val jsd for type: {jsd_val.jsd_for_each_feature()[1]}")
        print(f"\n val jsd for size: {jsd_val.jsd_for_each_feature()[2]}")
        print(f"\n val jsd for direction: {jsd_val.jsd_for_each_feature()[4]}")
        print(f"\n val jsd for price: {jsd_val.jsd_for_each_feature()[3]}")
        print(f"\n val ks {ks_val.calculate_ks()}")


    def on_test_epoch_end(self) -> None:
        loss = sum(self.test_losses) / len(self.test_losses)
        loss_ema = sum(self.test_ema_losses) / len(self.test_ema_losses)
        jsd_test = JSDCalculator(self.test_data, self.test_ema_reconstructions)
        ks_test = KSCalculator(self.test_data, self.test_ema_reconstructions)
        self.log('test_jsd', jsd_test.jsd_for_each_feature())
        self.log('test_ks', ks_test.calculate_ks())
        self.log('test_loss', loss)
        self.log('test_ema_loss', loss_ema)
        print(f"\n test loss {loss}")
        print(f"\n test ema loss {loss_ema}")
        print(f"\n test jsd for offset: {jsd_test.jsd_for_each_feature()[0]}")
        print(f"\n test jsd for type: {jsd_test.jsd_for_each_feature()[1]}")
        print(f"\n test jsd for size: {jsd_test.jsd_for_each_feature()[2]}")
        print(f"\n test jsd for direction: {jsd_test.jsd_for_each_feature()[4]}")
        print(f"\n test jsd for price: {jsd_test.jsd_for_each_feature()[3]}")
        print(f"\n test ks {ks_test.calculate_ks()}")

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
