import copy
from config import Configuration
import torch
import lightning as L
import constants as cst
from constants import LearningHyperParameter
import time

from models.diffusers.GaussianDiffusion import GaussianDiffusion
from models.diffusers.csdi.CSDI import CSDIDiffuser
from utils.utils_models import pick_diffuser
from models.feature_augmenters.LSTMAugmenter import LSTMAugmenter
from lion_pytorch import Lion
from torch_ema import ExponentialMovingAverage
from models.diffusers.DiT.Sampler import LossSecondMomentResampler


class NNEngine(L.LightningModule):
    
    def __init__(self, config: Configuration):
        super().__init__()
        """
        This is the skeleton of the diffusion models.
        """
        self.diffuser = pick_diffuser(config, config.CHOSEN_MODEL)
        self.lr = config.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE]
        self.optimizer = config.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER]
        self.momentum = config.HYPER_PARAMETERS[LearningHyperParameter.MOMENTUM]
        self.training = config.IS_TRAINING
        self.conditional_dropout = config.HYPER_PARAMETERS[LearningHyperParameter.CONDITIONAL_DROPOUT]
        self.batch_size = config.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE]
        self.cond_type = config.HYPER_PARAMETERS[LearningHyperParameter.COND_TYPE]
        self.epochs = config.HYPER_PARAMETERS[LearningHyperParameter.EPOCHS]
        self.train_losses = []
        self.val_losses = []
        self.val_ema_losses = []
        self.test_losses = []
        self.test_ema_losses = []
        self.num_timesteps = config.HYPER_PARAMETERS[LearningHyperParameter.NUM_TIMESTEPS]
        self.alphas_cumprod, self.betas = config.ALPHAS_CUMPROD, config.BETAS
        self.IS_AUGMENTATION_X = config.IS_AUGMENTATION_X
        self.IS_AUGMENTATION_COND = config.IS_AUGMENTATION_COND

        # TODO: Why not choose this augmenter from the config?
        # TODO: make both conditioning as default to switch to nn.Identity
        if (self.IS_AUGMENTATION_X):
            self.feature_augmenter = LSTMAugmenter(config, cst.LEN_EVENT)
        if (self.IS_AUGMENTATION_COND and self.cond_type == 'full'):
            self.conditioning_augmenter = LSTMAugmenter(config, config.COND_SIZE)

        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.999)
        self.sampler = LossSecondMomentResampler(self.num_timesteps)


    def forward(self, cond, x_0, is_train=True):
        # save the real input for future
        real_input, real_cond = copy.deepcopy(x_0), copy.deepcopy(cond)

        # augment
        x_0_aug, cond_aug = self.augment(x_0, cond)

        self.t = self.num_timesteps - 1
        if isinstance(self.diffuser, CSDIDiffuser):
            self.t = torch.randint(0, self.num_timesteps - 1) if is_train else self.t
        elif isinstance(self.diffuser, GaussianDiffusion):
            self.t, _ = self.sampler.sample(self.batch_size) if is_train else self.t

        # forward, if we want to compute x_t where 0 < t < T, just set diffusion_step to t
        x_T, context = self.diffuser.forward_reparametrized(x_0, self.t, **{"conditioning": cond})

        # reverse
        context.update({'is_train': is_train, 'cond_augmenter': self.conditioning_augmenter, 't': self.t, 'x_0': x_0})
        recon, reverse_context = self.diffuser(x_T, context)

        # de-augment the denoised input (recon)
        recon = self.deaugment(recon, real_input)

        reverse_context.update({'conditioning': real_cond})
        # return the deaugmented denoised input and the reverse context
        return recon, reverse_context


    # TODO: optimize (vectorized code)
    def deaugment(self, reversed_input: torch.Tensor, real_input: torch.Tensor):
        deaugmented = torch.zeros(reversed_input.shape[:-1] + (real_input.shape[-1],))
        for t in range(reversed_input.shape[0]):
            deaugmented[t] = self.feature_augmenter.deaugment(reversed_input[t])
        return deaugmented

    def augment(self, x_0: torch.Tensor, cond: torch.Tensor):
        if self.IS_AUGMENTATION_X:
            x_0 = self.feature_augmenter.augment(x_0)
        # x_0.shape = (batch_size, K, latent_dim)
        if self.IS_AUGMENTATION_COND and self.cond_type == 'full':
            cond = self.conditioning_augmenter.augment(cond)
        # cond.shape = (batch_size, cond_size, latent_dim)

        if self.IS_AUGMENTATION_COND and self.cond_type == 'only_event':
            cond = self.feature_augmenter.augment(cond)
        return x_0, cond


    def training_step(self, input, batch_idx):
        x_0 = input[1]
        cond = input[0]
        recon, reverse_context = self.forward(cond, x_0, is_train=True)
        reverse_context.update({'is_train': True})
        loss = self.loss(x_0, recon, **reverse_context)
        self.log('train_loss', loss)
        self.train_losses.append(loss)
        self.sampler.update_losses(self.t, loss)
        return loss

    def validation_step(self, input, batch_idx):
        x_0 = input[1]
        cond = input[0]
        recon, reverse_context = self.forward(cond, x_0, is_train=False)
        reverse_context.update({'is_train': False})
        loss = self.loss(x_0, recon, **reverse_context)
        self.log('val_loss', loss)
        self.val_losses.append(loss)

        # Validation: with EMA
        # (1) saves original parameters before replacing with EMA version
        # (2) copies EMA parameters to model
        # (3) after exiting the `with`, restore original parameters to resume training later
        with self.ema.average_parameters():
            recon, reverse_context = self.forward(cond, x_0, is_train=False)
            reverse_context.update({'is_train': False})
            ema_loss = self.loss(x_0, recon, **reverse_context)
            self.val_ema_losses.append(ema_loss)
        return loss

    def test_step(self, input, batch_idx):
        x_0 = input[1]
        cond = input[0]
        recon, reverse_context = self.forward(cond, x_0, is_train=False)
        reverse_context.update({'is_train': False})
        loss = self.loss(x_0, recon, **reverse_context)
        self.log('test_loss', loss)
        self.test_losses.append(loss)

        # Testing: with EMA
        # (1) saves original parameters before replacing with EMA version
        # (2) copies EMA parameters to model
        # (3) after exiting the `with`, restore original parameters to resume training later
        with self.ema.average_parameters():
            recon, reverse_context = self.forward(cond, x_0, is_train=False)
            reverse_context.update({'is_train': False})
            ema_loss = self.loss(x_0, recon, **reverse_context)
            self.val_ema_losses.append(ema_loss)
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
        print(f"\n train loss {loss}\n")

    def on_validation_epoch_end(self) -> None:
        loss = sum(self.val_losses) / len(self.val_losses)
        loss_ema = sum(self.val_ema_losses) / len(self.val_ema_losses)
        print(f"\n val loss {loss}")
        print(f"\n val ema loss {loss_ema}")

    def on_test_epoch_end(self) -> None:
        loss = sum(self.test_losses) / len(self.test_losses)
        loss_ema = sum(self.test_ema_losses) / len(self.test_ema_losses)
        print(f"\n test loss {loss}")
        print(f"\n test ema loss {loss_ema}")

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
