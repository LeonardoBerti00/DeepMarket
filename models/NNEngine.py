from config import Configuration
from models.diffusers.DiffusionAB import DiffusionAB
import torch
import lightning as L
import constants as cst
from constants import LearningHyperParameter
import time
from utils.utils_models import pick_diffuser
from models.feature_augmenters.LSTMAugmenter import LSTMAugmenter
from lion_pytorch import Lion
from torch_ema import ExponentialMovingAverage


class NNEngine(L.LightningModule):
    
    def __init__(self, config: Configuration):
        super().__init__()
        """
        This is the skeleton of the diffusion models.
        """
        self.diffuser: DiffusionAB = pick_diffuser(config, config.CHOSEN_MODEL)
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
        self.diffusion_steps = config.HYPER_PARAMETERS[LearningHyperParameter.DIFFUSION_STEPS]
        self.alphas_dash, self.betas = config.ALPHAS_DASH, config.BETAS
        self.IS_AUGMENTATION_X = config.IS_AUGMENTATION_X
        self.IS_AUGMENTATION_COND = config.IS_AUGMENTATION_COND

        if (self.IS_AUGMENTATION_X):
            self.augmenter = LSTMAugmenter(config, cst.LEN_EVENT)
        elif (self.IS_AUGMENTATION_COND and self.cond_type == 'full'):
            self.augmenter_cond = LSTMAugmenter(config, cst.COND_SIZE)

        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.999)


    def forward(self, cond, x_0):

        #print mean of both
        print(torch.mean(cond), torch.mean(x_0))

        # augment
        x_0, cond = self.augment(x_0, cond)

        # forward, if we want to compute x_t where 0 < t < T, just set diffusion_step to t
        x_T, context = self.diffuser.reparametrized_forward(x_0, self.diffusion_steps-1)

        # reverse
        recon = self.diffuser(x_T, context, cond)
        
        return recon

    def augment(self, x_0, cond):
        if self.IS_AUGMENTATION_X:
            x_0 = self.augmenter.augment(x_0)
        # x_0.shape = (batch_size, K, latent_dim)

        if self.IS_AUGMENTATION_COND and self.cond_type == 'full':
            cond = self.augmenter_cond.augment(cond)
        # cond.shape = (batch_size, cond_size, latent_dim)

        if self.IS_AUGMENTATION_COND and self.cond_type == 'only_event':
            cond = self.augmenter.augment(cond)
        return x_0, cond


    def training_step(self, input, batch_idx):
        x_0 = input[1]
        cond = input[0]
        recon = self.forward(cond, x_0)
        loss = self.loss(x_0, recon)
        self.log('train_loss', loss)
        self.train_losses.append(loss)
        return loss

    def validation_step(self, input, batch_idx):
        x_0 = input[1]
        cond = input[0]
        recon = self.forward(cond, x_0)
        loss = self.loss(x_0, recon)
        self.log('val_loss', loss)
        self.val_losses.append(loss)

        # Validation: with EMA
        # (1) saves original parameters before replacing with EMA version
        # (2) copies EMA parameters to model
        # (3) after exiting the `with`, restore original parameters to resume training later
        with self.ema.average_parameters():
            recon = self.forward(cond, x_0)
            ema_loss = self.loss(x_0, recon)
            self.val_ema_losses.append(ema_loss)
        return loss

    def test_step(self, input, batch_idx):
        x_0 = input[1]
        cond = input[0]
        recon = self.forward(cond, x_0)
        loss = self.loss(x_0, recon)
        self.log('test_loss', loss)
        self.test_losses.append(loss)

        # Testing: with EMA
        # (1) saves original parameters before replacing with EMA version
        # (2) copies EMA parameters to model
        # (3) after exiting the `with`, restore original parameters to resume training later
        with self.ema.average_parameters():
            recon = self.forward(cond, x_0)
            ema_loss = self.loss(x_0, recon)
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
