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

import wandb
from evaluation.quantitative.evaluation_utils import JSDCalculator
from models.diffusers.GaussianDiffusion import GaussianDiffusion
from models.diffusers.csdi.CSDI import CSDIDiffuser
from utils.utils_data import unnormalize
from utils.utils_models import pick_diffuser
from models.feature_augmenters.LSTMAugmenter import LSTMAugmenter
from lion_pytorch import Lion
from torch_ema import ExponentialMovingAverage
from models.diffusers.CDT.Sampler import LossSecondMomentResampler


class NNEngine(L.LightningModule):
    
    def __init__(self, config, test_num_steps: int, test_data):
        super().__init__()
        """
        This is the skeleton of the diffusion models.
        """
        self.test_num_steps = test_num_steps
        self.test_data = test_data
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
        self.train_losses, self.val_losses, self.test_losses = [], [], []
        self.val_ema_losses, self.test_ema_losses = [], []
        self.num_diffusionsteps = config.HYPER_PARAMETERS[LearningHyperParameter.NUM_DIFFUSIONSTEPS]
        self.size_depth_emb = config.HYPER_PARAMETERS[LearningHyperParameter.SIZE_DEPTH_EMB]
        self.size_order_emb = config.HYPER_PARAMETERS[LearningHyperParameter.SIZE_ORDER_EMB]
        self.betas = config.BETAS
        self.num_violations_price = 0
        self.num_violations_size = 0
        # TODO: Why not choose this augmenter from the config?
        # TODO: make both conditioning as default to switch to nn.Identity
        if self.IS_AUGMENTATION:
            self.feature_augmenter = LSTMAugmenter(config, self.size_order_emb).to(cst.DEVICE, non_blocking=True)
            self.diffuser = pick_diffuser(config, config.CHOSEN_MODEL, self.feature_augmenter)
        else:
            self.diffuser = pick_diffuser(config, config.CHOSEN_MODEL, None)
        if self.IS_AUGMENTATION and self.cond_type == 'full':
            self.conditioning_augmenter = LSTMAugmenter(config, config.COND_SIZE).to(cst.DEVICE, non_blocking=True)
        elif self.IS_AUGMENTATION and self.cond_type in ['only_event', 'only_lob']:
            self.conditioning_augmenter = self.feature_augmenter

        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.999)
        self.sampler = LossSecondMomentResampler(self.num_diffusionsteps)
        self.depth_embedder = nn.Embedding(cst.N_LOB_LEVELS, self.size_depth_emb)


    def forward(self, cond, x_0, is_train):
        x_0, cond = self.depth_embedding(x_0, cond)
        real_input, real_cond = x_0.detach().clone(), cond.detach().clone()
        self.t = torch.zeros(size=(x_0.shape[0],), device=cst.DEVICE, dtype=torch.int64)
        if isinstance(self.diffuser, CSDIDiffuser) and is_train:
            self.t = torch.randint(low=0, high=self.num_diffusionsteps - 1, size=(x_0.shape[0],), device=cst.DEVICE, dtype=torch.int64)
        elif isinstance(self.diffuser, GaussianDiffusion) and is_train:
            self.t, _ = self.sampler.sample(x_0.shape[0])

        if is_train:
            recon, context = self.single_step(cond, x_0, is_train, real_cond)
        else:
            for i in range(self.num_diffusionsteps):
                if i == 0 and self.IS_AUGMENTATION and self.cond_type == 'full':
                    cond = self.conditioning_augmenter.augment(cond)
                if i == self.num_diffusionsteps - 1:
                    pass
                recon, context = self.single_step(cond, x_0, is_train, real_cond)
                self.t += 1

        return recon, context


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
            'cond_augmenter': self.conditioning_augmenter
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
        elif self.IS_AUGMENTATION and self.cond_type == 'full' and is_train:
            cond = self.conditioning_augmenter.augment(cond)
            x_t = self.feature_augmenter.augment(x_t)
        elif self.IS_AUGMENTATION and self.cond_type == 'full' and not is_train:
            # if we are in test mode and cond type is full, we augment the conditioning only for the first step directly in forward
            x_t = self.feature_augmenter.augment(x_t)
        elif self.IS_AUGMENTATION and self.cond_type == 'only_lob':
            x_t = self.feature_augmenter.augment(x_t)
            assert self.augment_dim == self.cond_size
        return x_t, cond

    def depth_embedding(self, x_0, cond):
        depth = x_0[:, :, -1]
        depth_emb = self.depth_embedder(depth.long()).float()
        x_0 = torch.cat((x_0[:, :, :-1], depth_emb), dim=2)

        if self.cond_type == 'only_event':
            cond_depth = cond[:, :, -1]
            cond_depth_emb = self.depth_embedder(cond_depth.long()).float()
            cond = torch.cat((cond[:, :, :-1], cond_depth_emb), dim=2)
        return x_0, cond

    def loss(self, real, recon, **kwargs):
        return self.diffuser.loss(real, recon, **kwargs)

    def training_step(self, input, batch_idx):
        if self.global_step == 0 and self.IS_WANDB:
            self._define_log_metrics()
        x_0 = input[1]
        full_cond = input[0]
        cond = self._select_cond(full_cond, self.cond_type)
        recon, reverse_context = self.forward(cond, x_0, is_train=True)
        reverse_context.update({'is_train': True})
        batch_losses = self.loss(x_0, recon, **reverse_context)[0]
        batch_loss_mean = torch.mean(batch_losses)
        self.train_losses.append(batch_loss_mean.item())
        self.sampler.update_losses(self.t, batch_losses)
        if isinstance(self.diffuser, GaussianDiffusion):
            self.diffuser.init_losses()
        self.ema.update()
        return batch_loss_mean

    def on_validation_start(self) -> None:
        loss = sum(self.train_losses) / len(self.train_losses)
        if self.IS_WANDB:
            wandb.log({'train_loss': loss}, step=self.current_epoch + 1)
        print(f'\ntrain loss on epoch {self.current_epoch} is {loss}\n')

    def validation_step(self, input, batch_idx):
        x_0 = input[1]
        full_cond = input[0]
        cond = self._select_cond(full_cond, self.cond_type)
        recon, reverse_context = self.forward(cond, x_0, is_train=True)
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
            #recon = self._check_constraints(full_cond, recon)
            ema_loss = torch.mean(batch_ema_losses)
            self.val_ema_losses.append(ema_loss)
        if isinstance(self.diffuser, GaussianDiffusion):
            self.diffuser.init_losses()
        return batch_loss_mean

    def on_validation_epoch_end(self) -> None:
        self.val_loss = sum(self.val_losses) / len(self.val_losses)
        loss_ema = sum(self.val_ema_losses) / len(self.val_ema_losses)
        self.log('val_loss', self.val_loss)
        if self.IS_WANDB:
            wandb.log({'val_ema_loss': loss_ema}, step=self.current_epoch)
        print(f"\n val loss on epoch {self.current_epoch} is {self.val_loss}")
        print(f"\n val ema loss on epoch {self.current_epoch} is {loss_ema}\n")
        print(f"\n violations price on epoch {self.current_epoch} is {self.num_violations_price}\n")
        print(f"\n violations size on epoch {self.current_epoch} is {self.num_violations_size}\n")
        self.num_violations_price = 0
        self.num_violations_size = 0

    def on_test_start(self) -> None:
        self.test_reconstructions = np.zeros((self.test_num_steps, cst.LEN_EVENT))
        self.test_ema_reconstructions = np.zeros((self.test_num_steps, cst.LEN_EVENT))
        self.test_num_batches = int(self.test_num_steps / self.test_batch_size) + 1

    def test_step(self, input, batch_idx):
        x_0 = input[1]
        full_cond = input[0]
        cond = self._select_cond(full_cond, self.cond_type)
        recon, reverse_context = self.forward(cond, x_0, is_train=True)
        reverse_context.update({'is_train': False})
        batch_losses = self.loss(x_0, recon, **reverse_context)
        batch_loss_mean = torch.mean(batch_losses)
        self.test_losses.append(batch_loss_mean.item())
        recon = recon[:, 0, :]
        #recon = self._check_constraints(full_cond, recon)
        if batch_idx != self.test_num_batches - 1:
            self.test_reconstructions[batch_idx*self.test_batch_size:(batch_idx+1)*self.test_batch_size] = recon
        else:
            self.test_reconstructions[batch_idx*self.test_batch_size:] = recon
        if isinstance(self.diffuser, GaussianDiffusion):
            self.diffuser.init_losses()
        # Testing: with EMA
        with self.ema.average_parameters():
            recon, reverse_context = self.forward(cond, x_0, is_train=False)
            reverse_context.update({'is_train': False})
            batch_ema_losses = self.loss(x_0, recon, **reverse_context)
            ema_loss = torch.mean(batch_ema_losses)
            self.test_ema_losses.append(ema_loss.item())
            recon = recon[:, 0, :]
            #recon = self._check_constraints(full_cond, recon)
            if batch_idx != self.test_num_batches - 1:
                self.test_ema_reconstructions[batch_idx * self.test_batch_size:(batch_idx + 1) * self.test_batch_size] = recon
            else:
                self.test_ema_reconstructions[batch_idx * self.test_batch_size:] = recon
        if isinstance(self.diffuser, GaussianDiffusion):
            self.diffuser.init_losses()
        return batch_loss_mean

    def on_test_end(self) -> None:
        loss = sum(self.test_losses) / len(self.test_losses)
        loss_ema = sum(self.test_ema_losses) / len(self.test_ema_losses)
        jsd_test = JSDCalculator(self.test_data, self.test_ema_reconstructions)
        jsd_test_ema = JSDCalculator(self.test_data, self.test_reconstructions)
        if self.IS_WANDB:
            wandb.log({'test_jsd_ema_time': jsd_test_ema.calculate_jsd()[0]})
            wandb.log({'test_jsd_time': jsd_test.calculate_jsd()[0]})
            wandb.log({'test_jsd_ema_type': jsd_test_ema.calculate_jsd()[1]})
            wandb.log({'test_jsd_type': jsd_test.calculate_jsd()[1]})
            wandb.log({'test_jsd_ema_price': jsd_test_ema.calculate_jsd()[3]})
            wandb.log({'test_jsd_price': jsd_test.calculate_jsd()[3]})
            wandb.log({'test_jsd_ema_size': jsd_test_ema.calculate_jsd()[2]})
            wandb.log({'test_jsd_size': jsd_test.calculate_jsd()[2]})
            wandb.log({'test_loss': loss})
            wandb.log({'test_ema_loss': loss_ema})
            wandb.log({'test_violations_price': self.num_violations_price})
            wandb.log({'test_violations_size': self.num_violations_size})
        print(f"\n test loss on epoch {self.current_epoch} is {loss}\n")
        print(f"\n test ema loss on epoch {self.current_epoch} is {loss_ema}\n")
        print(f"\n violations price on epoch {self.current_epoch} is {self.num_violations_price}\n")
        print(f"\n violations size on epoch {self.current_epoch} is {self.num_violations_size}\n")
        np.save(cst.RECON_DIR + "/test_reconstructions.npy", self.test_reconstructions)
        np.save(cst.RECON_DIR + "/test_ema_reconstructions.npy", self.test_ema_reconstructions)


    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer == 'LION':
            self.optimizer = Lion(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler}

    def inference_time(self, cond, x):
        t0 = time.time()
        _ = self.forward(cond, x)
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

    def _select_cond(self, cond, cond_type):
        if cond_type == 'only_event':
            cond = cond[:, :, :cst.LEN_EVENT_ONE_HOT]
        elif cond_type == 'only_lob':
            cond = cond[:, :, cst.LEN_EVENT_ONE_HOT:]
        return cond

    def _check_constraints(self, event_and_lob, recon):
        event_and_lob = event_and_lob.cpu().detach().numpy()
        recon = recon.cpu().detach().numpy()
        lob, generated_events = self._to_original(event_and_lob, recon)
        for i in range(generated_events.shape[0]):
            price = generated_events[i, 3]
            size_event = generated_events[i, 2]
            type = generated_events[i, 1]
            if (type == 2 or type == 3 or type == 4):  # it is a deletion, execution or cancel
                # take the index of the lob with the same value of the price
                index = np.where(lob[i, :] == price)[0]
                # check if the price is in the lob so it is valid
                if (index.shape[0] == 0):
                    self.num_violations_price += 1
                    print(f"\nPROBLEM CONSTRAINT type: {type}\n")
                # check if the size is bigger than the size of the limit order
                size_limit_order = lob[i, index + 1]
                if (size_limit_order < size_event):
                    self.num_violations_size += 1
        return generated_events

    def _to_original(self, event_and_lob, recon):
        generated_events = np.zeros((recon.shape[0], cst.LEN_EVENT))
        type = np.argmax(recon[:, 1:4], axis=1)
        # type == 0 is add, type == 1 is cancel, type == 2 is deletion
        generated_events[:, 0] = recon[:, 0]
        generated_events[:, 1] = type
        generated_events[:, 2] = recon[:, 4]
        generated_events[:, 3] = recon[:, 5]

        generated_events[:, 3] = unnormalize(generated_events[:, 3], cst.TSLA_EVENT_MEAN_PRICE, cst.TSLA_EVENT_STD_PRICE)
        generated_events[:, 2] = unnormalize(generated_events[:, 2], cst.TSLA_EVENT_MEAN_SIZE, cst.TSLA_EVENT_STD_SIZE)
        generated_events[:, 0] = unnormalize(generated_events[:, 0], cst.TSLA_EVENT_MEAN_TIME, cst.TSLA_EVENT_STD_TIME)

        # we extract the last snapshot of the LOB before the event, that is i-1 in the sequence, if the event is at i
        lob = event_and_lob[:, -1, cst.LEN_EVENT_ONE_HOT:]
        lob[:, 0::2] = unnormalize(lob[:, 0::2], cst.TSLA_LOB_MEAN_PRICE_10, cst.TSLA_LOB_STD_PRICE_10)
        lob[:, 1::2] = unnormalize(lob[:, 1::2], cst.TSLA_LOB_MEAN_SIZE_10, cst.TSLA_LOB_STD_SIZE_10)

        # assert (generated_events.shape[0]+1 == lob.shape[0])
        # round price and size
        generated_events[:, 2] = np.around(generated_events[:, 2], decimals=0)
        generated_events[:, 3] = np.around(generated_events[:, 3], decimals=0)
        lob[:, 0::2] = np.around(lob[:, 0::2], decimals=0)
        lob[:, 1::2] = np.around(lob[:, 1::2], decimals=0)

        # return the order type and direction to the original value
        generated_events[:, 1] += 1
        generated_events = np.concatenate((generated_events, np.ones((generated_events.shape[0], 1))), axis=1)
        generated_events[:, 4] = np.where(generated_events[:, 2] < 0, -1, generated_events[:, 4])
        generated_events[:, 2] = np.abs(generated_events[:, 2])

        best_ask = lob[:, 0]
        best_bid = lob[:, 2]
        diff_prices_del_canc = []
        for i in range(generated_events.shape[0]):
            # if it is a cancel or delete order, we need to find the nearest price in the LOB
            if (generated_events[i, 1] == 2 or generated_events[i, 1] == 3):
                if (generated_events[i, 4] == 1):
                    #find the nearest price in the buy side
                    price = generated_events[i, 3]
                    buy_price_side = lob[i, 2::4]
                    difference = np.abs(buy_price_side - price)
                    violated_price = True
                    for j in range(difference.shape[0]):
                        if (difference[j] == 0):
                            violated_price = False
                    if violated_price:
                        min_diff_price = np.min(difference)
                        index_price = np.argmin(difference)
                        generated_events[i, 3] = buy_price_side[index_price]
                        diff_prices_del_canc.append(min_diff_price)

                else:
                    #find the nearest price in the sell side
                    price = generated_events[i, 3]
                    sell_price_side = lob[i, 0::4]
                    difference = np.abs(sell_price_side - price)
                    index = np.argmin(difference)
                    generated_events[i, 3] = sell_price_side[index]

            # if it is a buy order that is above the best ask, it means that it is a buy market order
            if generated_events[i, 4] == 1 and generated_events[i, 1] == 1 and generated_events[i, 3] >= best_ask[i]:
                generated_events[i, 1] = 4
                generated_events[i, 4] = -1
                generated_events[i, 3] = np.min(lob[i, 0::4][lob[i, 0::4] <= [generated_events[i, 3]]])

            elif generated_events[i, 4] == -1 and generated_events[i, 1] == 1 and generated_events[i, 3] <= best_bid[i]:
                generated_events[i, 1] = 4
                generated_events[i, 4] = 1
                generated_events[i, 3] = np.max(lob[i, 2::4][lob[i, 2::4] >= [generated_events[i, 3]]])

        return lob, generated_events