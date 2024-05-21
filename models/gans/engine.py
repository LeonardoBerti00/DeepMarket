from typing import Union

import torch
from configuration import Configuration
import constants as cst
from constants import LearningHyperParameter, LearningHyperParameter

import wandb
from lion_pytorch import Lion
from torch_ema import ExponentialMovingAverage
from models.NNEngine import NNEngine
from models.gans.cgan import Generator, Discriminator

class GANEngine(NNEngine):
    
    def __init__(self, config: Configuration):
        super().__init__(config)
        # hyperparameters for the GAN
        self.seq_len = config.HYPER_PARAMETERS[LearningHyperParameter.SEQ_SIZE]
        self.order_feature_dim=config.HYPER_PARAMETERS[LearningHyperParameter.ORDER_FEATURES_DIM]
        self.market_feature_dim = config.HYPER_PARAMETERS[LearningHyperParameter.MARKET_FEATURES_DIM]
        # generator's hyperparameters
        self.generator_lstm_hidden_state_dim=config.HYPER_PARAMETERS[LearningHyperParameter.GENERATOR_LSTM_HIDDEN_STATE_DIM]
        self.generator_hidden_fc_dim=config.HYPER_PARAMETERS[LearningHyperParameter.GENERATOR_FC_HIDDEN_DIM]
        self.generator_kernel_conv=config.HYPER_PARAMETERS[LearningHyperParameter.GENERATOR_KERNEL_SIZE]
        self.generator_num_fc_layers=config.HYPER_PARAMETERS[LearningHyperParameter.GENERATOR_NUM_FC_LAYERS]
        self.generator_stride=config.HYPER_PARAMETERS[LearningHyperParameter.GENERATOR_STRIDE]
        # discriminator's hyperparameters
        self.discriminator_lstm_input_dim=self.market_feature_dim + self.order_feature_dim
        self.discriminator_lstm_hidden_state_dim=config.HYPER_PARAMETERS[LearningHyperParameter.DISCRIMINATOR_LSTM_HIDDEN_STATE_DIM]
        self.discriminator_hidden_fc_dim=config.HYPER_PARAMETERS[LearningHyperParameter.DISCRIMINATOR_FC_HIDDEN_DIM]
        self.discriminator_kernel_conv=config.HYPER_PARAMETERS[LearningHyperParameter.DISCRIMINATOR_KERNEL_SIZE]
        self.discriminator_num_fc_layers=config.HYPER_PARAMETERS[LearningHyperParameter.DISCRIMINATOR_NUM_FC_LAYERS]
        self.discriminator_stride=config.HYPER_PARAMETERS[LearningHyperParameter.DISCRIMINATOR_STRIDE]
        self.disc_channels = config.HYPER_PARAMETERS[LearningHyperParameter.DISCRIMINATOR_CHANNELS]
        self.gen_channels = config.HYPER_PARAMETERS[LearningHyperParameter.GENERATOR_CHANNELS]
        # wasserstein gan c parameter as in Arjovsky et al. “Wasserstein generative adversarial networks.” ICML 2017
        self.c = 1e-2
        self.save_hyperparameters()
        # need to suppress automatic optimization since
        # there are two optimizers here
        self.automatic_optimization = False
        
        self.generator: Generator = Generator(seq_len=self.seq_len,
                                              order_feature_dim=self.order_feature_dim,
                                              lstm_input_dim=self.market_feature_dim,
                                              lstm_hidden_state_dim=self.generator_lstm_hidden_state_dim,
                                              hidden_fc_dim=self.generator_hidden_fc_dim,
                                              kernel_conv=self.generator_kernel_conv,
                                              num_fc_layers=self.generator_num_fc_layers,
                                              stride=self.generator_stride,
                                              device=cst.DEVICE,
                                              channels = self.gen_channels
                                              )
        
        self.discriminator: Discriminator = Discriminator(seq_len=self.seq_len,
                                                          lstm_input_dim=self.discriminator_lstm_input_dim,
                                                          lstm_hidden_state_dim=self.discriminator_lstm_hidden_state_dim,
                                                          hidden_fc_dim=self.discriminator_hidden_fc_dim,
                                                          kernel_conv=self.discriminator_kernel_conv,
                                                          num_fc_layers=self.discriminator_num_fc_layers,
                                                          stride=self.discriminator_stride,
                                                          device=cst.DEVICE,
                                                          channels = self.disc_channels)
            
        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.999)
        self.ema.to(cst.DEVICE)
        

    def forward(self, noise: torch.Tensor, y: torch.Tensor):
        return self.generator(noise, y)     
        
    def training_step(self, batch, batch_idx):
        # y.shape -> (batch,  seq_len, num_features)
        # market_orders.shape -> (batch, seq_len, market_orders_features)
        y, market_orders = batch

        optimizer_g, optimizer_d = self.optimizers()
        # critic step
        d_loss = self.__critic_step(y, market_orders, optimizer_d)
        g_loss = self.__generator_step(y, market_orders, optimizer_g)
        
        self.ema.update()
        
        if batch_idx == 8000:
            self.model_checkpointing(d_loss.item())
        #check if both discriminator and generator are trianing properly
        #print(self.discriminator.lstm.weight_hh_l0.grad)
        #print(self.discriminator.lstm.weight_hh_l0.data.sum())
        #print(self.generator.lstm.weight_hh_l0.grad)
        #print(self.generator.lstm.weight_hh_l0.data.sum())
        
        return g_loss + d_loss
    
    def sample(self, **kwargs):
        noise: torch.Tensor = kwargs['noise']
        y: torch.Tensor = kwargs['y']
        generated_order = self(noise, y)
        return generated_order

    def __generator_step(self, y: torch.Tensor, market_orders: torch.Tensor, optimizer: Union[torch.optim.Optimizer, Lion]):
        noise = torch.randn(y.shape[0], 1, self.generator_lstm_hidden_state_dim).type_as(y)
        
        self.toggle_optimizer(optimizer)
        # generate a fake order from pure noise based on the conditioning of real y
        generated_order = self(noise, y)
        generated_order = self.post_process_order(generated_order)
        market_orders = torch.cat([market_orders[:, :-1, :], generated_order], dim=1)
        fake_logits = self.discriminator(y, market_orders)
        # * min - E_{Z~P_Z}[C(g(z))]
        loss = -fake_logits.mean().view(-1)

        self.log("g_loss", loss, prog_bar=True)
        self.manual_backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        self.untoggle_optimizer(optimizer)
        
        return loss
        
    def __critic_step(self, y: torch.Tensor, market_orders: torch.Tensor, optimizer: Union[torch.optim.Optimizer,Lion]):
        noise = torch.randn(y.shape[0], 1, self.generator_lstm_hidden_state_dim).type_as(y)
        generated_order = self(noise, y)
        generated_order = self.post_process_order(generated_order)
        # get the critic score on the real market
        real_logits = self.discriminator(y, market_orders)
        # replace the last order in the market with the generated one
        market_orders = torch.cat([market_orders[:, :-1, :], generated_order], dim=1)
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
            p.data.clamp_(-self.c, self.c)
            
        self.untoggle_optimizer(optimizer)
        
        return loss
    
    def post_process_order(self, generated_order):
        # Create new tensors instead of modifying generated_order in-place
        if self.chosen_stock == cst.Stocks.TSLA.name:
            first_column = torch.where(
                generated_order[:, :, 0] < 0.1, -1, 
                torch.where(generated_order[:, :, 0] < 0.35, 0, 1
                            )
                )
        else:
            first_column = torch.where(
                generated_order[:, :, 0] < 0.15, -1, 
                torch.where(generated_order[:, :, 0] < 0.95, 0, 1
                            )
                )
        third_column = torch.where(generated_order[:, :, 2] > 0, 1, -1)
        last_column = torch.where(generated_order[:, :, -1] > 0, 1, -1)

        # Concatenate the new tensors with the unmodified columns of generated_order
        new_order = torch.cat((first_column.unsqueeze(2), generated_order[:, :, 1].unsqueeze(2), third_column.unsqueeze(2), generated_order[:, :, 3:-1], last_column.unsqueeze(2)), dim=2)

        return new_order
        
    def on_train_epoch_start(self) -> None:
        print(f'gen_lr: {self.optimizer_g.param_groups[0]["lr"]} -- discr_lr = {self.optimizer_d.param_groups[0]["lr"]}')  

    def validation_step(self, batch):
        y, market_orders = batch
        # Validation: with EMA
        with self.ema.average_parameters():
            noise = torch.randn(y.shape[0], 1, self.generator_lstm_hidden_state_dim).type_as(y)
            generated_order = self(noise, y)
            market_orders = torch.cat([market_orders[:, 1:, :], generated_order], dim=1)
            batch_loss = self.discriminator(y, market_orders)
            batch_loss_mean = torch.mean(batch_loss)
            self.val_ema_losses.append(batch_loss_mean.item())
        return batch_loss_mean


    def on_validation_epoch_end(self) -> None:
        loss_ema = sum(self.val_ema_losses) / len(self.val_ema_losses)

        # model checkpointing
        if loss_ema < self.min_loss_ema:
            # if the improvement is less than 0.01, we halve the learning rate
            self.min_loss_ema = loss_ema
        else:
            self.optimizer_g.param_groups[0]["lr"] /= 1.5
            self.optimizer_d.param_groups[0]["lr"] /= 1.5
            
        self.model_checkpointing(loss_ema)
        self.log('val_ema_loss', loss_ema)
        print(f"\n val ema loss on epoch {self.current_epoch} is {loss_ema}")
        

    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            self.optimizer_g, self.optimizer_d = torch.optim.Adam(self.generator.parameters(), lr=self.lr), torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        elif self.optimizer == 'RMSprop':
            self.optimizer_g, self.optimizer_d = torch.optim.RMSprop(self.generator.parameters(), lr=self.lr), torch.optim.RMSprop(self.discriminator.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            self.optimizer_g, self.optimizer_d = torch.optim.SGD(self.generator.parameters(), lr=self.lr, momentum=0.9), torch.optim.SGD(self.discriminator.parameters, lr=self.lr, momentum=0.9)
        elif self.optimizer == 'LION':
            self.optimizer_g, self.optimizer_d = Lion(self.generator.parameters(), lr=self.lr), Lion(self.discriminator.parameters(), lr=self.lr)
        return self.optimizer_g, self.optimizer_d

    def _define_log_metrics(self):
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("val_ema_loss", summary="min")

    def model_checkpointing(self, loss):
        #if self.last_path_ckpt_ema is not None:
        #    os.remove(self.last_path_ckpt_ema)
        filename_ckpt_ema = ("val_ema=" + str(loss) +
                             "_epoch=" + str(self.current_epoch) +
                             "_" + self.filename_ckpt +
                             ".ckpt"
                             )
        path_ckpt_ema = cst.DIR_SAVED_MODEL + "/" + str(self.chosen_model) + "/" + filename_ckpt_ema
        with self.ema.average_parameters():
            self.trainer.save_checkpoint(path_ckpt_ema)
        self.last_path_ckpt_ema = path_ckpt_ema


