import torch
import torch.nn as nn
from einops import rearrange
from utils.utils_gan import create_conv_layers

class Generator(nn.Module):
    
    def __init__(self,
                 seq_len: int = 256,
                 lstm_input_dim: int = 9,
                 order_feature_dim: int = 7, 
                 lstm_hidden_state_dim: int = 100,
                 hidden_fc_dim: int = 50,
                 kernel_conv: int = 3,
                 num_fc_layers: int = 2,
                 stride: int = 1,
                 device: str = 'cuda',
                 channels: list = []
                 ):
        super().__init__()
        
        self.lstm_input_dim: int = lstm_input_dim
        self.lstm_hidden_state_dim: int = lstm_hidden_state_dim
        self.kernel_conv: int = kernel_conv
        self.stride: int = stride
        self.num_fc_layers: int = num_fc_layers
        self.seq_len: int = seq_len
        self.order_feature_dim: int = order_feature_dim
        self.device: str = device
        self.channels: list = channels
        self.lstm = nn.LSTM(input_size=lstm_input_dim,
                            hidden_size=lstm_hidden_state_dim,
                            batch_first=True,
                            device=self.device)
        # initialize a number of linear layers without activation functions
        self.fc_layers = nn.Sequential()
        input_dim: int = self.lstm_hidden_state_dim
        self.fc_layers.append(nn.Linear(in_features=input_dim, out_features=hidden_fc_dim, device=self.device))
    
        self.fc_out_dim: int = hidden_fc_dim
        # initialize a number of conv1d layers with ReLU activation functions
        print("for generator")
        self.conv_layers, final_size = create_conv_layers(#channels=[2, 32, 16, 1],
                                                channels=self.channels,
                                                 input_size=self.fc_out_dim,
                                                 kernel_size=self.kernel_conv,
                                                 stride=self.stride,
                                                    padding = 0,
                                                 target_size=self.order_feature_dim,
                                                 device=self.device)
        self.fc_out = nn.Linear(in_features=final_size, out_features=order_feature_dim, device=self.device)
        self.tanh = nn.Tanh()
        
    def forward(self, noise: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # cond.shape = (batch_size, seq_len - 1, history_features)
        _, hidden = self.lstm(y)
        h_T, _ = hidden
        h_T = rearrange(h_T, 'l b h -> b l h')
        # run through the lstm and take the hidden state
        input_to_fc = torch.cat([h_T, noise], dim=1)
        # run through the fc layers
        out_fc = self.fc_layers(input_to_fc)
        # run through batch norm, relu and convtrans1d
        out_conv = self.conv_layers(out_fc)
        # apply tanh
        return self.tanh(out_conv) 
        
class Discriminator(nn.Module):
    
    def __init__(self,
                 seq_len: int = 256,
                 order_feature_dim: int = 7,
                 lstm_input_dim: int = 16, 
                 lstm_hidden_state_dim: int = 100,
                 hidden_fc_dim: int = 50,
                 kernel_conv: int = 3,
                 num_fc_layers: int = 2,
                 stride: int = 1,
                 device: str = 'cuda',
                 channels: list = []):
        super().__init__()
        
        self.lstm_input_dim: int = lstm_input_dim
        self.lstm_hidden_state_dim: int = lstm_hidden_state_dim
        self.kernel_conv: int = kernel_conv
        self.stride: int = stride
        self.num_fc_layers: int = num_fc_layers
        self.seq_len: int = seq_len
        self.order_feature_dim: int = order_feature_dim
        self.device: str = device
        self.channels: list = channels
        
        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=lstm_hidden_state_dim, batch_first=True, device=self.device)
        
        # initialize a number of linear layers without activation functions
        self.fc_layers = nn.Sequential()
        self.fc_layers.append(nn.Linear(in_features=self.lstm_hidden_state_dim, out_features=hidden_fc_dim, device=self.device))
        self.fc_out_dim: int = hidden_fc_dim
        print("for discriminator")
        # initialize a number of conv1d layers with ReLU activation functions
        self.conv_layers, current_size = create_conv_layers(channels=self.channels,
                                                 input_size=self.fc_out_dim,
                                                 kernel_size=self.kernel_conv,
                                                 stride=self.stride,
                                                 padding = 0,
                                                 target_size=1,
                                                 device=self.device)
        self.final_layer = nn.Linear(in_features=current_size, out_features=1, device=self.device)
                        
    def forward(self, y: torch.Tensor, market_orders: torch.Tensor) -> torch.Tensor:
        # run the lstm
        market_orders = torch.cat([y, market_orders], dim=-1)
        # run through the LSTM
        _, hidden = self.lstm(market_orders)
        h_T_next, _ = hidden
        h_T_next = rearrange(h_T_next, 'l b h -> b l h')
        # run the linear layers
        fc_out = self.fc_layers(h_T_next)
        # run the convolution
        conv_out = self.conv_layers(fc_out)
        # run last layer to map to 1
        return self.final_layer(conv_out)