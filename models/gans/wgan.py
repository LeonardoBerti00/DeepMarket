import torch
import torch.nn as nn

from models.gans.utils import create_conv_layers

class Generator(nn.Module):
    
    def __init__(self,
                 seq_len: int = 256,
                 lstm_input_dim: int = 9,
                 order_feature_dim: int = 7, 
                 lstm_hidden_state_dim: int = 100,
                 hidden_fc_dim: int = 50,
                 kernel_conv: int = 3,
                 num_fc_layers: int = 2,
                 num_conv_layers: int = 2,
                 stride: int = 1,
                 device: str = 'cuda'):
        
        self.lstm_input_dim: int = lstm_input_dim
        self.lstm_hidden_state_dim: int = lstm_hidden_state_dim
        self.kernel_conv: int = kernel_conv
        self.stride: int = stride
        self.num_fc_layers: int = num_fc_layers
        self.num_conv_layers: int = num_conv_layers
        self.seq_len: int = seq_len
        self.order_feature_dim: int = order_feature_dim
        self.device: str = device
        
        self.lstm: nn.LSTM = nn.LSTM(input_size=lstm_input_dim, hidden_size=lstm_hidden_state_dim, batch_first=True, device=self.device)
        
        # initialize a number of linear layers without activation functions
        fc_layers: list[nn.Linear] = []
        input_dim: int = self.lstm_hidden_state_dim
        for _ in range(self.num_fc_layers):
            fc_layers.append(nn.Linear(in_features=input_dim, out_features=hidden_fc_dim), device=self.device)
            input_dim = hidden_fc_dim
            hidden_fc_dim //= 2
        self.fc_layers = nn.Sequential(fc_layers)
    
        self.fc_out_dim: int = hidden_fc_dim
        # initialize a number of conv1d layers with ReLU activation functions
        self.conv_layers = nn.Sequential(create_conv_layers(input_channels=self.seq_len,
                                                            input_size=self.fc_out_dim,
                                                            output_size=self.order_feature_dim,
                                                            device=self.device))
        self.tanh = nn.Tanh()
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond.shape = (batch_size, seq_len - 1, history_features)
        # concatenate the market history with the noise in the time dimension
        cond = torch.cat([cond, x], dim=1)
        # run through the lstm and take the hidden state
        _, h_cond = self.lstm(cond)
        # run through the fc layers
        h_cond = torch.cat([h_cond, cond])
        out_fc = self.fc_layers(h_cond)
        # run through batch norm, relu and convtrans1d
        out_conv = self.conv_layers(out_fc)
        # apply tanh
        return self.tanh(out_conv) 
        
class Discriminator(nn.Module):
    
    def __init__(self,
                 seq_len: int = 256,
                 lstm_input_dim: int = 16, 
                 lstm_hidden_state_dim: int = 100,
                 hidden_fc_dim: int = 50,
                 kernel_conv: int = 3,
                 num_fc_layers: int = 2,
                 num_conv_layers: int = 2,
                 stride: int = 1,
                 device: str = 'cuda'):
        
        
        self.lstm_input_dim: int = lstm_input_dim
        self.lstm_hidden_state_dim: int = lstm_hidden_state_dim
        self.kernel_conv: int = kernel_conv
        self.stride: int = stride
        self.num_fc_layers: int = num_fc_layers
        self.num_conv_layers: int = num_conv_layers
        self.seq_len: int = seq_len
        self.device: str = device
        
        self.lstm: nn.LSTM = nn.LSTM(input_size=lstm_input_dim, hidden_size=lstm_hidden_state_dim, batch_first=True, device=self.device)
        
        # initialize a number of linear layers without activation functions
        fc_layers: list[nn.Linear] = []
        input_dim: int = self.lstm_hidden_state_dim
        for _ in range(self.num_fc_layers):
            fc_layers.append(nn.Linear(in_features=input_dim, out_features=hidden_fc_dim, device=self.device))
            input_dim = hidden_fc_dim
            hidden_fc_dim //= 2
        self.fc_layers = nn.Sequential(fc_layers)
        self.fc_out_dim: int = hidden_fc_dim
        # initialize a number of conv1d layers with ReLU activation functions
        self.conv_layers = nn.Sequential(create_conv_layers(input_channels=self.seq_len,
                                                            input_size=self.fc_out_dim,
                                                            output_size=self.fc_out_dim // 4,
                                                            device=self.device))
        
        self.last_fc_layer = nn.Linear(in_features=self.fc_out_dim, out_features=1, device=self.device)
                
    def forwad(self, y: torch.Tensor, gen: torch.Tensor) -> torch.Tensor:
        # concatenate on the feature dimension
        x = torch.cat([y, gen], dim=-1)
        # run the lstm
        _, h_x = self.lstm(x)
        # run the linear layers
        fc_out = self.fc_layers(h_x)
        # run the convolution
        conv_out = self.conv_layers(fc_out)
        # run last layer to map to 1
        return self.last_fc_layer(conv_out)