from torch import nn
import torch
from models.diffusers.TRADES.bin import BiN
class MLPLOB(nn.Module):
    def __init__(self, 
                 hidden_dim: int,
                 num_layers: int,
                 seq_size: int,
                 num_features: int,
                 dataset_type: str
                 ) -> None:
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dataset_type = dataset_type
        self.layers = nn.ModuleList()
        self.order_type_embedder = nn.Embedding(3, 1)
        self.first_layer = nn.Linear(num_features, hidden_dim)
        self.norm_layer = BiN(num_features, seq_size)
        self.layers.append(self.first_layer)
        self.layers.append(nn.GELU())
        for i in range(num_layers):
            if i != num_layers-1:
                self.layers.append(MLP(hidden_dim, hidden_dim*4, hidden_dim))
                self.layers.append(MLP(seq_size, seq_size*4, seq_size))
            else:
                self.layers.append(MLP(hidden_dim, hidden_dim*2, hidden_dim//4))
                self.layers.append(MLP(seq_size, seq_size*2, seq_size//4))
                
        total_dim = (hidden_dim//4)*(seq_size//4)
        self.final_layers = nn.ModuleList()
        while total_dim > 128:
            self.final_layers.append(nn.Linear(total_dim, total_dim//4))
            self.final_layers.append(nn.GELU())
            total_dim = total_dim//4
        self.final_layers.append(nn.Linear(total_dim, 3))
    
    def forward(self, input):
        if self.dataset_type == "LOBSTER":
            continuous_features = torch.cat([input[:, :, :41], input[:, :, 42:]], dim=2)
            order_type = input[:, :, 41].long()
            order_type_emb = self.order_type_embedder(order_type).detach()
            x = torch.cat([continuous_features, order_type_emb], dim=2)
        else:
            x = input
        x = x.permute(0, 2, 1)
        x = self.norm_layer(x)
        x = x.permute(0, 2, 1)
        for layer in self.layers:
            x = layer(x)
            x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], -1)
        for layer in self.final_layers:
            x = layer(x)
        return x
        
        
class MLP(nn.Module):
    def __init__(self, 
                 start_dim: int,
                 hidden_dim: int,
                 final_dim: int
                 ) -> None:
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(final_dim)
        self.fc = nn.Linear(start_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, final_dim)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        residual = x
        x = self.fc(x)
        x = self.gelu(x)
        x = self.fc2(x)
        if x.shape[2] == residual.shape[2]:
            x = x + residual
        x = self.layer_norm(x)
        x = self.gelu(x)
        return x
    
