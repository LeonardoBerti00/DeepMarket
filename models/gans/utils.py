import torch
import torch.nn as nn
from torch.nn import Sequential, Conv1d, BatchNorm1d, ReLU, Linear

def create_conv_layers(channels, input_size, target_size, kernel_size=1, device='cuda'):
    """
        Creates a sequential model of convolutional layers to progressively reduce the input size and the number of channels to 1,
        followed by a linear layer to map the final feature size to the desired output size.

        Args:
            input_channels (int): The number of input channels.
            input_size (int): The initial size of the input.
            kernel_size (int, optional): The kernel size for the convolutional layers. Default is 1.
            device (str, optional): The device to which the layers are to be moved ('cuda' or 'cpu'). Default is 'cuda'.

        Returns:
            nn.Sequential: A sequential model containing the convolutional layers, batch normalization layers, ReLU activations,
                        and a final linear layer to map the feature size to the output size.
    """
    conv_layers = Sequential()
    current_size = input_size
    for i in range(len(channels) - 1):
        stride = 2 if current_size > 1 else 1
        padding = (kernel_size - 1) // 2
        current_size = (current_size + 2 * padding - kernel_size) // stride + 1
        
        conv_layers.append(BatchNorm1d(num_features=channels[i], device=device))
        conv_layers.append(ReLU())
        conv_layers.append(Conv1d(in_channels=channels[i],
                                  out_channels=channels[i+1],
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  device=device))
    if current_size != target_size:
        conv_layers.append(nn.Linear(current_size, target_size, device=device))

    return conv_layers, current_size
