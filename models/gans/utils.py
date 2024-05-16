from torch.nn import Conv1d, BatchNorm1d, ReLU

def create_conv_layers(input_channels, input_size, output_size, min_kernel_size=1, device='cuda'):
    conv_layers = []
    current_size = input_size
    output_channels = input_channels // 2  # Initial output channels
    while current_size > output_size:
        # Calculate stride (reduce size by half at each step)
        stride = max(1, (current_size - output_size) // 2 + 1)
        # Calculate kernel size (odd to maintain symmetric padding)
        kernel_size = max(min_kernel_size, current_size - stride * (output_size - 1))
        # Calculate padding to maintain output size
        padding = max(0, ((output_size - 1) * stride + kernel_size - current_size) // 2)
        # append the batch norm and relu
        conv_layers.append(BatchNorm1d(num_features=current_size), device=device)
        conv_layers.append(ReLU())
        # Create Conv1D 
        conv_layer = Conv1d(input_channels, output_channels, kernel_size, stride=stride, padding=padding, device=device)
        conv_layers.append(conv_layer)
        # Update current size for next iteration
        current_size = (current_size - kernel_size + 2 * padding) // stride + 1
        input_channels = output_channels  # Update input channels for next layer
        output_channels = max(output_channels // 2, 1)  # Decrease output channels by a factor of 2, minimum 1
    return conv_layers