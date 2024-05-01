# Diffusion-Models-for-Time-Series
Diffusion Models for Time Series

TSLA:
limit order counts in the training set: 2019381 -> 0.49736
cancel order counts in the training set: 1919269 -> 0.4727
market order counts in the training set: 121515 -> 0.0299

INTC:
Number of order type 0:  7789392 -> 50.3%
Number of order type 1:  7199317 -> 46.5%
Number of order type 2:  484200 -> 3.1%

INTC:
size of train set:  torch.Size([15472909, 46])
size of val set:  torch.Size([1037221, 46])
size of test set:  torch.Size([1869236, 46])
total: 18,379,366

TSLA:
size of train set:  torch.Size([4465354, 46])
size of val set:  torch.Size([335236, 46])
size of test set:  torch.Size([618467, 46])
total: 5,419,057