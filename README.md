# Squeeze_net
Tensorflow implementation of SqueezeNet 
Original paper: [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size] (https://arxiv.org/abs/1602.07360)
Purpose: First trial of implement deep-learning algorithm 
To do: By pass algorithm, training with real data(GPU Resource is not yet prepared)
[Hyperparameter]
initial learning rate = 0.04(linearly decreasing- AdagradOptimizer)
batch size initial= 512

[Layer]
Convolution Layer -1
max-pooling - 1
Fire-module -4
max-pooling - 1
Fire-module -4(drop-out 50%)
max-pooling - 1
Fire-module - 1
Convolution Layer - 1
avg-max-pooling-1
soft-max
  

