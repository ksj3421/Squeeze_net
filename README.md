# Squeeze_net
Tensorflow implementation of SqueezeNet 
Original paper: [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size]
(https://arxiv.org/abs/1602.07360)
Purpose: First trial of implement deep-learning algorithm 
To do: By pass algorithm, training with real data(GPU Resource is not yet prepared)

[Hyperparameter]
initial learning rate = 0.04(linearly decreasing- AdagradOptimizer)
batch size initial= 512

[Layer]
1 Convolution Layer 
1 max-pooling Layer
4 Fire-module Layer
1 max-pooling Layer
4 Fire-module Layer(drop-out 50% at 8th Fire-module)
1 max-pooling Layer
1 Fire-module Layer
1 Convolution Layer 
1 avg-max-pooling Layer
  

