# Net2Net
numpy implementation of net 2 net from the paper Net2Net: Accelerating Learning via Knowledge Transfer http://arxiv.org/abs/1511.05641

# Requirements
- numpy

# Usage
Here is how you would use it to create a wider version of an existing layer

    import numpy as np
    
    weights = np.matrix([[1.0, 0.1, 0.5], [1.0, 0.1, 0.5]])
    bias = np.array([0.0, 0.0, 0.0])
    weights_next_layer = np.matrix([[1.0], [0.2], [0.5]])
    
    weights, bias, weights_next_layer = net_2_wider_net(weights, bias,
                                                      weights_next_layer,
                                                      new_layer_size=5)
Then simply use the new variables from then on.

Here is creating the weights and biases for a new layer using net 2 deeper net

    import numpy as np
    
    bias = np.array([0.0, 0.0, 0.0])
    
    next_layer_weights, next_layer_bias = net_2_deeper_net(bias)
