import numpy as np


def net_2_deeper_net(bias, noise_std=0.01):
    """
    This is a similar idea to net 2 deeper net from http://arxiv.org/pdf/1511.05641.pdf
    Assumes that this is a linear layer that is being extended and also adds some noise

    Args:
        bias (numpy.array): The bias for the layer we are adding after
        noise_std (Optional float): The amount of normal noise to add to the layer.
            If None then no noise is added
            Default is 0.01
    Returns:
        (numpy.matrix, numpy.array)
        The first item is the weights for the new layer
        Second item is the bias for the new layer
    """
    new_weights = np.matrix(np.eye(bias.shape[0], dtype=bias.dtype))
    new_bias = np.zeros(bias.shape, dtype=bias.dtype)

    if noise_std:
        new_weights = new_weights + np.random.normal(scale=noise_std, size=new_weights.shape)
        new_bias = new_bias + np.random.normal(scale=noise_std, size=new_bias.shape)

    return new_weights, new_bias
