import numpy as np


def net_2_wider_net(weights, bias, weight_next_layer,
                    noise_std=0.01,
                    new_layer_size=None,
                    split_max_weight_else_random=True):
    """
    Numpy implementation of net 2 wider net from http://arxiv.org/pdf/1511.05641.pdf

    Args:
        weights (numpy.matrix): The weights for the layer
        bias (numpy.array): The bias for the layer
        weight_next_layer (numpy.matrix): The weights for the next layer
        noise_std (Optional float): The amount of noise to add to the weights when we expand
            If None no noise is added
        new_layer_size (Optional int): The size the new layer should be. If none the size is set to 1 larger than the
            current layer
        split_max_weight_else_random (bool): If True we split by selecting the node with the largest activation.
            If False we split a random node

    Returns:
        (numpy.matrix, numpy.array, numpy.matrix): This tuple contains
        the new_weights, new_bias and new_weights_next_layer

        These will all be 1 size larger than the ones passed in as specified by net 2 net paper

    Raises:
        ValueError: If the weights shape second dimension doesn't equal the bias dimension
                    If the bias dimension doesnt equal the new_layer_weight first dimension
                    If the new_layer_size is not greater than the bias dimension
    """
    if weights.shape[1] != bias.shape[0]:
        raise ValueError('weights with shape %s must have same last dimension as bias which had shape %s' %
                         (weights.shape, bias.shape))

    if bias.shape[0] != weight_next_layer.shape[0]:
        raise ValueError(
            'bias with shape %s must have same size as weight_next_layer first dimension which has shape %s' %
            (weights.shape, bias.shape))

    if new_layer_size is None:
        new_layer_size = bias.shape[0] + 1
    elif new_layer_size <= bias.shape[0]:
        raise ValueError('New layer size must be greater than current layer size')

    while bias.shape[0] < new_layer_size:
        weights, bias, weight_next_layer = _net_2_wider_net_increase_size_by_one(weights, bias,
                                                                                 weight_next_layer,
                                                                                 noise_std,
                                                                                 split_max_weight_else_random)

    return weights, bias, weight_next_layer


def _net_2_wider_net_increase_size_by_one(weights, bias, weight_next_layer,
                                          noise_std=0.01,
                                          split_max_weight_else_random=True):
    if split_max_weight_else_random:
        # find the node with the highest activation
        split_index = np.argmax((np.ones(weights.shape[0]) * weights) + bias)
    else:
        split_index = np.random.randint(0, weights.shape[1])

    node_to_split_weights = weights[:, split_index]
    new_bias = np.r_[bias, [bias[split_index]]]
    new_weight_next_layer = weight_next_layer[split_index, :] * .5

    if noise_std:
        weight_noise = np.random.normal(scale=noise_std, size=node_to_split_weights.shape)
        node_to_split_weights += weight_noise

        bias_noise = np.random.normal(scale=noise_std)
        new_bias[-1] += bias_noise
        new_bias[split_index] -= bias_noise

        new_weight_next_layer += np.random.normal(scale=noise_std,
                                                  size=new_weight_next_layer.shape)

    new_weights = np.c_[weights, node_to_split_weights]
    new_weight_next_layer = np.r_[weight_next_layer,
                                  new_weight_next_layer]
    new_weight_next_layer[split_index, :] *= .5

    return new_weights, new_bias, new_weight_next_layer
