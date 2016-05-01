import numpy as np


def net_2_wider_net(weights, bias, weights_next_layer,
                    noise_std=0.01,
                    new_layer_size=None,
                    split_max_weight_else_random=True):
    """
    Numpy implementation of net 2 wider net from http://arxiv.org/pdf/1511.05641.pdf

    Args:
        weights (numpy.matrix|numpy.ndarray): The weights for the layer
        bias (numpy.array): The bias for the layer
        weights_next_layer (numpy.matrix|numpy.ndarray): The weights for the next layer
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

    if bias.shape[0] != weights_next_layer.shape[0]:
        raise ValueError(
            'bias with shape %s must have same size as weight_next_layer first dimension which has shape %s' %
            (weights.shape, bias.shape))

    if new_layer_size is None:
        new_layer_size = bias.shape[0] + 1
    elif new_layer_size <= bias.shape[0]:
        raise ValueError('New layer size must be greater than current layer size')

    while bias.shape[0] < new_layer_size:
        weights, bias, weights_next_layer = _net_2_wider_net_increase_size_by_one(weights, bias,
                                                                                  weights_next_layer,
                                                                                  noise_std,
                                                                                  split_max_weight_else_random)

    return weights, bias, weights_next_layer


def _net_2_wider_net_increase_size_by_one(weights, bias, weights_next_layer,
                                          noise_std=0.01,
                                          split_max_weight_else_random=True):
    if split_max_weight_else_random:
        # find the node with the highest activation
        split_index = np.argmax((np.dot(np.ones(weights.shape[0]), weights)) + bias)
    else:
        # randomly select a node to split, a new node will be created with the same weights as this one.
        split_index = np.random.randint(0, weights.shape[1])

    # add split node weights to layer weights
    node_to_split_weights = weights[:, split_index]

    # add new node bias to bias
    new_bias = np.r_[bias, [bias[split_index]]]

    # reduce the output connections to the next layer by half for the split node and the new node
    # this means the activation of the next layer will remain unchanged
    output_weights_for_split_node = weights_next_layer[split_index, :] * .5

    # if we got an ndarry as input we need to pad it out
    if output_weights_for_split_node.ndim == 1:
        output_weights_for_split_node = np.reshape(output_weights_for_split_node,
                                                   (1, output_weights_for_split_node.shape[0]))

    if noise_std:
        weight_noise = np.random.normal(scale=noise_std, size=node_to_split_weights.shape)
        node_to_split_weights += weight_noise

        bias_noise = np.random.normal(scale=noise_std)
        new_bias[-1] += bias_noise
        new_bias[split_index] -= bias_noise

        output_weights_for_split_node += np.random.normal(scale=noise_std,
                                                          size=output_weights_for_split_node.shape)

    new_weights = np.c_[weights, node_to_split_weights]

    new_weights_next_layer = np.r_[weights_next_layer,
                                   output_weights_for_split_node]

    new_weights_next_layer[split_index, :] *= .5

    return new_weights, new_bias, new_weights_next_layer
