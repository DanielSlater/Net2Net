import numpy as np
from unittest import TestCase

from net2net.net_2_deeper_net import net_2_deeper_net


class TestNet2DeeperNet(TestCase):
    SMALL_NOISE_EPSILON = 1e-6
    WEIGHTS = np.matrix([[1.0, 0.1, 0.5], [1.0, 0.1, 0.5]])
    BIAS = np.array([0.0, 0.0, 0.0])
    WEIGHTS_NEXT_LAYER = np.matrix([[1.0], [0.2], [0.5]])

    def test_activation_should_be_unchanged_after_adding_layer(self):
        inputs = np.array([0.1, 0.9])

        activation_pre_deepening = ((inputs * self.WEIGHTS) + self.BIAS)

        weight_new_layer, bias_new_layer = net_2_deeper_net(self.BIAS, noise_std=0.0001)

        activation_post_deepening = (activation_pre_deepening * weight_new_layer) + bias_new_layer

        np.testing.assert_array_almost_equal(activation_pre_deepening, activation_post_deepening, decimal=2,
                                             err_msg='Activation should be unchanged after adding a new layer')
