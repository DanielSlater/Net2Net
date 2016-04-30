import numpy as np
from unittest import TestCase

from net2net.net_2_wider_net import net_2_wider_net


class TestNet2WiderNet(TestCase):
    SMALL_NOISE_EPSILON = 1e-6
    WEIGHTS = np.matrix([[1.0, 0.1, 0.5], [1.0, 0.1, 0.5]])
    BIAS = np.array([0.0, 0.0, 0.0])
    WEIGHTS_NEXT_LAYER = np.matrix([[1.0], [0.2], [0.5]])

    def test_one_wider_random_split(self):
        inputs = np.array([0.1, 0.9])

        activation_pre_widening = ((inputs * self.WEIGHTS) + self.BIAS) * self.WEIGHTS_NEXT_LAYER

        weights_post, bias_post, weights_next_layer_post = net_2_wider_net(self.WEIGHTS, self.BIAS,
                                                                           self.WEIGHTS_NEXT_LAYER,
                                                                           noise_std=self.SMALL_NOISE_EPSILON,
                                                                           split_max_weight_else_random=False)

        activation_post_net_widening = ((inputs * weights_post) + bias_post) * weights_next_layer_post

        np.testing.assert_array_almost_equal(activation_pre_widening, activation_post_net_widening, decimal=2,
                                             err_msg='activation should not be significantly changed after widening')

        self.assertEqual(self.BIAS.shape[0] + 1, bias_post.shape[0], msg='bias should be one larger')

    def test_x_wider(self):
        inputs = np.array([0.1, 0.9])
        new_layer_size = 8

        activation_pre_widening = ((inputs * self.WEIGHTS) + self.BIAS) * self.WEIGHTS_NEXT_LAYER

        weights_post, bias_post, weights_next_layer_post = net_2_wider_net(self.WEIGHTS, self.BIAS,
                                                                           self.WEIGHTS_NEXT_LAYER,
                                                                           noise_std=self.SMALL_NOISE_EPSILON,
                                                                           split_max_weight_else_random=False,
                                                                           new_layer_size=new_layer_size)

        activation_post_net_widening = ((inputs * weights_post) + bias_post) * weights_next_layer_post

        np.testing.assert_array_almost_equal(activation_pre_widening, activation_post_net_widening, decimal=2,
                                             err_msg='activation should not be significantly changed after widening')

        self.assertEqual(new_layer_size, bias_post.shape[0], msg='bias should same size as new_layer_size')

    def test_one_wider_max_split(self):
        inputs = np.array([0.1, 0.9])

        activation_pre_widening = ((inputs * self.WEIGHTS) + self.BIAS) * self.WEIGHTS_NEXT_LAYER

        weights_post, bias_post, weights_next_layer_post = net_2_wider_net(self.WEIGHTS, self.BIAS,
                                                                           self.WEIGHTS_NEXT_LAYER,
                                                                           noise_std=self.SMALL_NOISE_EPSILON,
                                                                           split_max_weight_else_random=True)

        activation_post_net_widening = ((inputs * weights_post) + bias_post) * weights_next_layer_post

        np.testing.assert_array_almost_equal(activation_pre_widening, activation_post_net_widening, decimal=2,
                                             err_msg='activation should not be significantly changed after widening')

        self.assertEqual(self.BIAS.shape[0] + 1, bias_post.shape[0], msg='bias should be one larger')
        self.assertAlmostEqual(self.WEIGHTS_NEXT_LAYER[0, 0] / 2., weights_next_layer_post[0, 0],
                               msg='this weight was the max so should have been split in 2')

    def test_no_noise(self):
        weights_post, bias_post, weights_next_layer_post = net_2_wider_net(self.WEIGHTS, self.BIAS,
                                                                           self.WEIGHTS_NEXT_LAYER,
                                                                           noise_std=None,
                                                                           split_max_weight_else_random=True)

        self.assertEqual(self.WEIGHTS_NEXT_LAYER[1, 0], weights_next_layer_post[1, 0],
                         msg='this weight was not the max so should not have been split so should be exactly equal')
