"""
A Multilayer Perceptron implementation example using TensorFlow library.
A first network with 100, 100 hidden nodes is trained
Then a series of new networks of different sizes are cloned from it to find the best size

This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)

This is an extension of: https://github.com/aymericdamien/TensorFlow-Examples/
By: Aymeric Damien
"""
from copy import copy

import input_data
from net2net.net_2_wider_net import net_2_wider_net
import tensorflow as tf

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.005
minimal_model_training_epochs = 1
after_resize_training_epochs = 1
batch_size = 100

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)
minimal_n_hidden_1 = 100  # 1st layer num features
minimal_n_hidden_2 = 100  # 2nd layer num features

max_nodes_per_layer = 301
node_per_layer_step = 50

hidden_node_grid_search = [(x, y) for x in range(minimal_n_hidden_1, max_nodes_per_layer, node_per_layer_step) for y in
                           range(minimal_n_hidden_2, max_nodes_per_layer, node_per_layer_step) if x >= y]

print("We will be testing the following numbers of hidden nodes in layer 1 and 2:")
print(hidden_node_grid_search)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(input_placeholder, _weights, _biases):
    layer_1 = tf.nn.relu(
        tf.add(tf.matmul(input_placeholder, _weights[0]), _biases[0]))  # Hidden layer with RELU activation
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights[1]), _biases[1]))  # Hidden layer with RELU activation
    prediction_op = tf.matmul(layer_2, _weights[2]) + _biases[2]
    cost_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction_op, y))  # Softmax loss
    train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost_op)
    return cost_op, train_op, prediction_op


# method for training a network
def training_cycle(session, cost_fn, train_op, prediction, epochs):
    for epoch in range(epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            session.run(train_op, feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += session.run(cost_fn, feed_dict={x: batch_xs, y: batch_ys}) / total_batch

        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")
    # Test model
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    train_accuracy = accuracy.eval({x: mnist.train.images, y: mnist.train.labels})
    test_accuracy = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
    print("Train Accuracy: %s Test Accuracy: %s" % (train_accuracy, test_accuracy))

    return avg_cost, train_accuracy, test_accuracy


def clone_wider_network(minimal_network_weights,
                        minimal_network_biases,
                        new_n_hidden_nodes_1,
                        new_n_hidden_nodes_2):
    print("Creating network with hidden nodes %s, %s" % (new_n_hidden_nodes_1, new_n_hidden_nodes_2))
    new_weights = copy(minimal_network_weights)
    new_biases = copy(minimal_network_biases)
    # expand the layers that need expanding
    if new_biases[0].shape[0] < new_n_hidden_nodes_1:
        new_weights[0], new_biases[0], new_weights[1] = net_2_wider_net(new_weights[0], new_biases[0],
                                                                        new_weights[1],
                                                                        new_layer_size=new_n_hidden_nodes_1,
                                                                        noise_std=0.01)

    if new_biases[1].shape[0] < new_n_hidden_nodes_2:
        new_weights[1], new_biases[1], new_weights[2] = net_2_wider_net(new_weights[1], new_biases[1],
                                                                        new_weights[2],
                                                                        new_layer_size=new_n_hidden_nodes_2,
                                                                        noise_std=0.01)

    weights_variables = [
        tf.Variable(new_weights[0]),
        tf.Variable(new_weights[1]),
        tf.Variable(new_weights[2])]

    biases_variables = [
        tf.Variable(new_biases[0]),
        tf.Variable(new_biases[1]),
        tf.Variable(new_biases[2])]

    return weights_variables, biases_variables


# Store layers weight & bias
weight_variables = [tf.Variable(tf.random_normal([n_input, minimal_n_hidden_1])),
                    tf.Variable(tf.random_normal([minimal_n_hidden_1, minimal_n_hidden_2])),
                    tf.Variable(tf.random_normal([minimal_n_hidden_2, n_classes]))]

bias_variables = [tf.Variable(tf.random_normal([minimal_n_hidden_1])),
                  tf.Variable(tf.random_normal([minimal_n_hidden_2])),
                  tf.Variable(tf.random_normal([n_classes]))]

# Construct model
cost, optimizer, pred = multilayer_perceptron(x, weight_variables, bias_variables)

# Initializing the variables
init = tf.initialize_all_variables()

results = []

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    training_cycle(sess, cost, optimizer, pred, minimal_model_training_epochs)

    # get the values of the trained parameters
    minimal_network_weights = list(sess.run(weight_variables))
    minimal_network_biases = list(sess.run(bias_variables))

    for n_layer_h1, n_layer_h2 in hidden_node_grid_search:
        weight_variables, bias_variables = clone_wider_network(minimal_network_weights, minimal_network_biases,
                                                               n_layer_h1, n_layer_h2)

        # must initalize all these variables
        sess.run(tf.initialize_variables(weight_variables + bias_variables))

        new_cost, new_optimizer, new_pred = multilayer_perceptron(x, weight_variables, bias_variables)
        cost, train, test = training_cycle(sess, new_cost, new_optimizer, new_pred, after_resize_training_epochs)
        results.append((n_layer_h1, n_layer_h2, cost, train, test))

print(results)
print("best was " + str(max(results, key=lambda a: a[-1])))
