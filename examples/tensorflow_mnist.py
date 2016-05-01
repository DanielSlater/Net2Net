"""
A Multilayer Perceptron implementation example using TensorFlow library.
After training we then create a 2nd network which is a bigger version of the 1st network and train on that

This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)

This is an extension of: https://github.com/aymericdamien/TensorFlow-Examples/
By: Aymeric Damien
"""
import input_data
from net2net.net_2_wider_net import net_2_wider_net
import tensorflow as tf

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256  # 1st layer num features
n_hidden_2 = 40  # 2nd layer num features, for this example it is initialy over constrained
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)
n_hidden_2_nodes_after_resize = 200

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(_X, _weights, _biases):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights[0]), _biases[0]))  # Hidden layer with RELU activation
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights[1]), _biases[1]))  # Hidden layer with RELU activation
    return tf.matmul(layer_2, _weights[2]) + _biases[2]


# method for training a network
def training_cycle(session, cost_function, train_op, predication):
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            session.run(train_op, feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += session.run(cost_function, feed_dict={x: batch_xs, y: batch_ys}) / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost)
    print "Optimization Finished!"
    # Test model
    correct_prediction = tf.equal(tf.argmax(predication, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})


# Store layers weight & bias
weights = [
    tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    tf.Variable(tf.random_normal([n_hidden_2, n_classes]))]

biases = [
    tf.Variable(tf.random_normal([n_hidden_1])),
    tf.Variable(tf.random_normal([n_hidden_2])),
    tf.Variable(tf.random_normal([n_classes]))]

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    training_cycle(sess, cost, optimizer, pred)

    print("creating new network, increasing size of 2nd hidden layer from %s to %s" % (
        n_hidden_2, n_hidden_2_nodes_after_resize))

    # now we have trained the model lets copy the parameters and make one of the layer wider
    trained_weights_h1, trained_weights_h2, trained_weights_output = sess.run(
        [weights[0], weights[1], weights[2]])
    trained_bias_h1, trained_bias_h2, trained_bias_output = sess.run([biases[0], biases[1], biases[2]])

    # make the 2nd layer bigger
    new_weights_h2, new_bias_h2, new_weights_output = net_2_wider_net(trained_weights_h2, trained_bias_h2,
                                                                      trained_weights_output,
                                                                      new_layer_size=n_hidden_2_nodes_after_resize)

    # create new network with the changed weights, you can also simple reassign the existing variables to have these
    # new values, if validate is set to False it will all still work
    new_weights = [
        tf.Variable(trained_weights_h1),
        tf.Variable(new_weights_h2),
        tf.Variable(new_weights_output)]

    new_biases = [
        tf.Variable(trained_bias_h1),
        tf.Variable(new_bias_h2),
        tf.Variable(trained_bias_output)]

    # must initalize all these variables
    sess.run(tf.initialize_variables(new_weights + new_biases))

    new_pred = multilayer_perceptron(x, new_weights, new_biases)

    new_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(new_pred, y))
    # if we were to use an Adam optimizer we would have to do something clever about initializing it's variables
    new_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(new_cost)

    # loss will hopefully be better with more nodes in the 2nd layer
    training_cycle(sess, new_cost, new_optimizer, new_pred)
