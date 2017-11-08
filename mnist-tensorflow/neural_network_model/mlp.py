"""
Multi-Layer Perceptron (MLP).
A Neural Network, specifically MLP, implementation based on TensorFlow.

------------------------------------------------------------------------------
In compliance with License conditions, the code in this file was inspired
by Aymeric Damien's multilayer_perceptron.py:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/multilayer_perceptron.py

Copyright (c)
* 2015, Aymeric Damien.
* 2017, Alexander Hentschel, alex.hentschel@axiomzen.co

For further explanation, see:
https://gist.github.com/mick001/45a45b94eab29d81a5f1e46d88632053
http://www.jessicayung.com/explaining-tensorflow-code-for-a-multilayer-perceptron/
------------------------------------------------------------------------------
"""

from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
import numpy as np
import logging


class mnist_mlp(object):
    """
    TensorFlow based implementation for 3-Layer neural network for solving MNIST.
    Please do NOT use MULTIPLE INSTANCES within the same python kernel.

    """

    def __init__(self, seed=None):
        self._seed = seed
        self.logger = logging.getLogger("mnist-mlp")
        #
        # Create TensorFlow compute graph with network parameters
        n_input = 784 # MNIST data input (img shape: 28*28)
        n_hidden_1 = 256 # 1st layer number of neurons
        n_hidden_2 = 256 # 2nd layer number of neurons
        n_classes = 10 # MNIST total classes (0-9 digits)
        #
        # tf Graph input and output
        self._X = tf.placeholder("float", [None, n_input])
        self._Y = tf.placeholder("float", [None, n_classes])
        #
        # Store layers' weights & biases
        self._params = {
            'layer1_weights': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'layer2_weights': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'output_weights': tf.Variable(tf.random_normal([n_hidden_2, n_classes])),
            'layer1_bias': tf.Variable(tf.random_normal([n_hidden_1])),
            'layer2_bias': tf.Variable(tf.random_normal([n_hidden_2])),
            'output_bias': tf.Variable(tf.random_normal([n_classes]))
        }
        #
        # define neural network
        layer_1 = tf.add(tf.matmul(self._X, self._params['layer1_weights']), self._params['layer1_bias'])
        layer_1 = tf.nn.tanh(layer_1)
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, self._params['layer2_weights']), self._params['layer2_bias'])
        layer_2 = tf.nn.tanh(layer_2)
        # Output fully connected layer with a neuron for each class
        logits_output = tf.add(tf.matmul(layer_2, self._params['output_weights']), self._params['output_bias'])
        #
        # Define loss function:
        # currently cross-entropy loss, we should probably replace that with the
        # smooth-hinge loss function to be consistent.
        self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_output, labels=self._Y))
        #
        self._gradients = {key : tf.gradients(self._loss, [var])[0] for key, var in self._params.items() }
        #
        self._session_started = False


    def start(self):
        if not self._session_started:
            if self._seed:
                tf.set_random_seed(self._seed)
            self._session = tf.InteractiveSession()
            tf.global_variables_initializer().run()
            self._session_started = True


    def close(self):
        if self._session_started:
            self._session.close()
            self._session_started = False

    def get_network_parameters(self):
        if not self._session_started:
            raise ValueError("Please start TensorFlow session by calling function 'start'")
        return {key : self._session.run(var) for key, var in self._params.items() }

    def compute_gradient(self, x, y):
        if not self._session_started:
            raise ValueError("Please start TensorFlow session by calling function 'start'")
        loss, c = sess.run([self._loss, loss_op], feed_dict={X: batch_x,
                                                        Y: batch_y})

        return {key : self._session.run(var) for key, var in self._gradients.items() }

    def


# support methods
# =======================================================================================

def save_dataset(tf_sess, tf_vars, working_dir, target_file):
    values = {key: tf_sess.run(var) for key, var in tf_vars.items()}
    target = os.path.abspath(os.path.join(working_dir, target_file))
    target_dir = os.path.abspath(os.path.join(target, os.pardir))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    np.savez_compressed(target, **values)

def save_nn_params(tf_sess, id, weight_vars, bias_vars, working_dir):
    save_dataset(tf_sess, weight_vars, working_dir, "weights_" + str(id) + ".npz")
    save_dataset(tf_sess, bias_vars, working_dir, "biases_" + str(id) + ".npz")


# Self-contained session
# =======================================================================================
temp_dir = "temp2"

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
        #
        target_dir = os.path.join(working_dir, temp_dir)
        save_nn_params(sess, '%04d' % epoch, weights, biases, target_dir)
        np.savez_compressed(os.path.join(target_dir, "data_batch_%04d.npz" % epoch), batch_x=batch_x, batch_y=batch_y)

    print("Optimization Finished!")

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))









# Self-contained session
# =======================================================================================

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))




# Interactive session
# =======================================================================================

tf.set_random_seed(1234)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

temp_dir = "temp1"

id = '%03d' % 0
save_nn_params(sess, "x", weights, biases, os.path.join(working_dir, temp_dir))

loaded = np.load(os.path.join(os.path.join(working_dir, "temp1"), "weights_x.npz"))
loaded.keys()

ref_values = np.load(os.path.join(os.path.join(working_dir, "temp"), "weights_000.npz"))

for key in ref_values.keys():
    abs_error = np.sum(np.absolute(loaded[key] - ref_values[key]))
    print("Abs Error for matrix '%s': %f" % (key,abs_error))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sess.close()

# get weights:
# https://stackoverflow.com/questions/33679382/tensorflow-get-current-value-of-a-variable
#

# Tensorflow get gradient
# https://stackoverflow.com/questions/35226428/how-do-i-get-the-gradient-of-the-loss-at-a-tensorflow-variable

# assign weights
# https://github.com/tensorflow/tensorflow/issues/2854



loaded = np.load(os.path.join(os.path.join(working_dir, "temp"), "weights_1.npz"))
loaded.keys()

ref_values = np.load(os.path.join(os.path.join(working_dir, "temp"), "biases_3.npz"))

for key in ref_values.keys():
    abs_error = np.sum(np.absolute(loaded[key] - ref_values[key]))
    print("Abs Error for matrix '%s': %f" % (key,abs_error))

# =======================================================================================

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

temp_dir = "temp"
for i in np.arange(0,5):
    weight_values = np.load(os.path.join(os.path.join(working_dir, temp_dir), "weights_" + str(i) + ".npz"))
    biases_values = np.load(os.path.join(os.path.join(working_dir, temp_dir), "biases_" + str(i) + ".npz"))
    data = np.load(os.path.join(os.path.join(working_dir, temp_dir), "data_batch_%04d.npz" % i))
    for k,v in weight_values.items():
        sess.run(tf.assign(weights[k], v))
    for k,v in biases_values.items():
        sess.run(tf.assign(biases[k], v))
    _, c = sess.run([train_op, loss_op], feed_dict={X: data["batch_x"], Y: data["batch_y"]})
    print("Cost for batch %d: %.9f" %(i,c))




sess.close()

k = 'h1'
loaded = np.load(os.path.join(os.path.join(working_dir, "temp"), "weights_1.npz"))

# difference between randomly initialized value and loaded weights
t = sess.run(weights[k]).copy()
er = np.sum(np.absolute(loaded[k] - t))
print("Tensor difference between current weight '%s' and loaded matrix: %.9f" % (k, er))

# override with loaded weights: assign loaded weights
assign_op = tf.assign(weights[k], loaded[k])
sess.run(assign_op)
t = sess.run(weights[k]).copy()
er = np.sum(np.absolute(loaded[k] - t))
print("Tensor difference between current weight '%s' and loaded matrix: %.9f" % (k, er))




llevel = logging.DEBUG
logging.basicConfig(stream=sys.stdout, level=llevel)
logger = logging.getLogger("protoype")
logger.setLevel(llevel)