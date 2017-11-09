""" Multilayer Perceptron.

A Multilayer Perceptron (Neural Network) implementation example using
TensorFlow library. This example is using the MNIST database of handwritten
digits (http://yann.lecun.com/exdb/mnist/).

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/

------------------------------------------------------------------------------
In compliance with License conditions, the code in this file was inspired
by Aymeric Damien's multilayer_perceptron.py:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/multilayer_perceptron.py

Author: Alexander Hentschel
        alex.hentschel@axiomzen.co

see also:
https://gist.github.com/mick001/45a45b94eab29d81a5f1e46d88632053
http://www.jessicayung.com/explaining-tensorflow-code-for-a-multilayer-perceptron/
"""


#
# ------------------------------------------------------------------------------

from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
import numpy as np
import pickle

from neural_network_model.mlp import mnist_mlp

tf.set_random_seed(1234)




# IMPORT DATA
# =======================================================================================
# * input data: mnist.train.images
#   Tensor (an n-dimensional array) with a shape of [55000, 784].
#   The first dimension is an index into the list of images and the second dimension is
#   the index for each pixel in each image. Each entry in the tensor is a pixel intensity
#   between 0 and 1, for a particular pixel in a particular image.
# output data: mnist.train.labels
# * Lables indicating number (0 ... 9) in one-hot encoding


def import_data(working_dir, data_dir):
    abs_data_dir = os.path.abspath(os.path.join(working_dir, data_dir))
    print("Storing temporary MNIST data in '%s'" % abs_data_dir)
    if not os.path.exists(abs_data_dir):
        os.makedirs(abs_data_dir)
    mnist = input_data.read_data_sets(abs_data_dir, one_hot=True)
    return mnist

def construct_nn():
    # Parameters
    learning_rate = 0.001
    training_epochs = 5
    batch_size = 100
    display_step = 1

    # Network Parameters
    n_input = 784 # MNIST data input (img shape: 28*28)
    n_hidden_1 = 256 # 1st layer number of neurons
    n_hidden_2 = 256 # 2nd layer number of neurons
    n_classes = 10 # MNIST total classes (0-9 digits)

    # tf Graph input
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_classes])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Create model
    def multilayer_perceptron(x):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.tanh(layer_1)
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.tanh(layer_2)
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer

    # Construct model
    logits = multilayer_perceptron(X)

    # Define loss and optimizer
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    return [optimizer, cross_entropy]


# =======================================================================================

# RUN TRAINING: through InteractiveSession
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# -----
init_params = np.load(os.path.join(os.path.join(working_dir, "params"), "params_0000.npz" ))
var_mapping = {weights['h1']: 'layer1_weights', biases['b1']:'layer1_bias',
               weights['h2']: 'layer2_weights', biases['b2']:'layer2_bias',
               weights['out']: 'output_layer_weights', biases['out']:'output_layer_bias'
              }

sess.run( [tf.assign(v, init_params[ref]) for v,ref in var_mapping.items()] )


for iteration in np.arange(5):
    data = np.load(os.path.join(os.path.join(working_dir, "batches"), "data_batch_%04d.npz" % iteration))
    _, l = sess.run([optimizer, cross_entropy], feed_dict={X: data['batch_x'], Y: data['batch_y']})
    print("loss %.09f" % l)

# =======================================================================================

working_dir = "/Users/alex/Temp/MNIST"
mnist = import_data(working_dir, "MNIST_data")

batch_x, batch_y = mnist.train.next_batch(batch_size)

T = mnist_mlp()
params = T.generate_initial_network_parameters(seed=1234)

T.start()
ipc = T.to_clearnames(params)

learning_rate = 0.5

for iteration in np.arange(5):
    data = np.load(os.path.join(os.path.join(working_dir, "batches"), "data_batch_%04d.npz" % iteration))
    l, grad = T.evaluate(data['batch_x'], data['batch_y'], params)
    for k in params.keys():
        params[k] -= learning_rate * grad[k]
    print("loss %.09f" % l)
    target_file = os.path.abspath(os.path.join(working_dir, param_dir, "params_%04d.npz" % (iteration+1)))
    save_dataset(T.to_clearnames(params), target_file)

T.close()
