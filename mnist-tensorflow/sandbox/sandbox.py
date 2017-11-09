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


# tf.set_random_seed(1234)

import pickle


# IMPORT DATA
# =======================================================================================
# * input data: mnist.train.images
#   Tensor (an n-dimensional array) with a shape of [55000, 784].
#   The first dimension is an index into the list of images and the second dimension is
#   the index for each pixel in each image. Each entry in the tensor is a pixel intensity
#   between 0 and 1, for a particular pixel in a particular image.
# output data: mnist.train.labels
# * Lables indicating number (0 ... 9) in one-hot encoding

working_dir = "/Users/alex/Temp/MNIST"
data_dir = "MNIST_data"

abs_data_dir = os.path.abspath(os.path.join(working_dir, data_dir))
print("Storing temporary MNIST data in '%s'" % abs_data_dir)
if not os.path.exists(abs_data_dir):
    os.makedirs(abs_data_dir)
mnist = input_data.read_data_sets(abs_data_dir, one_hot=True)


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

# http://deeplearning.net/tutorial/mlp.html
# rng.uniform(
#     low=-numpy.sqrt(6. / (n_in + n_out)),
#     high=numpy.sqrt(6. / (n_in + n_out)),
#     size=(n_in, n_out)
# ),

# MISSING: non-linear activation function
#  * tf.nn.tanh()
#  * tf.nn.relu(...)
#  * alternatively, use tf.layers.dense()

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
    out_layer = tf.nn.tanh(out_layer)
    return out_layer

# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


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




