"""
MNIST classification with
* Model: Linear Logistic (Softmax) Regression Model
* Loss / Error Function: Cross-entropy
* Training algorithm: stochastic gradient decent with mini-batches
* Performance metric: classification accuracy


"""

import os

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


# this is adopted from
# https://www.tensorflow.org/get_started/mnist/beginners

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
mnist = input_data.read_data_sets(os.path.abspath(os.path.join(working_dir, data_dir)), one_hot=True)



# Logistic (Softmax) Regression Model
# =======================================================================================
# Linear Model with softmax loss function:
#  * evidence that image is of class i (for individual row vector x):  e_i = x_k W_ki + b_i
#  * probability for class membership i: softmax(e_i) = exp(e_i) / sum_j exp(e_j)

# Reference to input data:
# batch with an arbitrary number of MNIST images, each flattened into a 784-dimensional vector.
# We represent this as a 2-D tensor of floating-point numbers, with a shape [None, 784].
# None means that a dimension can be of any length.
x = tf.placeholder(tf.float32, [None, 784])

# model parameters
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# definition of model outputs
evidence = tf.matmul(x, W) + b
prediction = tf.nn.softmax(evidence)


# Cross-entropy loss function
# =======================================================================================
# * cross entropy one image: H(y|t) = - sum_j t_j log(y_j)
#   for
#   o target t vector (one-hot encoding)
#   o prediction y vector(membership probabilities)
#

# reference for target data
target = tf.placeholder(tf.float32, [None, 10])

# loss function
#  * tf.reduce_sum computes H(y|t) for one each image individually, i.e.
#    it adds the elements in the second dimension of y,
#    due to the reduction_indices=[1] parameter
#  * tf.reduce_mean computes the MEAN of the cross-entropy loss over the
#    given data set
cross_entropy = tf.reduce_mean(-tf.reduce_sum(target * tf.log(prediction), reduction_indices=[1]))
# note that this can be NUMERICALLY UNSTABLE

# So here we use tf.nn.softmax_cross_entropy_with_logits on the evidence and then average across the batch.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=evidence))


# Training: stochastic gradient decent with mini-batches
# =======================================================================================

# optimize by gradient descent with learning rate 0.5
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# RUN TRAINING: through InteractiveSession
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train 1000 batches
number_batches = 1000
batch_sise = 100
for _ in range(number_batches):
    batch_xs, batch_ys = mnist.train.next_batch(batch_sise)
    sess.run(optimizer, feed_dict={x: batch_xs, target: batch_ys})


# Evaluate Model with performance metric: classification accuracy
# =======================================================================================

# classification accuracy: percentage of correctly classified samples
correct_prediction = tf.equal(tf.argmax(evidence,1), tf.argmax(target,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# execute
print(sess.run(accuracy, feed_dict={x: mnist.test.images, target: mnist.test.labels}))

