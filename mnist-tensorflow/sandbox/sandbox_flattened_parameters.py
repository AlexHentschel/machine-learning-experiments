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
import tensorflow.examples.tutorials.mnist
import tensorflow as tf
import os
import numpy as np
import logging
import sys

# ------------------------------------------------------------------------------

def init_logger():
    llevel = logging.DEBUG
    logging.basicConfig(stream=sys.stdout, level=llevel)
    logger = logging.getLogger("mlp_usage_example")
    logger.setLevel(llevel)
    return logger

def import_data(working_dir, data_dir):
    abs_data_dir = os.path.abspath(os.path.join(working_dir, data_dir))
    logger.info("Directory for MNIST data: '%s'", abs_data_dir)
    if not os.path.exists(abs_data_dir):
        logger.info("creating directory: '%s'", abs_data_dir)
        os.makedirs(abs_data_dir)
    return tensorflow.examples.tutorials.mnist.input_data.read_data_sets(abs_data_dir, one_hot=True)



# ------------------------------------------------------------------------------

tf.set_random_seed(1234)
logger = init_logger()


# =========================================================================== #

working_dir = "/Users/alex/Temp/MNIST"
param_dir = "params"
mnist = import_data(working_dir, "MNIST_data")
learning_rate = 0.5

def flatten_nn_params(nn, params):
    P = nn.to_clearnames(params)
    p = np.append(P['layer1 weights'].flatten(), P['layer1 bias'])
    p = np.append(p, P['layer2 weights'].flatten())
    p = np.append(p, P['layer2 bias'])
    p = np.append(p, P['output layer weights'].flatten())
    p = np.append(p, P['output layer bias'])
    return p


# =======================================================================================
from neural_network_model.mlp import mnist_mlp as m_ref

working_dir = "/Users/alex/Temp/MNIST"

R = m_ref()
p = R.generate_initial_network_parameters(seed=1234)

R.start()

for iteration in np.arange(5):
    data = np.load(os.path.join(os.path.join(working_dir, "batches"), "data_batch_%04d.npz" % iteration))
    l, grad = R.evaluate(data['batch_x'], data['batch_y'], p)
    for k in p.keys():
        p[k] -= learning_rate * grad[k]
    print("loss %.09f" % l)

R.close()

# ....................................................................
from neural_network_model.mlp_flattend_parameters import mnist_mlp as m_test

T = m_test()
p = flatten_nn_params(R, R.generate_initial_network_parameters(seed=1234))
T.start()

for iteration in np.arange(5):
    data = np.load(os.path.join(os.path.join(working_dir, "batches"), "data_batch_%04d.npz" % iteration))
    l = T.eval(data['batch_x'], data['batch_y'], p)
    grad = T.grad(data['batch_x'], data['batch_y'], p)
    p -= learning_rate * grad[0]
    print("loss %.09f" % l)

T.close()



