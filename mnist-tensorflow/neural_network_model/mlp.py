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
import sys


class mnist_mlp(object):
    """
    TensorFlow based implementation for 3-Layer neural network for solving MNIST.
    Please do NOT use MULTIPLE INSTANCES within the same python kernel.

    """

    def __init__(self):
        self.logger = logging.getLogger("mnist-mlp")
        #
        # Create TensorFlow compute graph with network parameters
        self._n_input = 784 # MNIST data input (img shape: 28*28)
        self._n_hidden_1 = 256 # 1st layer number of neurons
        self._n_hidden_2 = 256 # 2nd layer number of neurons
        self._n_classes = 10 # MNIST total classes (0-9 digits)
        #
        # tf Graph input and output
        self._input_data = tf.placeholder("float", [None, self._n_input])
        self._output_labels = tf.placeholder("float", [None, self._n_classes])
        #
        # Store layers' weights & biases
        self._params = {
            'layer1 weights': tf.placeholder("float", [self._n_input, self._n_hidden_1]),
            'layer2 weights': tf.placeholder("float", [self._n_hidden_1, self._n_hidden_2]),
            'output layer weights': tf.placeholder("float", [self._n_hidden_2, self._n_classes]),
            'layer1 bias': tf.placeholder("float", [self._n_hidden_1]),
            'layer2 bias': tf.placeholder("float", [self._n_hidden_2]),
            'output layer bias': tf.placeholder("float", [self._n_classes])
        }
        #
        # define neural network
        layer_1 = tf.add(tf.matmul(self._input_data, self._params['layer1 weights']), self._params['layer1 bias'])
        layer_1 = tf.nn.tanh(layer_1)
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, self._params['layer2 weights']), self._params['layer2 bias'])
        layer_2 = tf.nn.tanh(layer_2)
        # Output fully connected layer with a neuron for each class
        logits_output = tf.add(tf.matmul(layer_2, self._params['output layer weights']), self._params['output layer bias'])
        #
        # Define loss function:
        # currently cross-entropy loss, we should probably replace that with the
        # smooth-hinge loss function to be consistent.
        self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_output, labels=self._output_labels))
        #
        self._param_keys = list(self._params.keys())
        self._param_variables = [self._params[k] for k in self._param_keys]
        self._param_gradients = tf.gradients(self._loss, self._param_variables)
        #
        # self._gradients = {k : tf.gradients(self._loss, [var])[0] for k, var in self._params.items() }
        #
        self._session = None

    def start(self):
        if self._session is not None:
            raise ValueError("TensorFlow session already started")
        self._session = tf.InteractiveSession()
        # tf.global_variables_initializer().run()
        uninit_vars = self._session.run(tf.report_uninitialized_variables())
        if len(uninit_vars) > 0: # Note: we shouldn't need to initialize as values for all variables are fed in externally
            raise ValueError("Uninitialized variables in TensorFlow session")
        print(uninit_vars)

    def close(self):
        if self._session is not None:
            self._session.close()
            self._session = None

    def _ensure_session_stated(self):
        if self._session is None:
            raise ValueError("Please start TensorFlow session by calling function 'start()'")

    def compute_gradient(self, x, y, ws):
        self._ensure_session_stated()
        #
        feed_dict = dict(ws)                  # copy dictionary to prevent modification of function input
        feed_dict[self._input_data] = x       # add user-specified value for input data batch
        feed_dict[self._output_labels] = y    # add user-specified value for target labels
        loss, gradients_list = self._session.run([self._loss, self._param_gradients], feed_dict=feed_dict)
        return loss, dict(zip(self._param_variables, gradients_list))

    def generate_initial_network_parameters(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return { v: np.random.normal(size=v.shape.as_list()) for v in self._params.values() }

    def to_clearnames(self, parameter_dictionary):
        var_to_clearname_dic = { v:k for k,v in self._params.items() }
        var_to_clearname_dic[self._input_data] = 'input data batch'
        var_to_clearname_dic[self._output_labels] = 'target label batch'
        return { var_to_clearname_dic[k] : v for k, v in parameter_dictionary.items() }

    def to_intervars(self, parameter_dictionary):
        return {self._params[k] :parameter_dictionary[k]  for k in self._param_keys}


# ============================= Main for testing ==============================


def save_dataset(values, target_file):
    values = {key.replace(' ', '_') : var for key, var in values.items()}
    target_dir = os.path.abspath(os.path.join(target_file, os.pardir))
    if not os.path.exists(target_dir):
        print("creating folder")
        os.makedirs(target_dir)
    target_file = os.path.abspath(target_file)
    print("writing file '%s'" % target_file)
    np.savez_compressed(target_file, **values)

if __name__ == "__main__":
    llevel = logging.DEBUG
    logging.basicConfig(stream=sys.stdout, level=llevel)
    logger = logging.getLogger("protoype")
    logger.setLevel(llevel)


llevel = logging.DEBUG
logging.basicConfig(stream=sys.stdout, level=llevel)
logger = logging.getLogger("protoype")
logger.setLevel(llevel)

T = mnist_mlp()
params = T.generate_initial_network_parameters(seed=1234)
ipc = T.to_clearnames(params)

# save starting parameters
working_dir = "/Users/alex/Temp/MNIST"
param_dir = "params"
iteration = 0
file_name = "params_%04d.npz" % iteration

target_file = os.path.abspath(os.path.join(working_dir, param_dir, "params_%04d.npz" % iteration))
save_dataset(ipc, target_file)


learning_rate = 0.5
T.start()
for iteration in np.arange(5):
    data = np.load(os.path.join(os.path.join(working_dir, "batches"), "data_batch_%04d.npz" % iteration))
    l, grad = T.compute_gradient(data['batch_x'], data['batch_y'], params)
    for k in params.keys():
        params[k] -= learning_rate * grad[k]
    print("loss %.09f" % l)
    target_file = os.path.abspath(os.path.join(working_dir, param_dir, "params_%04d.npz" % (iteration+1)))
    save_dataset(T.to_clearnames(params), target_file)

T.close()

