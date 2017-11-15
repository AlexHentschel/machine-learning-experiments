"""
Logistic regression model for solving MNIST.
The model is a Multi-Layer Perceptron (MLP) implemented in TensorFlow.

Model specifics:
- input layer for 784 features, representing the 28*28 image pixels
- two hidden layers contain 256 neurons each
  with tanh activation
- output layer is the softmax function

Training objective:
- minimize cross entropy loss

Class only computes:
- numerical loss value for given data set
- gradients of the loss function with respect to internal network parameters

Notes:
- Most likely, several instances of this class can be used within the same Python kernel
- However, concurrency of multiple instances has not been tested and such uisage is therefore discouraged.

Tested with
- Anaconda 5.0.1 x64, Python 3.6.2
- TensorFlow 1.4.0
- on MacOS High Sierra (10.13.1)


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

import tensorflow as tf
import numpy as np
import logging


class mnist_mlp(object):
    """
    TensorFlow based implementation for 3-Layer neural network for solving MNIST.
    Please do NOT use MULTIPLE INSTANCES within the same python kernel.

    """

    @staticmethod
    def slice_weightand_bias_tensors(fanin, fanout, param_vector, start):
        """
        Slices and re-shapes a weight matrix and a bias vector for forward-connecting
        the mlp layer fanin to fanout.
        The weight matrix is constructed from param_vector starting at position
        start. The weight matrix has dimension fanin x fanout. Furthermore, a bias vector
        of size fanout is constructed using the subsequent fanout number of parameters
        in param_vector.

        :param fanin: size of source network layer
        :param fanout: size of forward network layer
        :param param_vector: 1-D Tensor (vector) containing all network parameters
        :param start: index (zero-based) for first parameter that can be taken
        :return: triple
                 (weight matrix, bias vector, index of next parameter that can be taken for further layers)
                 of type: (Tensor, Tensor, int)
        """
        number_weights = fanin * fanout
        weights = tf.slice(param_vector, [start], [number_weights])
        weight_matrix = tf.reshape(weights, [fanin, fanout])
        bias = tf.slice(param_vector, [start + number_weights], [fanout])
        updated_start = start + number_weights + fanout
        return weight_matrix, bias, updated_start

    @staticmethod
    def construct_layer(input, number_neurons, param_vector, start, activation=None):
        """
        Constructs an MLP Layer with `number_neurons` hidden neurons using `input` as source.
        The parameters are taken from `param_vector` where `start` marks the index
        of the first parameter that has not been used for previous layers.

        :param input: Tensor representing the source layer
        :param number_neurons:  number of hidden neurons in layer to be constructed
        :param param_vector: 1-D Tensor (vector) containing all network parameters
        :param start: index (zero-based) for first parameter that can be taken
        :param activation: [optional] activation function
        :return: pair
                 (Tensor representing the newly constructed layer, index of next parameter that can be taken for further layers)
                 of type (Tensor, int)
        """
        fanin_size = int(input.get_shape()[-1])
        weight_matrix, bias, start = mnist_mlp.slice_weightand_bias_tensors(fanin_size, number_neurons, param_vector, start)
        layer = tf.add(tf.matmul(input, weight_matrix), bias)
        if activation is not None:
            layer = activation(layer)
        return layer, start


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
        #
        # ordering convention for serialized network parameters:
        # Layer 1:       - weights: n_input x n_hidden_1
        #                - bias n_hidden_1
        #                total: (n_input + 1) * n_hidden_1
        # Layer 2:       - weights: n_hidden_1 x n_hidden_2
        #                - bias n_hidden_2
        #                total: (n_hidden_1 + 1) * n_hidden_2
        # Output Layer:  - weights: n_hidden_2 x n_classes
        #                - bias n_classes
        #                total: (n_hidden_2 + 1) * n_classes
        # Grand total:  (n_input + 1) * n_hidden_1 + (n_hidden_1 + 1) * n_hidden_2 + (n_hidden_2 + 1) * n_classes
        self._total_parameter_number = (self._n_input + 1) * self._n_hidden_1 \
                                     + (self._n_hidden_1 + 1) * self._n_hidden_2 \
                                     + (self._n_hidden_2 + 1) * self._n_classes
        self._params = tf.placeholder("float", self._total_parameter_number)
        start = 0
        layer_1, start = mnist_mlp.construct_layer(self._input_data, self._n_hidden_1, self._params, start, activation=tf.nn.tanh)
        layer_2, start = mnist_mlp.construct_layer(layer_1, self._n_hidden_2, self._params, start, activation=tf.nn.tanh)
        logits_output, start = mnist_mlp.construct_layer(layer_2, self._n_classes, self._params, start)
        #
        # Define loss function:
        # currently cross-entropy loss, we should probably replace that with the
        # smooth-hinge loss function to be consistent.
        self._output_labels = tf.placeholder("float", [None, self._n_classes])  # Placeholder for loading target values into
        self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_output, labels=self._output_labels))
        self._gradients = tf.gradients(self._loss, self._params)
        #
        # Formula for computing classification accuracy
        class_probabilities = tf.nn.softmax(logits_output)
        correct_prediction = tf.equal(tf.argmax(class_probabilities, 1), tf.argmax(self._output_labels, 1))
        self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        #
        self._session = None


    def start(self):
        """
        Starts up the internal TensorFlow session,
        which required to compute anything. Please close the
        session at the end by calling mlp.close()
        """
        if self._session is not None:
            raise ValueError("TensorFlow session already started")
        self._session = tf.InteractiveSession()
        # tf.global_variables_initializer().run()
        uninit_vars = self._session.run(tf.report_uninitialized_variables())
        if len(uninit_vars) > 0: # Note: we shouldn't need to initialize as values for all variables are fed in externally
            raise ValueError("Uninitialized variables in TensorFlow session")
        self.logger.info("Successfully started TensorFlow session")

    def close(self):
        """
        Terminates internal TensorFlow session.
        Performs no operation if no TensorFlow session is currently running.
        """
        if self._session is not None:
            self._session.close()
            self._session = None
            self.logger.info("TensorFlow session terminated")
        else:
            self.logger.debug("No TensorFlow session running")

    def _ensure_session_stated(self):
        if self._session is None:
            raise ValueError("Please start TensorFlow session by calling function 'start()'")

    def eval(self, x, y, w):
        """
        Computes for the given data set (x,y):
        - the model's cross-entropy loss

        :param x: Test data set.
                  Expected input: numpy matrix of dimension N x 784, where N represents the number of
                  images in the test data set. Each row is expected to contain 784 features
                  representing the 28*28 image pixels
        :param y: Target labels for rows of x.
                  Expected input: numpy matrix of dimension N x 10, where N represents the number of
                  images in the test data set. Each row represents the image's class as one-hot encoded vector.
        :param w: Model parameters, i.e. neural network's weight matrices and bias vectors.
                  (all represented in one gigantic vector)
        :return: the model's cross-entropy loss (scalar)
        """
        self._ensure_session_stated()
        loss = self._session.run(self._loss, feed_dict={self._input_data:x, self._output_labels:y, self._params:w})
        return loss

    def grad(self, x, y, w):
        """
        Computes for the given data set (x,y):
        - the gradients of the loss function with respect to the network's internal parameters

        :param x: Test data set.
                  Expected input: numpy matrix of dimension N x 784, where N represents the number of
                  images in the test data set. Each row is expected to contain 784 features
                  representing the 28*28 image pixels
        :param y: Target labels for rows of x.
                  Expected input: numpy matrix of dimension N x 10, where N represents the number of
                  images in the test data set. Each row represents the image's class as one-hot encoded vector.
        :param w: Model parameters, i.e. neural network's weight matrices and bias vectors.
                  (all represented in one gigantic vector)
        :return: gradients for network parameters (1-dimensional numpy array).
        """
        self._ensure_session_stated()
        grads = self._session.run(self._gradients, feed_dict={self._input_data:x, self._output_labels:y, self._params:w})
        return grads

    def compute_accuracy(self, x, y, w):
        """
        Computes the model's classification accuracy (percentage of correctly classified images).

        :param x: Test data set.
                  Expected input: numpy matrix of dimension N x 784, where N represents the number of
                  images in the test data set. Each row is expected to contain 784 features
                  representing the 28*28 image pixels
        :param y: Target labels for rows of x.
                  Expected input: numpy matrix of dimension N x 10, where N represents the number of
                  images in the test data set. Each row represents the image's class as one-hot encoded vector.
        :param w: Model parameters, i.e. neural network's weight matrices and bias vectors.
                  (all represented in one gigantic vector)
        :return: classification accuracy in range [0,1]
        """
        self._ensure_session_stated()
        accuracy = self._session.run(self._accuracy, feed_dict={self._input_data:x, self._output_labels:y, self._params:w})
        return accuracy

    def generate_initial_network_parameters(self, seed=None):
        """
        Generates random starting parameters for network's internal weight matrices and bias vectors.

        :param seed: Optional seeding parameter for numpy random
        :return: A dictionary containing values for each of the network's internal parameters.
                 Specifically, the dictionary's keys are TensorFlow variables
                 and the corresponding values are numpy float matrices.
        """
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(size=self._total_parameter_number)

