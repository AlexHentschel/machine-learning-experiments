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

    def _construct_feed_dictionary(self, x, y, ws):
        feed_dict = dict(ws)                  # copy dictionary to prevent modification of function input
        feed_dict[self._input_data] = x       # add user-specified value for input data batch
        feed_dict[self._output_labels] = y    # add user-specified value for target labels
        return feed_dict

    def evaluate(self, x, y, ws):
        """
        Computes for the given data set (x,y):
        - the model's cross-entropy loss
        - the gradients of the loss function with respect to the network's internal parameters

        :param x: Test data set.
                  Expected input: numpy matrix of dimension N x 784, where N represents the number of
                  images in the test data set. Each row is expected to contain 784 features
                  representing the 28*28 image pixels
        :param y: Target labels for rows of x.
                  Expected input: numpy matrix of dimension N x 10, where N represents the number of
                  images in the test data set. Each row represents the image's class as one-hot encoded vector.
        :param ws: Model parameters, i.e. neural network's weight matrices and bias vectors.
                   A dictionary-like object is expected to have the same key-set and array-shapes as the
                   dictionary returned mlp.generate_initial_network_parameters(). Specifically, the dictionary's
                   keys are TensorFlow variables and the corresponding values are numpy float matrices.
        :return: Pair (loss, gradients);
                 loss:      the model's cross-entropy loss;
                 gradients: dictionary containing the gradients for each of the network's internal weight matrices
                            and bias vectors. The key-set and array-shapes are identical to the dictionary returned by
                            function mlp.generate_initial_network_parameters().
        """
        self._ensure_session_stated()
        feed_dict = self._construct_feed_dictionary(x, y, ws)
        loss, gradients_list = self._session.run([self._loss, self._param_gradients], feed_dict=feed_dict)
        return loss, dict(zip(self._param_variables, gradients_list))

    def compute_accuracy(self, x, y, ws):
        """
        Computes the model's classification accuracy (percentage of correctly classified images).

        :param x: Test data set.
                  Expected input: numpy matrix of dimension N x 784, where N represents the number of
                  images in the test data set. Each row is expected to contain 784 features
                  representing the 28*28 image pixels
        :param y: Target labels for rows of x.
                  Expected input: numpy matrix of dimension N x 10, where N represents the number of
                  images in the test data set. Each row represents the image's class as one-hot encoded vector.
        :param ws: Model parameters, i.e. neural network's weight matrices and bias vectors.
                   A dictionary-like object is expected to have the same key-set and array-shapes as the
                   dictionary returned mlp.generate_initial_network_parameters(). Specifically, the dictionary's
                   keys are TensorFlow variables and the corresponding values are numpy float matrices.
        :return: classification accuracy in range [0,1]
        """
        self._ensure_session_stated()
        feed_dict = self._construct_feed_dictionary(x, y, ws)
        accuracy = self._session.run(self._accuracy, feed_dict=feed_dict) # tf.Session.run() always returns a list with results
        return accuracy                                                # as we evaluated only one graph, the output is a list of length one

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
        return { v: np.random.normal(size=v.shape.as_list()) for v in self._params.values() }

    def to_clearnames(self, parameter_dictionary):
        """
        Non-essential function. Converts the key-set of the parameter_dictionary
        to human-readable strings. This is useful for saving the training results.

        :param parameter_dictionary:
               A dictionary containing numpy matrices for each of the network's internal TensorFlow placeholders.
               Specifically, the dictionary's keys are TensorFlow variables
               and the corresponding values are numpy float matrices.
        :return: A dictionary with the same values as input parameter_dictionary
                 but with keys replaced by human-readable strings.
        """
        var_to_clearname_dic = { v:k for k,v in self._params.items() }
        var_to_clearname_dic[self._input_data] = 'input data batch'
        var_to_clearname_dic[self._output_labels] = 'target label batch'
        return { var_to_clearname_dic[k] : v for k, v in parameter_dictionary.items() }

    def to_intervars(self, parameter_dictionary):
        """
        Non-essential function. Converts the key-set of the parameter_dictionary
        from human-readable strings to network's internal TensorFlow placeholders.
        This is useful when loading parameter matrices into the model
        from previously saved files. The dictionary returned by this function an be
        directly used as `ws` when calling mlp.evaluate() or mlp.compute_accuracy()

        :param parameter_dictionary:
               A dictionary containing numpy matrices for each of the network's weight matrices
               and bias vectors. The keys must correspond to the human-readable string generated
               by function mlp.to_clearnames().
        :return: A dictionary with the same values as input parameter_dictionary
                 but with keys replaced by internal TensorFlow placeholders.
        """
        return {self._params[k] :parameter_dictionary[k]  for k in self._param_keys}

