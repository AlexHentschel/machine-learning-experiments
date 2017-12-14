"""
Example for usage of class mlp.py.
Implements a Stochastic Gradient Decent (SGD).

Author: Alexander Hentschel, alex.hentschel@axiomzen.co
"""

import os
import numpy as np
import logging
import sys

from tensorflow.examples.tutorials.mnist import input_data
from neural_network_model.mlp import mnist_mlp


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
    return input_data.read_data_sets(abs_data_dir, one_hot=True)


# Please customize:
# =========================================================================== #

working_dir = "/Users/alex/Temp/MNIST"
batch_size = 1
training_epochs = 20
learning_rate = 0.5

# =========================================================================== #

if __name__ == "__main__":
    # initialization
    # ....................................................................
    logger = init_logger()
    mnist = import_data(working_dir, "MNIST_data")

    nn = mnist_mlp()   # create neural network
    nn.start()         # need to start up nn's internal TensorFlow engine to run any computations

    # Randomly generate initial values for neural network's weight matrices and bias vectors
    # This function returns a DICTIONARY with
    #    key: references to TensorFlow-internal variable
    #    value: numpy float matrix
    params = nn.generate_initial_network_parameters()

    # Stochastic gradient decent
    # ....................................................................
    for epoch in range(training_epochs):
        number_batches = int(mnist.train.num_examples / batch_size)
        mean_loss = 0.
        for i in range(number_batches):
            input_feature_batch, labels_batch = mnist.train.next_batch(batch_size)     # get mini-batch
            batch_loss, grad = nn.evaluate(input_feature_batch, labels_batch, params)  # compute mean loss and mean gradient over data batch
            for k in params.keys():                                                    # parameter update step
                params[k] -= learning_rate * grad[k]
            mean_loss += batch_loss / number_batches                                   # accumulate mean loss over entire dataset, i.e. over entire epoch
        logger.info("Mean loss after epoch %3d: %.9f", epoch+1, mean_loss)
    logger.info("Optimization Finished!")

    accuracy = nn.compute_accuracy(mnist.test.images, mnist.test.labels, params)
    logger.info("Accuracy of final model on test data set: %.3f%%" , accuracy * 100)

    # Cleanup
    # ....................................................................
    nn.close()

