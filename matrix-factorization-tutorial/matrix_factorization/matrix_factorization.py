"""
TUTORIAL implementation of Tensor Factorization in TensorFlow.

NOTE: this code designed as a tutorial. It is not a high-performance production implementation.

Author: Alexander Hentschel
        alex.hentschel@axiomzen.co
"""

import tensorflow as tf
from matrix_factorization.synthetic_data_generator import synthetic_data_generator
import matrix_factorization.utils as utils


import os
import numpy as np
import pandas as pd
import logging
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime



# ##################################################################################### #
#                                   PREDICTOR MODEL                                     #
# ##################################################################################### #

# ======================================= MODEL =========================================
# Model properties:
# o number of users: number_users
# o number of items: number_items
# o number of hidden classes the model is supposed to learn: number_hidden_classes
# The model consists of
#  o User preference Matrix: matrix of dimension [number_users x number_hidden_classes]
#  o Item relevance Matrix:  matrix of dimension [number_items x number_hidden_classes]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

number_users = 100
number_items = 1000
number_hidden_classes = 5


# User Preference Model
prefmod = tf.get_variable("Preferences", [number_users, number_hidden_classes], dtype=tf.float64)

# Item Relevance Model
relmod = tf.get_variable("Relevances", [number_items, number_hidden_classes], dtype=tf.float64)


# ============================= Variables for Training data =============================
#  Defining arrays for holding the input data (during training)
# * Each data point is represented as a triple (u_n, d_n, r_n)
#   for - n the data point index
#       - u_n the user's index
#       - d_n the item's index
#       - r_n the (average) ranking score given by user u_n to item d_n
# * Training: - (u_n, d_n) is the input
#             - r_n is the target
#   Note that (u_n, d_n) are integer indices while r_n is a floating point number.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# we use a fixed batch size (this is not necessary but more instructive for the tutorial)
batch_size = 100

# During training, we will hand an existing array to the TensorFlow backend
# Hence, we define array references
user_indices = tf.placeholder(tf.int32, [batch_size])
item_indices = tf.placeholder(tf.int32, [batch_size])
target_score = tf.placeholder(tf.float64, [batch_size])



# ================================== Model Predictions ==================================
#  Define how to compute the model predictions for the given training batch:
#   o for one pair (u,d)
#       - u:  user's index
#       - d: item's index
#     the predicted score is
#        sum( prefmod[u] * relmod[d] )
# Below, we compute the prediction over the entire mini-batch:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# for each pair (u,d): select rows of preference matrix and relevance matrix
prefs_batch = tf.gather(prefmod, user_indices, axis=0)  # selects the rows specified by user_indices from the preference matrix
rels_batch = tf.gather(relmod, item_indices, axis=0)    # selects the rows specified by item_indices from the relevance matrix

# for each pair (u,d): compute predictions
predictions = tf.reduce_sum(tf.multiply(prefs_batch, rels_batch), axis=1) # vector over mini-batch


# ##################################################################################### #
#                                        OPTIMIZER                                      #
# ##################################################################################### #


# ==================================== Loss Function ====================================
#  The loss function has two components:
#   o regression loss: errors in model predictions compared to observed targets
#   o regularization loss: complexity of the model
#                         (models with lower complexity are assumed to have less bias)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# REGRESSION LOSS: Mean Squared Error (MSE)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# average difference between prediction and target
squared_errors = tf.squared_difference(predictions, target_score)   # squared errors (vector over mini-batch)
regression_loss = tf.reduce_mean(squared_errors)                    # mean (scalar) of the squared errors


# REGULARIZATION LOSS: l1 norm
#  o average absolute value of the preference matrix entries
#  o average absolute value of the preference matrix entries
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pref_regularizagion_loss = tf.norm(prefmod, ord=1) / np.prod(prefmod.get_shape().as_list())
rel_regularizagion_loss = tf.norm(relmod, ord=1) / np.prod(relmod.get_shape().as_list())
regularization_loss = pref_regularizagion_loss + rel_regularizagion_loss

# TOTAL LOSS:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
regularization_weight = 0.01
total_loss = regression_loss + regularization_weight * regularization_loss


# =================================== Gradient Decent ===================================
#  Here, we use simple gradient decent with a fixed learning rate.
#  (Better choices for production implementations are adaptive gradient decent
#  optimizers, such as Adam)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

learning_rate = 0.8
trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(total_loss, var_list=[prefmod, relmod])


# ##################################################################################### #
#                                       TRAINING                                        #
# ##################################################################################### #


# ================================== Create Some Data ===================================
# generate some synthetic training data
full_data = synthetic_data_generator(number_users, number_items, number_hidden_classes).full_observations

# shuffle data and randomly select
full_data_shuffled = full_data.sample(frac=1)

# In total, there are 100,000 possible observations. we will use 5% for training
# and track the model's performance on the remaining 95%.
# NOTE: generally, we don't have access to the full possible data set. We also
#       don't use it for training here. Only to show that the model actually learns
#       generate good predictions using only 5% of theoretical full data set.
training_data = full_data_shuffled.iloc[:10000]
test_data = full_data_shuffled.iloc[10000:]

# =================================== Initialize Model ==================================
#  Generate initial random values for
#   o preference matrix
#   o relevance matrix
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



preference_matrix_initialization = utils.random_matrix([number_users, number_hidden_classes])
relevance_matrix_initialization = utils.random_matrix([number_items, number_hidden_classes])


# =================================== Run Training ==================================

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# load initial values for preference and relevance matrix into model
sess.run(tf.assign(prefmod, preference_matrix_initialization))
sess.run(tf.assign(relmod, relevance_matrix_initialization))



# run 20 times (epochs) over training data:
training_epochs = 50
training_metrics = pd.DataFrame(index=np.arange(training_epochs+1), columns=["Training loss", "Mean Squared Error on Test Data"])

for epoch in range(training_epochs):
    # compute performance metrics on model
    training_metrics.loc[epoch, "Training loss"] = utils.compute_loss(sess, total_loss, batch_size, training_data, user_indices, item_indices, target_score)
    training_metrics.loc[epoch, "Mean Squared Error on Test Data"] = utils.compute_loss(sess, regression_loss, batch_size, test_data, user_indices, item_indices, target_score)
    # run one training epoch
    idx = 0
    while (idx+batch_size < len(training_data)):
        # take a mini-batch of training data and select values for
        batch = training_data.iloc[idx : idx+batch_size]
        sess.run(trainer, feed_dict={ user_indices: batch['user index'].as_matrix(),
                                      item_indices: batch['item index'].as_matrix(),
                                      target_score: batch['score'].as_matrix()
                                     })
        idx += batch_size
    print("Traing epoch % 4d completed" % (epoch+1))

# compute performance metrics on final model
training_metrics.loc[epoch+1, "Training loss"] = utils.compute_loss(sess, total_loss, batch_size, training_data, user_indices, item_indices, target_score)
training_metrics.loc[epoch+1, "Mean Squared Error on Test Data"] = utils.compute_loss(sess, regression_loss, batch_size, test_data, user_indices, item_indices, target_score)


sess.close()



# =================================== Output training progress ==================================
print(training_metrics)
training_metrics.plot()






















