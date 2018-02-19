"""
Variational Autoencoder in Keras using MNIST

This Code is HEAVILY based on Louis Tiao's blog post:
http://louistiao.me/posts/implementing-variational-autoencoders-in-keras-beyond-the-quickstart-tutorial/

"""

import sys
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.callbacks import Callback
from keras import backend as K

from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential
from keras.datasets import mnist



# ------------------------------------------------------------------------------

def init_logger(llevel=logging.DEBUG):
    logging.basicConfig(stream=sys.stdout, level=llevel)
    logger = logging.getLogger("variational_autoencoder")
    logger.setLevel(llevel)
    return logger

logger = init_logger()



# Loading and pre-processing data
# =======================================================================================
# MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape each image into one long vector
# and normalize gray-color level from [0,...,255] to range [0,1]
original_dim = 784
x_train = x_train.reshape(-1, original_dim) / 255.
x_test = x_test.reshape(-1, original_dim) / 255.




# Setup VAE
# =======================================================================================

intermediate_dim = 256
latent_dim = 2
epsilon_std = 1.0


def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """
    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs


# Generative model (Decoder): dimension-reduced representation [2] -> image vector [784]
# .......................................................................................
decoder = Sequential([
    Dense(intermediate_dim, input_dim=latent_dim, activation='relu'),
    Dense(original_dim, activation='sigmoid')
])
# Note: Sequential() is constructor for a Model

# Inference Network (Encoder): image vector [784] -> dimension-reduced representation [2]
# .......................................................................................
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)

z_mu = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
encoder = Model(x, z_mu)

# Stochastic sampling branch for generating random inputs for decoder
# .......................................................................................
z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)
eps = Input(tensor=K.random_normal(stddev=epsilon_std, shape=(K.shape(x)[0], latent_dim)))
z_eps = Multiply()([z_sigma, eps])
z = Add()([z_mu, z_eps])

# Stochastic sampling branch for generating random inputs for decoder
# .......................................................................................
vae = Model(inputs=[x, eps], outputs=decoder(z))

# Train VAE
# =======================================================================================
# for callbacks see: https://keunwoochoi.wordpress.com/2016/07/16/keras-callbacks/
class HistoryEncodings(Callback):
    def _save_batch(self):
        logger.debug("\nSaving batch %d", self.batch_count)
        self.encodings[self.batch_count] = encoder.predict(x_test)

    def on_train_begin(self, logs={}):
        self.batch_count = 0
        self.batch_separator = 1
        self.remaining_batches_to_skip = 1
        self.encodings = {}

    def on_batch_begin(self, batch, logs={}):
        if self.batch_count < 100:
            self._save_batch()
        else:
            if self.remaining_batches_to_skip < 1:
                self._save_batch()
                self.remaining_batches_to_skip = self.batch_separator
                self.batch_separator += 1
            else:
                self.remaining_batches_to_skip -= 1
        self.batch_count += 1
        return


batch_size = 100
epochs = 50

vae.compile(optimizer='rmsprop', loss=nll)
encoding_history = HistoryEncodings()
hist = vae.fit(x_train,
        x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test),
        callbacks=[encoding_history])


# Plot Results
# =======================================================================================

# Encodings of digits in dimension-reduced representation
# .......................................................................................
def plot_encodings(zs, box_subdomain=None, image_destination=None, close_image=False, plot_limits=10):
    """
    display a 2D plot of the digit classes in the latent space
    :param zs:
    :param box_subdomain:
    :param image_destination:
    :return:
    """
    # there is a problem with https://github.com/matplotlib/matplotlib/issues/8736
    f = plt.figure(figsize=(7,7))
    for k in range(10):
        mask = y_test == k
        plt.scatter(zs[mask, 0], zs[mask, 1], alpha=.2, s=40, edgecolor='none', label=str(k))
    lgnd = plt.legend(bbox_to_anchor=(1.12, 1), borderaxespad=0.0, scatterpoints=1, fontsize=10)
    for handle in lgnd.legendHandles:
        handle.set_sizes([60.0])
        handle.set_alpha(1.0)
    if box_subdomain is not None:
        plt.plot([-box_subdomain, box_subdomain], [box_subdomain, box_subdomain], color='black', linestyle=':', linewidth=1)
        plt.plot([-box_subdomain, box_subdomain], [-box_subdomain, -box_subdomain], color='black', linestyle=':', linewidth=1)
        plt.plot([box_subdomain, box_subdomain], [-box_subdomain, box_subdomain], color='black', linestyle=':', linewidth=1)
        plt.plot([-box_subdomain, -box_subdomain], [-box_subdomain, box_subdomain], color='black', linestyle=':', linewidth=1)
    plt.xlabel('Latent Variable $\mu_1$', fontsize=18)
    plt.ylabel('Latent Variable $\mu_2$', fontsize=18)
    plt.xlim([-plot_limits, plot_limits])
    plt.ylim([-plot_limits, plot_limits])
    plt.draw()
    plt.show()
    if image_destination is not None:
        image_destination = os.path.abspath(image_destination)
        logger.debug("saving image to '%s'", image_destination)
        dir = os.path.abspath(os.path.join(image_destination, os.pardir))
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(image_destination, dpi=100)
    if close_image:
        plt.close(f)

figure_dir = "/Users/alex/Temp/VAE-Training-Figures/"
figure_name_template = "encoding_batch%0" + str(len(str(encoding_history.batch_count)))+ "d.png"

for b, z_test in encoding_history.encodings.items():
    dest = os.path.join(figure_dir, figure_name_template % b)
    plot_encodings(z_test, image_destination=dest, close_image=True)

quantile_norm = 3.0902323061678132
z_test = encoder.predict(x_test)
dest = os.path.join(figure_dir, figure_name_template % encoding_history.batch_count)
plot_encodings(z_test, image_destination=dest, close_image=False)
dest = os.path.join(figure_dir, "zoomed_" + figure_name_template % encoding_history.batch_count)
plot_encodings(z_test, image_destination=dest, close_image=False, plot_limits=quantile_norm)

dest = os.path.join(figure_dir, "boxed_" + figure_name_template % encoding_history.batch_count)
plot_encodings(z_test, box_subdomain=quantile_norm, image_destination=dest, close_image=False)

# Reconstructed image from dimension-reduced representation
#  - plot reconstructions for a grid of values
# .......................................................................................

# original digit size (number pixels in each dimenstion):
digit_size = 28

# display a 2D manifold of the images
n = 15  # figure with 15x15 images
figure_data_grid = np.zeros((digit_size * n, digit_size * n))

# linearly spaced coordinates on the unit square were transformed
# through the inverse CDF (ppf) of the Gaussian to produce values
# of the latent variables z, since the prior of the latent space
# is Gaussian
quantile_min = 0.001
quantile_max = 0.999
z1 = norm.ppf(np.linspace(quantile_min, quantile_max, n))
z2 = norm.ppf(np.linspace(quantile_max, quantile_min, n))

for i, yi in enumerate(z2):
    for j, xi in enumerate(z1):
        z_sample = np.array([[xi, yi]]) * epsilon_std
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded.reshape(digit_size, digit_size)
        figure_data_grid[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(figure_data_grid)

ax.set_xticks(np.arange(0, n*digit_size, digit_size) + .5 * digit_size)
ax.set_xticklabels(map('{:.2f}'.format, z1), rotation=90)

ax.set_yticks(np.arange(0, n*digit_size, digit_size) + .5 * digit_size)
ax.set_yticklabels(map('{:.2f}'.format, z2))

plt.xlabel('Latent Variable $\mu_1$', fontsize=18)
plt.ylabel('Latent Variable $\mu_2$', fontsize=18)
plt.show()

figure_dir = "/Users/alex/Temp/VAE-Training-Figures/"
figure_name = "reconstruction.png"
plt.savefig(os.path.join(figure_dir, figure_name), dpi=100)
