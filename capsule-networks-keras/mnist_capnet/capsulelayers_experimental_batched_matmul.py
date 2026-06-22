"""
Capsule Network layers — EXPERIMENTAL VARIANT of `capsulelayers.py`.

  ============================ AI-GENERATED DRAFT =============================
  This file is an AI-authored draft. It is UNTESTED. Verify shapes, run the
  test suite, and benchmark against `capsulelayers.py` before relying on it.
  ============================================================================

Only the `CapsuleLayer` prediction-vector transform differs from
`capsulelayers.py`; `Length`, `Mask`, `squash`, and the routing algorithm are
unchanged and shape-identical. `PrimaryCap` is adapted only to emit the 5-D
input layout this transform consumes.

Difference — how `inputs_hat` (the prediction vectors u_hat) is computed:

  capsulelayers.py  (per-sample scan):
      inputs_hat = K.map_fn(lambda x: K.batch_dot(x, W, [2, 3]), elems=inputs_tiled)
      -> serialises the multiply-accumulate over the batch dimension.

  this file  (single fused batched matmul over a 5-D layout):
      inputs.shape = [None, 1,           input_num_capsule, 1,           input_dim_capsule]
           w.shape = [1,    num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
      inputs_hat   = squeeze( matmul(inputs, w, transpose_b=True), axis=3 )
                   = [None, num_capsule, input_num_capsule, dim_capsule]

Nature of the optimization:
  tf.matmul treats the last two axes as the matrix and broadcasts the leading
  (batch) axes. This (a) removes the per-batch scan, so the whole batch runs as
  one parallel device kernel, and (b) fuses the multiply-accumulate, so the
  reduction over input_dim_capsule happens inside the kernel — no large
  intermediate is written. (Contrast the elementwise K.sum(inputs * w, -1) form,
  which first materializes the full
  [None, num_capsule, input_num_capsule, dim_capsule, input_dim_capsule] product.)

Limits / when it does NOT help:
  - Requires tf.matmul batch-dim broadcasting: TF >= 1.14 / TF 2.x. Older TF
    raises on the singleton-vs-num_capsule batch mismatch.
  - It performs the same number of arithmetic operations as the scan; the win is
    parallelism (vs. the per-sample scan) and reduced memory traffic (vs. the
    elementwise multiply-reduce, which materializes a large intermediate), not a
    lower operation count. On CPU, or at small batch size / few capsules, the gain
    is therefore marginal.
  - It computes the same quantity as the scan via the same dynamic-routing
    algorithm; it is not a different model. Floating-point results may differ in
    the last bits, because the reduction order inside matmul differs from the
    scan's.

To utilize best:
  - Run on GPU (the batched matmul is where the parallelism pays off).
  - Use a reasonably large batch size and capsule counts to amortize kernel
    launch overhead.
  - Feed it the 5-D layout from this file's `PrimaryCap`: its singleton axis 1 is
    what lets the matmul broadcast against W's num_capsule axis without tiling,
    while its singleton axis 3 provides the matrix row dimension.

Transform/routing algorithm author: Xifeng Guo
(E-mail: guoxifeng1990@163.com, Github: https://github.com/XifengGuo/CapsNet-Keras).
"""

import numpy as np
import keras.backend as K
import tensorflow as tf
from keras import initializers, layers


class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, num_capsule, dim_vector] either by the capsule with max length or by an additional 
    input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the
    masked Tensor.
    For example:
        ```
        x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
        y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
        out = Mask()(x)  # out.shape=[8, 6]
        # or
        out2 = Mask()([x, y])  # out2.shape=[8,6]. Masked with true labels y. Of course y can also be manipulated.
        ```
    """
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = K.sqrt(K.sum(K.square(inputs), -1))
            # generate the mask which is a one-hot code.
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        # inputs.shape=[None, num_capsule, dim_capsule]
        # mask.shape=[None, num_capsule]
        # masked.shape=[None, num_capsule * dim_capsule]
        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config


def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


class CapsuleLayer(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar output of the
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector:
        input shape = [None, input_num_capsule, input_dim_capsule] and
       output shape = [None, num_capsule, dim_capsule].
    For Dense Layer, input_dim_capsule = dim_capsule = 1.

    *Experimental variant*: the prediction-vector transform is computed by a single
    batched matrix multiplication over a 5-D layout (see module docstring), rather
    than by scanning per sample. This requires the input to be the 5-D tensor
    [None, 1, input_num_capsule, 1, input_dim_capsule] produced by `PrimaryCap`.

    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        # This variant consumes the 5-D layout emitted by the adapted PrimaryCap:
        #   input_shape = [None, 1, input_num_capsule, 1, input_dim_capsule]
        assert len(input_shape) >= 5, \
            "The input Tensor should have shape=[None, 1, input_num_capsule, 1, input_dim_capsule]"
        self.input_num_capsule = input_shape[2]
        self.input_dim_capsule = input_shape[4]

        # Transform matrix, laid out for the batched matmul in call(), where its
        # last two axes (dim_capsule, input_dim_capsule) are the matrix and the
        # first three are batch axes:
        #   axis 0 (leading singleton): broadcasts over the input's None batch.
        #   axis 1 (num_capsule):       broadcasts against the input's singleton axis 1.
        #   axis 2 (input_num_capsule): matches the input's input_num_capsule axis.
        # The matmul (with transpose_b) then maps each input capsule's
        # input_dim_capsule vector to a dim_capsule prediction vector.
        self.w = self.add_weight(shape=[1, self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape = [None, 1,           input_num_capsule, 1,           input_dim_capsule]
        #      w.shape = [1,    num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        #
        # Fused, vectorized transform via batched matmul. tf.matmul treats the last
        # two axes as the matrix and BROADCASTS the leading (batch) axes, so the
        # singleton dims line up without tiling and without a large intermediate:
        #   matrix: inputs[..., 1, input_dim_capsule] @ (w[..., dim_capsule, input_dim_capsule])^T
        #         = [..., 1, dim_capsule]
        #   batch:  [None, 1, input_num_capsule]  broadcast  [1, num_capsule, input_num_capsule]
        #         = [None, num_capsule, input_num_capsule]
        #   => [None, num_capsule, input_num_capsule, 1, dim_capsule]
        # Squeeze the singleton (axis 3) for the routing-ready shape.
        # Requires matmul batch-dim broadcasting (TF >= 1.14 / TF 2.x).
        # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        inputs_hat = tf.squeeze(tf.matmul(inputs, self.w, transpose_b=True), axis=3)

        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.num_capsule, self.input_num_capsule].
        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # c.shape=[batch_size, num_capsule, input_num_capsule]
            c = tf.nn.softmax(b, dim=1)

            # c.shape =  [batch_size, num_capsule, input_num_capsule]
            # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
            # The first two dimensions as `batch` dimension, then
            # matmul: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
            # outputs.shape=[None, num_capsule, dim_capsule]
            outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))

            if i < self.routings - 1:
                # outputs.shape =  [None, num_capsule, dim_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension, then
                # matmul: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
                # b.shape=[batch_size, num_capsule, input_num_capsule]
                b += K.batch_dot(outputs, inputs_hat, [2, 3])
        # End: Routing algorithm -----------------------------------------------------------------------#

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules.

    *Experimental variant*: emits the 5-D layout
        [None, 1, num_primary_capsule, 1, dim_capsule]
    expected by this file's `CapsuleLayer`. The two singleton axes play distinct
    roles in that layer's batched matmul: the first (axis 1) is a batch axis that
    broadcasts against the transform weight's num_capsule axis, while the second
    (axis 3) is the row dimension of the left matrix (each input capsule enters as
    a 1 x dim_capsule row vector); it stays 1 through the matmul and is squeezed
    afterward, the output's dim_capsule appearing in the trailing axis.

    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_capsule: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: tensor with shape [None, 1, num_primary_capsule, 1, dim_capsule]
    """
    output = layers.Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv2d')(inputs)
    # For dim_capsule=8, n_channels=32: output is [None, 6, 6, 32*8].
    # num_primary_capsule = 6*6*32 = 1152.
    _number_capsules = np.prod(output.get_shape().as_list()[1:-1]) * n_channels
    # 5-D reshape target (batch dim excluded): [1, num_primary_capsule, 1, dim_capsule]
    # -> full shape [None, 1, num_primary_capsule, 1, dim_capsule].
    outputs = layers.Reshape(target_shape=[1, _number_capsules, 1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)
