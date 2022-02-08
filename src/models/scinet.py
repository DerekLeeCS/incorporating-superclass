from typing import Tuple

import tensorflow as tf

from models.baseline import stack_blocks, REGULARIZER
from models.base_module import BaseModule


class SCIN(tf.keras.layers.Layer):
    """Perform Superclass Conditional Instance Normalization."""
    def __init__(self, num_superclasses: int, num_channels: int):
        super().__init__()

        self.gamma = tf.Variable(tf.random.normal(shape=(num_superclasses, num_channels), mean=0, stddev=0.5),
                                 dtype=tf.float32, trainable=True)
        self.beta = tf.Variable(tf.random.normal(shape=(num_superclasses, num_channels), mean=0, stddev=0.5),
                                dtype=tf.float32, trainable=True)

    def call(self, x: tf.Tensor, super_ind: tf.Tensor, **kwargs) -> tf.Tensor:
        sig = tf.nn.embedding_lookup(self.gamma, super_ind)
        mu = tf.nn.embedding_lookup(self.beta, super_ind)

        # Add singleton dimension for broadcasting
        # BxC -> Bx1x1xC
        sig = tf.expand_dims(tf.expand_dims(sig, axis=1), axis=1)
        mu = tf.expand_dims(tf.expand_dims(mu, axis=1), axis=1)

        # Normalize
        x_mu, x_sig_squared = tf.nn.moments(x, axes=[-1])
        x_sig = tf.sqrt(x_sig_squared)

        # Add singleton dimension for broadcasting
        # BxC -> Bx1xC
        x_sig = tf.expand_dims(x_sig, axis=-1)
        x_mu = tf.expand_dims(x_mu, axis=-1)
        x = (x - x_mu) / x_sig

        # Scale and shift
        x = (x * sig) + mu

        return x


def residual_block_with_scin(inputs: Tuple[tf.Tensor, int], num_superclasses: int, filters: Tuple[int, int], s: int = 1,
                             conv_shortcut: bool = False) -> tf.Tensor:
    """
    :param inputs: a tuple containing the input tensor and the sparse superclass prediction
    :param num_superclasses: the number of superclasses in the dataset
    :param filters: a tuple containing the two filters for the residual block
    :param s: the stride to downsample with
    :param conv_shortcut: whether to use a convolutional shortcut or the identity shortcut
    :return: the output of the residual block
    """
    input_tensor, super_ind = inputs
    f1, f2 = filters
    k = 3  # Kernel size

    pre_act = tf.keras.layers.BatchNormalization()(input_tensor)
    pre_act = tf.keras.layers.ReLU()(pre_act)

    if conv_shortcut:
        x_short = tf.keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(s, s),
                                         kernel_regularizer=REGULARIZER)(pre_act)
    else:
        x_short = tf.keras.layers.MaxPooling2D((1, 1), strides=(s, s))(input_tensor) if s > 1 else input_tensor

    # Block 1
    x = tf.keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), kernel_regularizer=REGULARIZER)(pre_act)

    # Block 2
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(f1, kernel_size=(k, k), strides=(s, s), padding='same',
                               kernel_regularizer=REGULARIZER)(x)
    x = SCIN(num_superclasses, f1)(x, super_ind)

    # Block 3
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), kernel_regularizer=REGULARIZER)(x)

    # Output
    x += x_short

    return x


class SCINet(BaseModule):
    def __init__(self, num_classes: int, num_superclasses: int, img_size: int, loss: tf.keras.losses.Loss,
                 optimizer: tf.keras.optimizers.Optimizer, metric: tf.keras.metrics.Metric):
        super().__init__()
        inp = tf.keras.layers.Input(shape=(img_size, img_size, 3))

        x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', kernel_regularizer=REGULARIZER)(inp)
        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        # Stage 1
        x = stack_blocks(x, filters=(64, 256), num_blocks=3)

        # Stage 2
        x = stack_blocks(x, filters=(128, 512), num_blocks=4)

        # Stage 3
        x = stack_blocks(x, filters=(256, 1024), num_blocks=6)

        # Auxiliary Classifier
        aux = x
        aux = tf.keras.layers.GlobalAveragePooling2D()(aux)
        out_aux = tf.keras.layers.Dense(num_superclasses, activation='softmax', kernel_regularizer=REGULARIZER,
                                        name=self._output_coarse_name)(aux)

        # Superclass Conditional Instance Normalization
        super_ind = tf.argmax(out_aux, axis=-1)
        # x = SCIN(num_superclasses, 1024)(x, super_ind)

        # Stage 4
        x = residual_block_with_scin((x, super_ind), num_superclasses=num_superclasses, filters=(512, 2048), s=2)
        # x = stack_blocks(x, filters=(512, 2048), num_blocks=3)

        # Pooling
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        # Output
        out_main = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=REGULARIZER,
                                         name=self._output_fine_name)(x)

        self.model = tf.keras.Model(inputs=inp, outputs=[out_main, out_aux])
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metric,
                           loss_weights={self._output_coarse_name: 0.5, self._output_fine_name: 1})
        self.model.summary()
