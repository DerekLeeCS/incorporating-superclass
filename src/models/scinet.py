from typing import Tuple

import tensorflow as tf

from models.baseline import ResidualBlock
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


class ResidualBlockWithSCIN(tf.keras.Model):
    """Residual Block for a ResNet with Full Pre-activation"""

    regularizer = tf.keras.regularizers.l2(1e-3)

    def __init__(self, num_superclasses: int, filters: Tuple[int, int], s: int = None):
        super().__init__()
        f1, f2 = filters

        if s is not None:
            self.downsample = True
            self.conv2a = tf.keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid',
                                                 kernel_regularizer=self.regularizer)
            self.conv2Shortcut = tf.keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid',
                                                        kernel_regularizer=self.regularizer)
            self.bn2Shortcut = tf.keras.layers.BatchNormalization()
        else:
            self.downsample = False
            self.conv2a = tf.keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                                 kernel_regularizer=self.regularizer)

        self.bn2a = tf.keras.layers.BatchNormalization()

        # self.conv2b = tf.keras.layers.Conv2D(f1, kernel_size=(k, k), strides=(1, 1), padding='same',
        #                                      kernel_regularizer=self.regularizer)
        # self.bn2b = tf.keras.layers.BatchNormalization()
        self.scin = SCIN(num_superclasses, f1)

        self.conv2c = tf.keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                             kernel_regularizer=self.regularizer)
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, inputs: Tuple[tf.Tensor, int], training: bool = False, **kwargs):
        """
        :param inputs: a tuple containing the input tensor and the sparse superclass prediction
        :param training: a boolean indicating whether the layer should behave in training mode or in inference mode
        """
        input_tensor = inputs[0]
        super_ind = inputs[1]

        x = input_tensor
        x_short = input_tensor

        if self.downsample:
            x_short = self.bn2Shortcut(x_short, training=training)
            x_short = tf.keras.layers.ReLU()(x_short)
            x_short = self.conv2Shortcut(x_short)

        # Block 1
        x = self.bn2a(x, training=training)
        x = tf.keras.layers.ReLU()(x)
        x = self.conv2a(x)

        # Block 2
        x = self.scin(x, super_ind)

        # Block 3
        x = self.bn2c(x, training=training)
        x = tf.keras.layers.ReLU()(x)
        x = self.conv2c(x)

        # Output
        x += x_short

        return x


class SCINet(BaseModule):
    def __init__(self, num_classes: int, num_superclasses: int, img_size: int, loss: tf.keras.losses.Loss,
                 optimizer: tf.keras.optimizers.Optimizer, metric: tf.keras.metrics.Metric):
        super().__init__()
        inp = tf.keras.layers.Input(shape=(img_size, img_size, 3))

        x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inp)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        # Stage 1
        x = ResidualBlock(filters=(64, 256), s=1)(x)
        for _ in range(2):
            x = ResidualBlock(filters=(64, 256))(x)

        # Stage 2
        x = ResidualBlock(filters=(128, 512), s=2)(x)
        for _ in range(3):
            x = ResidualBlock(filters=(128, 512))(x)

        # Stage 3
        x = ResidualBlock(filters=(256, 1024), s=2)(x)
        for _ in range(5):
            x = ResidualBlock(filters=(256, 1024))(x)

        # Auxiliary Classifier
        aux = x
        aux = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(aux)
        aux = tf.keras.layers.Flatten()(aux)
        out_aux = tf.keras.layers.Dense(num_superclasses, activation='softmax', name=self._output_coarse_name)(aux)

        # Superclass Conditional Instance Normalization
        super_ind = tf.argmax(out_aux, axis=-1)
        # x = SCIN(num_superclasses, 1024)(x, super_ind)

        # Stage 4
        x = ResidualBlockWithSCIN(num_superclasses=num_superclasses, filters=(512, 2048), s=2)((x, super_ind))
        # x = ResidualBlock(filters=(512, 2048), s=2)(x)
        for _ in range(2):
            x = ResidualBlock(filters=(512, 2048))(x)

        # Pooling
        x = tf.keras.layers.AveragePooling2D((2, 2))(x)

        # Output
        x = tf.keras.layers.Flatten()(x)
        out_main = tf.keras.layers.Dense(num_classes, activation='softmax', name=self._output_fine_name)(x)

        self.model = tf.keras.Model(inputs=inp, outputs=[out_main, out_aux])
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metric,
                           loss_weights={self._output_coarse_name: 0.5, self._output_fine_name: 1})
        self.model.summary()
