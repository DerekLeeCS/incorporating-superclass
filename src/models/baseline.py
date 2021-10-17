from abc import ABC
from typing import Tuple

import tensorflow as tf

from models.base_module import BaseModule

# Constants
REGULARIZER = tf.keras.regularizers.l2(1e-3)


class ResidualBlock(tf.keras.Model, ABC):
    """Residual Block for a ResNet with Full Pre-activation"""

    def __init__(self, filters: Tuple[int, int], s: int = None):
        super(ResidualBlock, self).__init__(name='')
        f1, f2 = filters
        k = 3  # Kernel size

        if s is not None:
            self.downsample = True
            self.conv2a = tf.keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid',
                                                 kernel_regularizer=REGULARIZER)
            self.conv2Shortcut = tf.keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid',
                                                        kernel_regularizer=REGULARIZER)
            self.bn2Shortcut = tf.keras.layers.BatchNormalization()
        else:
            self.downsample = False
            self.conv2a = tf.keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                                 kernel_regularizer=REGULARIZER)

        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(f1, kernel_size=(k, k), strides=(1, 1), padding='same',
                                             kernel_regularizer=REGULARIZER)
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                             kernel_regularizer=REGULARIZER)
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor: tf.Tensor, training: bool = False, **kwargs):
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
        x = self.bn2b(x, training=training)
        x = tf.keras.layers.ReLU()(x)
        x = self.conv2b(x)

        # Block 3
        x = self.bn2c(x, training=training)
        x = tf.keras.layers.ReLU()(x)
        x = self.conv2c(x)

        # Output
        x += x_short

        return x


class ResNet50(BaseModule):
    def __init__(self, num_classes: int, img_size: int, loss: tf.keras.losses.Loss,
                 optimizer: tf.keras.optimizers.Optimizer, metric: tf.keras.metrics.Metric):
        super().__init__()
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape=(img_size, img_size, 3)))

        self.model.add(tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2)))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.ReLU())
        self.model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)))

        # Stage 1
        self.model.add(ResidualBlock(filters=(64, 256), s=1))
        for _ in range(2):
            self.model.add(ResidualBlock(filters=(64, 256)))

        # Stage 2
        self.model.add(ResidualBlock(filters=(128, 512), s=2))
        for _ in range(3):
            self.model.add(ResidualBlock(filters=(128, 512)))

        # Stage 3
        self.model.add(ResidualBlock(filters=(256, 1024), s=2))
        for _ in range(5):
            self.model.add(ResidualBlock(filters=(256, 1024)))

        # Stage 4
        self.model.add(ResidualBlock(filters=(512, 2048), s=2))
        for _ in range(2):
            self.model.add(ResidualBlock(filters=(512, 2048)))

        # Pooling
        self.model.add(tf.keras.layers.AveragePooling2D((2, 2), padding='same'))

        # Output
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metric)
        self.model.summary()
