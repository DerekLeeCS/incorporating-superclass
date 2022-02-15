from typing import Tuple

import tensorflow as tf

from models.base_module import BaseModule, REGULARIZER


def residual_block(input_tensor: tf.Tensor, filters: Tuple[int, int], s: int = 1,
                   conv_shortcut: bool = False) -> tf.Tensor:
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

    # Block 3
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), kernel_regularizer=REGULARIZER)(x)

    # Output
    x += x_short

    return x


def stack_blocks(x: tf.Tensor, filters: Tuple[int, int], num_blocks: int, s: int = 2) -> tf.Tensor:
    x = residual_block(x, filters, conv_shortcut=True)
    for _ in range(2, num_blocks):
        x = residual_block(x, filters)
    x = residual_block(x, filters, s)
    return x


class ResNet50v2(BaseModule):
    def __init__(self, num_classes: int, img_size: int, loss: tf.keras.losses.Loss,
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

        # Stage 4
        x = stack_blocks(x, filters=(512, 2048), num_blocks=3)

        # Pooling
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        # Output
        out = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=REGULARIZER,
                                    name=self._output_fine_name)(x)

        self.model = tf.keras.Model(inputs=inp, outputs=out)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metric)
        self.model.summary()
