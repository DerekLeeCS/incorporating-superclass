import tensorflow as tf

from models.resnet50v2 import stack_blocks, residual_block
from models.base_module import BaseModule, REGULARIZER


class SGNet(BaseModule):
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

        # Auxiliary Classifier
        aux = x
        aux = residual_block(aux, filters=(256, 1024), conv_shortcut=True)
        aux = tf.keras.layers.BatchNormalization()(aux)
        aux = tf.keras.layers.ReLU()(aux)
        aux = tf.keras.layers.AveragePooling2D((2, 2))(aux)
        out_aux = tf.keras.layers.Flatten()(aux)
        out_aux = tf.keras.layers.Dense(num_superclasses, activation='softmax', kernel_regularizer=REGULARIZER,
                                        name=self._output_coarse_name)(out_aux)

        # Stage 3
        x = stack_blocks(x, filters=(256, 1024), num_blocks=6)

        # Stage 4
        x = stack_blocks(x, filters=(512, 2048), s=1, num_blocks=3)

        # Pooling
        x = tf.concat([x, aux], -1)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        # Output
        out_main = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=REGULARIZER,
                                         name=self._output_fine_name)(x)

        self.model = tf.keras.Model(inputs=inp, outputs=[out_main, out_aux])
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metric,
                           loss_weights={self._output_coarse_name: 0.5, self._output_fine_name: 0.5})
        self.model.summary()
