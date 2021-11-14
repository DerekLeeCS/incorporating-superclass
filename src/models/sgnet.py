import tensorflow as tf

from models.baseline import ResidualBlock
from models.base_module import BaseModule


class SGNet(BaseModule):
    def __init__(self, num_classes: int, num_superclasses: int, img_size: int, loss: tf.keras.losses.Loss,
                 optimizer: tf.keras.optimizers.Optimizer, metric: tf.keras.metrics.Metric):
        super().__init__()
        inp = tf.keras.layers.Input(shape=(img_size, img_size, 3))

        x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2))(inp)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

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
        aux = tf.keras.layers.Conv2D(2048, kernel_size=(1, 1), strides=(2, 2), padding='valid',
                                     kernel_regularizer=ResidualBlock.regularizer)(aux)
        aux = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(aux)
        out_aux = tf.keras.layers.Flatten()(aux)
        out_aux = tf.keras.layers.Dense(num_superclasses, activation='softmax', name=self._output_coarse_name)(out_aux)

        # Stage 4
        x = ResidualBlock(filters=(512, 2048), s=2)(x)
        for _ in range(2):
            x = ResidualBlock(filters=(512, 2048))(x)

        # Pooling
        x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)

        # Output
        x = tf.concat([x, aux], -1)
        x = tf.keras.layers.Flatten()(x)
        out_main = tf.keras.layers.Dense(num_classes, activation='softmax', name=self._output_fine_name)(x)

        self.model = tf.keras.Model(inputs=inp, outputs=[out_main, out_aux])
        self.model.compile(optimizer=optimizer, loss=loss,
                           loss_weights={self._output_coarse_name: 0.5, self._output_fine_name: 1},
                           metrics=metric)
        self.model.summary()