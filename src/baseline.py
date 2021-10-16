from typing import Tuple

import tensorflow as tf
import datetime

from matplotlib import pyplot as plt

# Constants
REGULARIZER = tf.keras.regularizers.l2(1e-3)

# Plot Constants
PLOT_METRIC = 'sparse_top_k_categorical_accuracy'
PLOT_LABEL = 'Top 5 Accuracy'
FIG_HEIGHT = 5
FIG_WIDTH = 15


class ResidualBlock(tf.keras.Model):
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
            x_short = tf.nn.relu(x_short)
            x_short = self.conv2Shortcut(x_short)

        # Block 1
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2a(x)

        # Block 2
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2b(x)

        # Block 3
        x = self.bn2c(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2c(x)

        # Output
        x += x_short

        return x


class ResNet50(tf.Module):
    checkpoint_path = "checkpoints/"
    saved_model_path = "saved_model/ResNet50/"

    def __init__(self, num_classes: int, img_size: int, optimizer: tf.keras.optimizers.Optimizer,
                 metric: tf.keras.metrics.Metric):
        super(ResNet50, self).__init__()
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

        self.model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           optimizer=optimizer, metrics=metric)

    def train(self, train_dataset: tf.data.Dataset, valid_dataset: tf.data.Dataset, num_epochs: int,
              steps_per_epoch: int):
        lr_decay = tf.keras.callbacks.LearningRateScheduler(self._lr_decay)

        # Used for TensorBoard
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path,
            verbose=1,
            save_weights_only=True,
            period=20,  # Saves every 20 epochs
            save_best_only=True)

        self.history = self.model.fit(train_dataset, epochs=num_epochs, validation_data=valid_dataset,
                                      steps_per_epoch=steps_per_epoch,
                                      callbacks=[lr_decay, tensorboard_callback, cp_callback])

    def test(self, test_dataset: tf.data.Dataset):
        self.model.evaluate(test_dataset)

    def load_weights(self):
        self.model.load_weights(self.checkpoint_path)

    def save(self):
        self.model.save(self.saved_model_path)

    # Plots accuracy over time
    def plot_accuracy(self):
        plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
        plt.plot(self.history.history[PLOT_METRIC])
        plt.plot(self.history.history['val_' + PLOT_METRIC])
        plt.title('Model ' + PLOT_LABEL)
        plt.xlabel('Epochs')
        plt.ylabel(PLOT_LABEL, rotation='horizontal', ha='right')
        plt.legend(['Train', 'Valid'], loc='upper left')
        plt.show()

    @staticmethod
    def _lr_decay(epoch: int):
        lr = 1e-3
        if epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1

        return lr
