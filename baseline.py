import tensorflow as tf
import numpy as np
import datetime

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from datasets.cifar100 import CIFAR100

# From:
# https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

# Constants
NUM_EPOCHS = 160
BATCH_SIZE = 64
VALID_SIZE = 0.2
OPTIMIZER = tf.keras.optimizers.Adam()
REGULARIZER = tf.keras.regularizers.l2(0.001)
METRIC = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)

# Plot Constants
PLOT_METRIC = 'sparse_top_k_categorical_accuracy'
PLOT_LABEL = 'Top 5 Accuracy'
FIG_HEIGHT = 5
FIG_WIDTH = 15


class IdentityBlock(tf.keras.Model):
    """Identity Block for a ResNet with Full Pre-activation"""

    def __init__(self, filters):
        super(IdentityBlock, self).__init__(name='')
        f1, f2 = filters
        k = 3  # Kernel size

        self.conv2a = tf.keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                             kernel_regularizer=REGULARIZER)
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(f1, kernel_size=(k, k), strides=(1, 1), padding='same',
                                             kernel_regularizer=REGULARIZER)
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                             kernel_regularizer=REGULARIZER)
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = input_tensor

        # Block 1
        x = self.bn2a(x, training=training)
        x = tf.nn.leaky_relu(x)
        x = self.conv2a(x)

        # Block 2
        x = self.bn2b(x, training=training)
        x = tf.nn.leaky_relu(x)
        x = self.conv2b(x)

        # Block 3
        x = self.bn2c(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2c(x)

        # Output
        x += input_tensor

        return x


class ConvBlock(tf.keras.Model):
    """Convolutional Block for a ResNet with Full Pre-activation"""

    def __init__(self, filters, s):
        super(ConvBlock, self).__init__(name='')
        f1, f2 = filters
        k = 3  # Kernel size

        self.conv2a = tf.keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid',
                                             kernel_regularizer=REGULARIZER)
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(f1, kernel_size=(k, k), strides=(1, 1), padding='same',
                                             kernel_regularizer=REGULARIZER)
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                             kernel_regularizer=REGULARIZER)
        self.bn2c = tf.keras.layers.BatchNormalization()

        self.conv2Shortcut = tf.keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid',
                                                    kernel_regularizer=REGULARIZER)
        self.bn2Shortcut = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = input_tensor
        x_short = input_tensor

        # Block 1
        x = self.bn2a(x, training=training)
        x = tf.nn.leaky_relu(x)
        x = self.conv2a(x)

        # Block 2
        x = self.bn2b(x, training=training)
        x = tf.nn.leaky_relu(x)
        x = self.conv2b(x)

        # Block 3
        x = self.bn2c(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2c(x)

        # Shortcut
        x_short = self.bn2Shortcut(x_short, training=training)
        x_short = tf.nn.relu(x_short)
        x_short = self.conv2Shortcut(x_short)

        # Output
        x += x_short

        return x


class ResNet50(tf.Module):
    def __init__(self, num_classes: int):
        self.model = tf.keras.models.Sequential()

        self.model.add(tf.keras.layers.ZeroPadding2D((3, 3)))
        self.model.add(tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2)))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.ReLU())
        self.model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)))

        # Stage 1
        self.model.add(ConvBlock(filters=[64, 256], s=1))
        for _ in range(2):
            self.model.add(IdentityBlock(filters=[64, 256]))

        # Stage 2
        self.model.add(ConvBlock(filters=[128, 512], s=2))
        for _ in range(3):
            self.model.add(IdentityBlock(filters=[128, 512]))

        # Stage 3
        self.model.add(ConvBlock(filters=[256, 1024], s=2))
        for _ in range(5):
            self.model.add(IdentityBlock(filters=[256, 1024]))

        # Stage 4
        self.model.add(ConvBlock(filters=[512, 2048], s=2))
        for _ in range(2):
            self.model.add(IdentityBlock(filters=[512, 2048]))

        # Pooling
        self.model.add(tf.keras.layers.AveragePooling2D((2, 2), padding='same'))

        # Output
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(num_classes, activation='softmax', kernel_initializer='he_normal'))

    def train(self, data_gen_train, train_img, train_label, valid_img, valid_label):
        lrdecay = tf.keras.callbacks.LearningRateScheduler(self.lrdecay)
        self.model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           optimizer=OPTIMIZER, metrics=METRIC)

        # Used for TensorBoard
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Create a callback that saves the model's weights
        checkpoint_path = "checkpoints/"
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            period=20,  # Saves every 20 epochs
            save_best_only=True)

        self.history = self.model.fit(data_gen_train.flow(train_img, train_label, batch_size=BATCH_SIZE), epochs=NUM_EPOCHS,
                                      validation_data=(valid_img, valid_label),
                                      callbacks=[lrdecay, tensorboard_callback, cp_callback])

    def test(self, test_img, test_label):
        self.model.evaluate(test_img, test_label)

    def lrdecay(self, epoch):
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1

        return lr

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


if __name__ == '__main__':
    # Get data
    dataset = CIFAR100()
    data_train = dataset.get_data(True)
    data_test = dataset.get_data(False)
    num_classes = dataset.get_num_classes()

    # Extract data
    img_train = data_train[b'data']
    label_train = data_train[b'fine_labels']
    img_test = data_test[b'data']
    label_test = data_test[b'fine_labels']

    # Train / Valid Split
    train_img, valid_img, train_label, valid_label = train_test_split(img_train, label_train, test_size=VALID_SIZE)

    # Convert to tensor
    train_img = tf.convert_to_tensor(train_img, dtype=tf.float32)
    train_label = tf.convert_to_tensor(train_label)
    valid_img = tf.convert_to_tensor(valid_img, dtype=tf.float32)
    valid_label = tf.convert_to_tensor(valid_label)
    test_img = tf.convert_to_tensor(img_test, dtype=tf.float32)
    test_label = tf.convert_to_tensor(label_test)

    # Data Augmentation
    data_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                                                                     height_shift_range=0.1, horizontal_flip=True)
    data_gen_train.fit(train_img)

    # Run model
    model = ResNet50(num_classes)
    model.train(data_gen_train, train_img, train_label, valid_img, valid_label)
    model.test(test_img, test_label)
    model.plot_accuracy()
