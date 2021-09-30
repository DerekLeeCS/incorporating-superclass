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

    def call(self, inputTensor, training=False):
        x = inputTensor

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
        x += inputTensor

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

    def call(self, inputTensor, training=False):
        x = inputTensor
        xShort = inputTensor

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
        xShort = self.bn2Shortcut(xShort, training=training)
        xShort = tf.nn.relu(xShort)
        xShort = self.conv2Shortcut(xShort)

        # Output
        x += xShort

        return x


class ResNet50(tf.Module):
    def __init__(self, numClasses: int):
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
        self.model.add(tf.keras.layers.Dense(numClasses, activation='softmax', kernel_initializer='he_normal'))

    def train(self, dataGenTrain, trainImg, trainLabel, validImg, validLabel):
        lrdecay = tf.keras.callbacks.LearningRateScheduler(self.lrdecay)  # Learning rate decay
        self.model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           optimizer=OPTIMIZER, metrics=METRIC)

        trainSteps = int(trainImg.shape[0] / BATCH_SIZE)

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

        self.history = self.model.fit(dataGenTrain.flow(trainImg, trainLabel, batch_size=BATCH_SIZE), epochs=NUM_EPOCHS,
                                      validation_data=(validImg, validLabel),
                                      callbacks=[lrdecay, tensorboard_callback, cp_callback])

    def test(self, testImg, testLabel):
        self.model.evaluate(testImg, testLabel)

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
    def plotAccuracy(self):
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
    dataTrain = dataset.getData(True)
    dataTest = dataset.getData(False)
    numClasses = dataset.getNumClasses()

    imgTrain = dataTrain[b'data']
    labelTrain = dataTrain[b'fine_labels']
    imgTest = dataTest[b'data']
    labelTest = dataTest[b'fine_labels']

    # Reshapes each image into 32x32 and 3 channels ( RGB )
    imgTrain = np.reshape(imgTrain, [-1, 32, 32, 3], order='F')
    imgTest = np.reshape(imgTest, [-1, 32, 32, 3], order='F')

    # Train / Valid Split
    trainImg, validImg, trainLabel, validLabel = train_test_split(imgTrain, labelTrain, test_size=VALID_SIZE)

    # Convert to tensor
    trainImg = tf.convert_to_tensor(trainImg, dtype=tf.float32)
    trainLabel = tf.convert_to_tensor(trainLabel)
    validImg = tf.convert_to_tensor(validImg, dtype=tf.float32)
    validLabel = tf.convert_to_tensor(validLabel)
    testImg = tf.convert_to_tensor(imgTest, dtype=tf.float32)
    testLabel = tf.convert_to_tensor(labelTest)

    # Data Augmentation
    dataGenTrain = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                                                                   height_shift_range=0.1, horizontal_flip=True)
    dataGenTrain.fit(trainImg)

    # Run model
    model = ResNet50(numClasses)
    model.train(dataGenTrain, trainImg, trainLabel, validImg, validLabel)
    model.test(testImg, testLabel)
    model.plotAccuracy()
