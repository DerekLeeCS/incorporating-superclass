from typing import List, Dict, Tuple

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

# Datasets
from datasets.cifar_100_python.cifar_100 import CIFAR100

# Models
from models.base_module import BaseModule
from models.baseline import ResNet50
from models.baseline_auxiliary import ResNet50WithAux
from models.msgnet import MSGNet
from models.sgnet import SGNet
from models.scinet import SCINet

# From:
# https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
from tfrecord_handler import TFRecordHandler

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
IMG_SIZE = 32
BATCH_SIZE = 64
VALID_SIZE = 0.2
NUM_EPOCHS = 100
AUTOTUNE = tf.data.experimental.AUTOTUNE
OPTIMIZER = tf.keras.optimizers.Adam()
LOSS = tf.keras.losses.SparseCategoricalCrossentropy()
METRIC = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
IS_TRAINING = True


def preprocess(example: Dict) -> Tuple[tf.Tensor, Dict]:
    # Prepare the labels
    fine_label = example.pop('fine_label')
    coarse_label = example.pop('coarse_label')
    label = {
        BaseModule.get_output_fine_name(): tf.convert_to_tensor(fine_label),
        # BaseModule.get_output_coarse_name(): tf.convert_to_tensor(coarse_label),
    }

    return example['image'], label


if __name__ == '__main__':
    # Get data
    dataset = CIFAR100()
    train_dataset, valid_dataset, test_dataset = dataset.get_data()
    num_classes, num_superclasses = dataset.get_num_classes()
    num_train_examples = TFRecordHandler.count_size(train_dataset)

    # Define data augmentation
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(15 / 360),
        tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='nearest',
                                          interpolation='nearest'),
    ])

    train_dataset = (
        train_dataset
            .map(preprocess, num_parallel_calls=AUTOTUNE)
            .cache()
            .shuffle(tf.cast(num_train_examples, tf.int64))
            .batch(BATCH_SIZE)
            .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
            .prefetch(AUTOTUNE)
    )
    valid_dataset = (
        valid_dataset
            .batch(BATCH_SIZE)
            .map(preprocess, num_parallel_calls=AUTOTUNE)
            .cache()
            .prefetch(AUTOTUNE)
    )
    test_dataset = (
        test_dataset
            .batch(BATCH_SIZE)
            .map(preprocess, num_parallel_calls=AUTOTUNE)
            .cache()
            .prefetch(AUTOTUNE)
    )

    # Run model
    module = ResNet50(num_classes, IMG_SIZE, LOSS, OPTIMIZER, METRIC)
    if IS_TRAINING:
        module.train(train_dataset, valid_dataset, NUM_EPOCHS)
        module.load_weights()  # Ensure the best weights are used for saving
        module.save()
    else:
        module.load_weights()

    module.test(test_dataset)
