from typing import List, Dict, Tuple

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from datasets.cifar100 import CIFAR100
from models.base_module import BaseModule
from models.baseline import ResNet50
from models.baseline_auxiliary import ResNet50WithAux
from models.msgnet import MSGNet
from models.sgnet import SGNet

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
IMG_SIZE = 32
BATCH_SIZE = 64
VALID_SIZE = 0.2
NUM_EPOCHS = 160
AUTOTUNE = tf.data.experimental.AUTOTUNE
OPTIMIZER = tf.keras.optimizers.Adam()
LOSS = tf.keras.losses.SparseCategoricalCrossentropy()
METRIC = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
IS_TRAINING = True


class MultiOutputDataGenerator(tf.keras.preprocessing.image.ImageDataGenerator):
    """A custom ImageDataGenerator that takes a single input and produces multiple outputs.
    From:
    https://github.com/keras-team/keras/issues/12639
    """

    def flow(self,
             x,
             y=None,
             batch_size=32,
             shuffle=True,
             sample_weight=None,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png',
             subset=None):

        targets = None
        target_lengths = {}
        ordered_outputs = []
        for output, target in y.items():
            if targets is None:
                targets = target
            else:
                targets = np.concatenate((targets, target), axis=1)
            target_lengths[output] = target.shape[1]
            ordered_outputs.append(output)

        for flow_x, flow_y in super().flow(x, targets, batch_size=batch_size,
                                           shuffle=shuffle):
            target_dict = {}
            i = 0
            for output in ordered_outputs:
                target_length = target_lengths[output]
                target_dict[output] = flow_y[:, i: i + target_length]
                i += target_length

            yield flow_x, target_dict


def get_formatted_labels(fine_label: List[int], coarse_label: List[int]) -> Dict[str, np.ndarray]:
    """Convert the provided labels into the required format for the model.
    Create a dictionary to map the output layer to the corresponding output labels.
    Add a singleton dimension to each label: N -> Nx1.
    """
    return {
        BaseModule.get_output_fine_name(): tf.reshape(tf.convert_to_tensor(fine_label), [-1, 1]),
        BaseModule.get_output_coarse_name(): tf.reshape(tf.convert_to_tensor(coarse_label), [-1, 1]),
    }


def get_train_valid_split(img: np.ndarray, fine_label: List[int], coarse_label: List[int]) -> \
        Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Split the provided images and labels into a training and validation set.
    Convert the labels into the required format for the model.

    Returns:
        Training images, validation images, training labels, validation images, in that order.
    """
    # Associate the fine and coarse labels together so we can randomize them
    label = list(tuple(zip(fine_label, coarse_label)))

    # Split into training and validation sets
    train_img, valid_img, train_label, valid_label = train_test_split(img, label, test_size=VALID_SIZE)
    train_label = get_formatted_labels(fine_label=[x[0] for x in train_label], coarse_label=[x[1] for x in train_label])
    valid_label = get_formatted_labels(fine_label=[x[0] for x in valid_label], coarse_label=[x[1] for x in valid_label])

    return train_img, valid_img, train_label, valid_label


if __name__ == '__main__':
    # Get data
    dataset = CIFAR100()
    data_train = dataset.get_data(True)
    data_test = dataset.get_data(False)
    num_classes, num_superclasses = dataset.get_num_classes()

    # Extract data
    train_img, valid_img, train_label, valid_label = get_train_valid_split(img=data_train[b'data'],
                                                                           fine_label=data_train[b'fine_labels'],
                                                                           coarse_label=data_train[b'coarse_labels'])
    test_img = data_test[b'data']
    test_label = get_formatted_labels(fine_label=data_test[b'fine_labels'], coarse_label=data_test[b'coarse_labels'])

    # Calculate number of steps per epoch
    steps_per_epoch = int(tf.shape(train_img)[0] / BATCH_SIZE)

    # Data Augmentation
    data_gen_train = MultiOutputDataGenerator(rotation_range=15, width_shift_range=0.1,
                                              height_shift_range=0.1, horizontal_flip=True)
    data_gen_train.fit(train_img)

    # Convert to Dataset
    train_dataset = (
        tf.data.Dataset.from_generator(lambda: data_gen_train.flow(train_img, train_label, batch_size=BATCH_SIZE),
                                       output_signature=(
                                           tf.TensorSpec(shape=(BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
                                           {output_name: tf.TensorSpec(shape=(BATCH_SIZE, 1), dtype=tf.int32)
                                            for output_name in train_label}
                                       ))
        .prefetch(AUTOTUNE)
    )
    valid_dataset = (
        tf.data.Dataset.from_tensor_slices((valid_img, valid_label))
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(AUTOTUNE)
    )
    test_dataset = (
        tf.data.Dataset.from_tensor_slices((test_img, test_label))
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(AUTOTUNE)
    )

    # Run model
    model = MSGNet(num_classes, num_superclasses, IMG_SIZE, LOSS, OPTIMIZER, METRIC)
    if IS_TRAINING:
        model.train(train_dataset, valid_dataset, NUM_EPOCHS, steps_per_epoch)
        model.load_weights()  # Ensure the best weights are used for saving
        model.save()
    else:
        model.load_weights()

    model.test(test_dataset)
