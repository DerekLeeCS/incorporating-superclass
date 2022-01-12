from pathlib import Path
from typing import Dict, Tuple, List
import pickle

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from tfrecord_handler import TFRecordHandler
from datasets.dataset_interface import Dataset

VALID_SIZE = 0.2
AUTOTUNE = tf.data.experimental.AUTOTUNE


class CIFAR100(Dataset):
    FILE_TRAIN = Path(__file__, '../data/train')
    FILE_TEST = Path(__file__, '../data/test')

    FILE_TFRECORD_TRAIN = Path(__file__, '../data/train.tfrecord')
    FILE_TFRECORD_VALID = Path(__file__, '../data/valid.tfrecord')
    FILE_TFRECORD_TEST = Path(__file__, '../data/test.tfrecord')

    _num_classes = 100
    _num_superclasses = 20

    def __init__(self):
        self.preprocess = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1. / 255),  # Normalize the values to a range of [0, 1]
        ])

    @staticmethod
    def _unpickle(file: str) -> Dict:
        """Process an uncompressed file from the CIFAR-100 dataset into a dictionary.
        The dataset and the function is from:
        https://www.cs.toronto.edu/~kriz/cifar.html
        """
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    @staticmethod
    def read_pickled_examples(file_name: str) -> Dict:
        # Define which dictionary keys to keep from the pickled examples
        examples = {
            b'fine_labels': [],
            b'coarse_labels': [],
            b'data': []
        }
        examples.update(CIFAR100._unpickle(file_name))

        # Reshape each image into 32x32 and 3 channels (RGB)
        examples[b'data'] = np.reshape(examples[b'data'], [-1, 3, 32, 32]).transpose([0, 2, 3, 1])

        return examples

    @staticmethod
    def _unpack_examples(examples: Dict) -> Tuple[List[np.array], List[int], List[int]]:
        return examples[b'data'], examples[b'fine_labels'], examples[b'coarse_labels']

    def preprocess_image(self, example: Dict) -> Dict:
        example['image'] = self.preprocess(example['image'])
        return example

    def preprocess_tfrecord(self, file_name: Path) -> tf.data.TFRecordDataset:
        """Apply preprocessing to each element in the dataset and cache the results for future use."""
        return (
            TFRecordHandler.read_examples(str(file_name))
                .map(self.preprocess_image, num_parallel_calls=AUTOTUNE)
                .cache()
        )

    def get_data(self) -> Tuple[tf.data.TFRecordDataset, tf.data.TFRecordDataset, tf.data.TFRecordDataset]:
        return self.preprocess_tfrecord(self.FILE_TFRECORD_TRAIN), self.preprocess_tfrecord(self.FILE_TFRECORD_VALID), \
               self.preprocess_tfrecord(self.FILE_TFRECORD_TEST)

    def get_num_classes(self) -> Tuple[int, int]:
        return self._num_classes, self._num_superclasses


def write_dataset_to_tfrecord():
    """Split the dataset into training, validation, and test sets. Write each to a TFRecord."""
    dataset = CIFAR100()

    train_data = CIFAR100.read_pickled_examples(str(dataset.FILE_TRAIN))
    test_data = CIFAR100.read_pickled_examples(str(dataset.FILE_TEST))

    # Associate the fine and coarse labels together so we can randomize them
    label = list(tuple(zip(train_data[b'fine_labels'], train_data[b'coarse_labels'])))
    train_img, valid_img, train_label, valid_label = train_test_split(train_data[b'data'], label,
                                                                      test_size=VALID_SIZE, random_state=1129)

    def unpack_labels(labels: List[Tuple[int, int]]) -> Tuple[List[int], List[int]]:
        """Return the fine and coarse labels, in that order."""
        return [x[0] for x in labels], [x[1] for x in labels]

    # Unpack the examples
    train_fine_label, train_coarse_label = unpack_labels(train_label)
    valid_fine_label, valid_coarse_label = unpack_labels(valid_label)
    test_img = test_data[b'data']
    test_fine_label = test_data[b'fine_labels']
    test_coarse_label = test_data[b'coarse_labels']

    TFRecordHandler.write_examples(str(dataset.FILE_TFRECORD_TRAIN), train_img, train_fine_label, train_coarse_label)
    TFRecordHandler.write_examples(str(dataset.FILE_TFRECORD_VALID), valid_img, valid_fine_label, valid_coarse_label)
    TFRecordHandler.write_examples(str(dataset.FILE_TFRECORD_TEST), test_img, test_fine_label, test_coarse_label)


if __name__ == '__main__':
    write_dataset_to_tfrecord()

    # Test reading a TFRecord
    import matplotlib.pyplot as plt

    dataset = CIFAR100()
    _, _, ds = dataset.get_data()
    for example in ds.take(1):
        img = example['image'].numpy()
        plt.imshow(img)
        plt.show()
        print(img)
        print(example['fine_label'], example['coarse_label'])
