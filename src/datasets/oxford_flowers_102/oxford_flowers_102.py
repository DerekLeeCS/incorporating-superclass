import math
import os
import pickle
from pathlib import Path
from typing import Dict, Tuple

import tensorflow as tf

from datasets.oxford_flowers_102.superclass import SUPERCLASS_MAPPINGS_FILE_NAME
from datasets.dataset_interface import Dataset
from tfrecord_handler import TFRecordHandler

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_image(file_name: str) -> tf.Tensor:
    """Read an image given the file name.
    From:
    https://www.tensorflow.org/api_docs/python/tf/io/read_file
    """
    raw = tf.io.read_file(file_name)
    return tf.image.decode_jpeg(raw, channels=3)


class OxfordFlowers102(Dataset):
    # Note that we swap the train and test sets because the test set is larger than the train set
    FILE_TRAIN = Path(__file__, '../data/test.txt')
    FILE_TEST = Path(__file__, '../data/train.txt')
    FILE_VALID = Path(__file__, '../data/valid.txt')

    DIR_TFRECORD_TRAIN = Path(__file__, '../data/train/')
    DIR_TFRECORD_VALID = Path(__file__, '../data/valid/')
    DIR_TFRECORD_TEST = Path(__file__, '../data/test/')

    _FILE_IMAGE_LABELS = Path(__file__, '../data/labels.txt')
    _DIR_IMAGE = Path(__file__, '../data/jpg')

    _IMG_SIZE = 128
    _BATCH_SIZE = 32

    def __init__(self):
        with open(SUPERCLASS_MAPPINGS_FILE_NAME, 'rb') as f:
            self.subclass_to_superclass = pickle.load(f)

        self._num_classes = len(set(self.subclass_to_superclass.keys()))
        self._num_superclasses = len(set(self.subclass_to_superclass.values()))

    @staticmethod
    def _get_file_name(image_id: str) -> str:
        """Get the file name corresponding to the provided image id. Remove any whitespace in the image_id (e.g. '\n').
        Ex:
            1 -> image_00001.jpg
            1234 -> image_01234.jpg
        """
        return 'image_' + image_id.strip().rjust(5, '0') + '.jpg'

    def read_examples(self, file_name: Path) -> Dict:
        dict_data = {
            'fine_labels': [],
            'coarse_labels': [],
            'data': []
        }

        # Get the mappings from image ids to image labels
        with open(self._FILE_IMAGE_LABELS, 'r') as f:
            image_labels = f.read().split(',')
            image_id_to_label = {k + 1: int(v) for k, v in enumerate(image_labels)}

        with open(file_name, 'r') as f:
            image_ids = f.read().split(',')
            for image_id in image_ids:
                # Read the image
                image_file_name = OxfordFlowers102._get_file_name(image_id)
                image_file_path = str(self._DIR_IMAGE.joinpath(image_file_name))
                dict_data['data'].append(load_image(image_file_path))

                # Get the labels
                fine_label = image_id_to_label[int(image_id)]
                dict_data['fine_labels'].append(fine_label)
                dict_data['coarse_labels'].append(self.subclass_to_superclass[fine_label])

        return dict_data

    def preprocess_tfrecord(self, dir_name: Path) -> tf.data.TFRecordDataset:
        """Apply preprocessing to each element in the dataset and cache the results for future use."""

        resize_and_rescale = tf.keras.Sequential([
            tf.keras.layers.Resizing(self._IMG_SIZE, self._IMG_SIZE),
            tf.keras.layers.Rescaling(1. / 255),  # Normalize the values to a range of [0, 1]
        ])

        def preprocess_example(example: Dict) -> Dict:
            example['image'] = resize_and_rescale(example['image'])

            # Shift labels from (1 to N) to (0 to N-1) for the correct format for sparse categorical crossentropy
            example['fine_label'] -= 1
            example['coarse_label'] -= 1

            return example

        # Get the absolute path for every TFRecord in the directory
        file_names = [os.path.join(dir_name, file_name) for file_name in os.listdir(dir_name)]

        return (
            TFRecordHandler.read_examples(file_names)
                .map(preprocess_example, num_parallel_calls=AUTOTUNE)
        )

    def get_data(self) -> Tuple[tf.data.TFRecordDataset, tf.data.TFRecordDataset, tf.data.TFRecordDataset]:
        return self.preprocess_tfrecord(self.DIR_TFRECORD_TRAIN), self.preprocess_tfrecord(self.DIR_TFRECORD_VALID), \
               self.preprocess_tfrecord(self.DIR_TFRECORD_TEST)

    def get_batch_size(self) -> int:
        return self._BATCH_SIZE

    def get_image_size(self) -> int:
        return self._IMG_SIZE

    def get_num_classes(self) -> Tuple[int, int]:
        return self._num_classes, self._num_superclasses


# We want 10 TFRecords per host, as long as each TFRecord is 100 MB+
# https://www.tensorflow.org/tutorials/load_data/tfrecord
def write_dataset_to_tfrecord():
    """Split the dataset into training, validation, and test sets. Write each to a TFRecord."""
    dataset = OxfordFlowers102()

    def write_split_to_tfrecord(data_file_name: Path, dir_name: Path, num_files: int):
        """Write a dataset split to a TFRecord.

        :param data_file_name: the name of the file that contains the raw data
        :param dir_name: the name of the directory to save the TFRecords to
        :param num_files: the number of files to divide the data into, where each chunk is 100 MB+
        """
        # Create the TFRecord directory if needed
        os.makedirs(dir_name, exist_ok=True)

        data = dataset.read_examples(data_file_name)

        # Calculate the amount of data in each TFRecord
        n = len(data['data'])
        chunk_size = math.ceil(n / num_files)

        # Write the data to a TFRecord in chunks
        count = 1
        for i in range(0, n, chunk_size):
            file_name = dir_name / (str(count) + '.tfrecord')
            TFRecordHandler.write_examples(file_name, data['data'][i:i + chunk_size],
                                           data['fine_labels'][i:i + chunk_size],
                                           data['coarse_labels'][i:i + chunk_size])
            count += 1

    # Write each split to a TFRecord
    write_split_to_tfrecord(dataset.FILE_TRAIN, dataset.DIR_TFRECORD_TRAIN, 10)
    write_split_to_tfrecord(dataset.FILE_VALID, dataset.DIR_TFRECORD_VALID, 10)
    write_split_to_tfrecord(dataset.FILE_TEST, dataset.DIR_TFRECORD_TEST, 10)


if __name__ == '__main__':
    write_dataset_to_tfrecord()

    # Test reading a TFRecord
    import matplotlib.pyplot as plt

    dataset = OxfordFlowers102()

    _, _, ds = dataset.get_data()
    for example in ds.take(5):
        img = example['image'].numpy()
        plt.imshow(img)
        plt.show()
        print(img)
        print(example['fine_label'], example['coarse_label'])
