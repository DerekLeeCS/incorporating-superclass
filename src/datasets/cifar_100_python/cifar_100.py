from pathlib import Path
from typing import Dict, Tuple
import pickle

import numpy as np
from matplotlib import pyplot as plt

from datasets.dataset_interface import Dataset


class CIFAR100(Dataset):
    _file_train = Path(__file__, '../data/train')
    _file_test = Path(__file__, '../data/test')
    _num_classes = 100
    _num_superclasses = 20

    @staticmethod
    def _unpickle(file: str) -> Dict:
        """Process an uncompressed file from the CIFAR-100 dataset into a dictionary.
        The dataset and the function is from:
        https://www.cs.toronto.edu/~kriz/cifar.html
        """
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def get_data(self, is_training: bool) -> Dict:
        if is_training:
            file_name = self._file_train
        else:
            file_name = self._file_test

        # Define which dictionary keys to keep from the unpickled file
        dict_examples = {
            b'fine_labels': [],
            b'coarse_labels': [],
            b'data': []
        }
        dict_examples.update(CIFAR100._unpickle(str(file_name)))

        # Reshape each image into 32x32 and 3 channels (RGB)
        dict_examples[b'data'] = np.reshape(dict_examples[b'data'], [-1, 3, 32, 32]).transpose([0, 2, 3, 1])

        # Normalize images
        dict_examples[b'data'] = dict_examples[b'data'] / 255

        return dict_examples

    def get_num_classes(self) -> Tuple[int, int]:
        return self._num_classes, self._num_superclasses


if __name__ == '__main__':
    dataset = CIFAR100()
    data = dataset.get_data(False)
    print(data.keys())
    print(data[b'coarse_labels'])

    # Display an image
    plt.imshow(data[b'data'][0])
    plt.show()
    print(data[b'data'][0])
