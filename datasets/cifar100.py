from pathlib import Path
from typing import Dict
import pickle

from datasetInterface import Dataset


class CIFAR100(Dataset):
    _file_train = Path(__file__, '../cifar-100-python/train')
    _file_test = Path(__file__, '../cifar-100-python/test')
    _num_classes = 100

    """This function processes an uncompressed file from the CIFAR100 dataset into a dictionary.
    The dataset and the function is from:
    https://www.cs.toronto.edu/~kriz/cifar.html
    """
    def unpickle(self, file: str) -> Dict:
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def get_data(self, is_training: bool) -> Dict:
        if is_training:
            file_name = self._file_train
        else:
            file_name = self._file_test

        # Defines which dictionary keys to keep from the unpickled file
        dict_data = {
            b'fine_labels': [],
            b'coarse_labels': [],
            b'data': []
        }
        dict_data.update(self.unpickle(str(file_name)))

        return dict_data

    def get_num_classes(self) -> int:
        return self._num_classes


if __name__ == '__main__':
    dataset = CIFAR100()
    data = dataset.get_data(False)
    print(data.keys())
    print(data[b'coarse_labels'])
