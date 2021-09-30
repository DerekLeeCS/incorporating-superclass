from pathlib import Path
from typing import Dict
import pickle

from datasetInterface import Dataset


class CIFAR100(Dataset):
    _fileTrain = Path(__file__, '../cifar-100-python/train')
    _fileTest = Path(__file__, '../cifar-100-python/test')
    _numClasses = 100

    """This function processes an uncompressed file from the CIFAR100 dataset into a dictionary.
    The dataset and the function is from:
    https://www.cs.toronto.edu/~kriz/cifar.html
    """
    def unpickle(self, file: str) -> Dict:
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def getData(self, isTraining: bool) -> Dict:
        if isTraining:
            fileName = self._fileTrain
        else:
            fileName = self._fileTest

        # Defines which dictionary keys to keep from the unpickled file
        dictData = {
            b'fine_labels': [],
            b'coarse_labels': [],
            b'data': []
        }
        dictData.update(self.unpickle(str(fileName)))

        return dictData

    def getNumClasses(self) -> int:
        return self._numClasses


if __name__ == '__main__':
    dataset = CIFAR100()
    data = dataset.getData(False)
    print(data.keys())
    print(data[b'coarse_labels'])
