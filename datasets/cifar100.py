from typing import Dict
import pickle
from datasetInterface import Dataset


class CIFAR100(Dataset):
    fileTrain = 'cifar-100-python/train'
    fileTest = 'cifar-100-python/test'

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
            fileName = self.fileTrain
        else:
            fileName = self.fileTest

        # Defines which dictionary keys to keep from the unpickled file
        dictData = {
            b'fine_labels': [],
            b'coarse_labels': [],
            b'data': []
        }

        dictData.update(self.unpickle(fileName))
        print(dictData.keys())
        print(dictData[b'coarse_labels'])

        return dictData
