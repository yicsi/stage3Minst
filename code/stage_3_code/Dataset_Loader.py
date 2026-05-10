'''
Dataset Loader for Stage 3 image datasets
'''

from code.base_class.dataset import dataset
import pickle
import numpy as np


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None

    def __init__(self, dataset_name, dDescription=None):
        dataset.__init__(self, dataset_name, dDescription)

    def load(self):
        print('loading data...')

        if self.dataset_name.lower() == 'mnist':
            data_file = self.dataset_source_folder_path + 'MNIST'
        elif self.dataset_name.lower() == 'orl':
            data_file = self.dataset_source_folder_path + 'ORL'
        elif self.dataset_name.lower() == 'cifar':
            data_file = self.dataset_source_folder_path + 'CIFAR'
        else:
            raise ValueError('Unknown dataset name: ' + self.dataset_name)

        with open(data_file, 'rb') as f:
            raw_data = pickle.load(f)

        train_X = []
        train_y = []
        test_X = []
        test_y = []

        for item in raw_data['train']:
            image = item['image']
            label = item['label']

            if self.dataset_name.lower() == 'mnist':
                image = np.array(image).reshape(-1) / 255.0

            elif self.dataset_name.lower() == 'orl':
                image = np.array(image)[:, :, 0].reshape(-1) / 255.0
                label = label - 1

            elif self.dataset_name.lower() == 'cifar':
                image = np.array(image).reshape(-1) / 255.0

            train_X.append(image)
            train_y.append(label)

        for item in raw_data['test']:
            image = item['image']
            label = item['label']

            if self.dataset_name.lower() == 'mnist':
                image = np.array(image).reshape(-1) / 255.0

            elif self.dataset_name.lower() == 'orl':
                image = np.array(image)[:, :, 0].reshape(-1) / 255.0
                label = label - 1

            elif self.dataset_name.lower() == 'cifar':
                image = np.array(image).reshape(-1) / 255.0

            test_X.append(image)
            test_y.append(label)

        return {
            'train': {
                'X': train_X,
                'y': train_y
            },
            'test': {
                'X': test_X,
                'y': test_y
            }
        }
