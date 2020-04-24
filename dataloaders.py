"""
PyTorch dataloaders to load the weird dataformat I'm using for my datasets. In
order to conserve memory, my datasets are pickled lists of PNG encoded images.
Reading these datasets involves using the Python Imaging Library (PIL) to
decode the PNG binary strings into PIL images.

The CombinedPickledPNGs dataloader will concatenate multiple datasets together.
This is useful is you want to train on for example both imagenet and word
datasets.
"""
from io import BytesIO
import os.path as op
import pickle

from torchvision.datasets import VisionDataset
from PIL import Image
import numpy as np


class PickledPNGs(VisionDataset):
    """Reads datasets in the form of pickled lists of PNG bytes.

    Args:
        root (string): Root directory of dataset
        train (bool, optional): If True, creates dataset from training set,
            otherwise creates from test set.
        transform (callable, optional): A function/transform that takes in an
            PIL image and returns a transformed version. E.g,
            ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        labels ('int' | 'vector'): What kind of labels to use. Either a integer
            class label or a distributed (word2vec) vector.
        label_offset (int): offset for 'int' style labels
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, labels='int', label_offset=0):
        super().__init__(root, transform=transform,
                         target_transform=target_transform)

        with open(op.join(root, 'train' if train else 'test'), 'rb') as f:
            dataset = pickle.load(f)
        with open(op.join(root, 'meta'), 'rb') as f:
            meta = pickle.load(f)

        self.data = dataset['data']
        if labels == 'int':
            self.targets = [l + label_offset for l in dataset['labels']]
            #self.vectors = meta['vectors']
        elif labels == 'vector':
            self.targets = [meta['vectors'][l] for l in dataset['labels']]
            self.vectors = meta['vectors']
        else:
            raise ValueError('`labels` should be either "int" or "vector"')

        self.classes = meta['label_names']
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.open(BytesIO(img))  # Decode the PNG bytes

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class CombinedPickledPNGs(VisionDataset):
    """Reads and combined multiple datasets in the form of pickles lists of PNG
    bytes.

    Args:
        roots (list of string): Root directorys of multiple datasets
        train (bool, optional): If True, creates dataset from training sets,
            otherwise creates from test sets.
        transform (callable, optional): A function/transform that takes in an
            PIL image and returns a transformed version. E.g,
            ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        labels ('int' | 'vector'): What kind of labels to use. Either a integer
            class label or a distributed (word2vec) vector.
    """
    def __init__(self, roots, train=True, transform=None,
                 target_transform=None, labels='int'):
        super().__init__(', '.join(roots), transform=transform,
                         target_transform=target_transform)

        self.data = []
        self.targets = []
        self.classes = []
        self.vectors = []
        target_offset = 0
        for root in roots:
            with open(op.join(root, 'train' if train else 'test'), 'rb') as f:
                dataset = pickle.load(f)
            with open(op.join(root, 'meta'), 'rb') as f:
                meta = pickle.load(f)

            self.data.extend(dataset['data'])
            if labels == 'int':
                self.targets.extend([l + target_offset for l in dataset['labels']])
            elif labels == 'vector':
                self.targets.extend([meta['vectors'][l] for l in dataset['labels']])
                self.vectors.extend(meta['vectors'])
            else:
                raise ValueError('`labels` should be either "int" or "vector"')
            self.classes.extend(meta['label_names'])
            self.class_to_idx = {name: i + target_offset for i, name in enumerate(self.classes)}

            target_offset = np.max(self.targets)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.open(BytesIO(img))  # Decode the PNG bytes

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
