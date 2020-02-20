from io import BytesIO
import os.path as op
import pickle
import os
import torch
import torch.utils.data as data

from PIL import Image
import pandas as pd
import numpy as np


class VisionDataset(data.Dataset):
    _repr_indent = 4

    def __init__(self, root, transforms=None, transform=None, target_transform=None):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can "
                             "be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        return ""


class StandardTransform(object):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input, target):
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self):
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transform: ")

        return '\n'.join(body)


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
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, labels='int'):
        super().__init__(root, transform=transform,
                         target_transform=target_transform)

        with open(op.join(root, 'train' if train else 'test'), 'rb') as f:
            dataset = pickle.load(f)
        with open(op.join(root, 'meta'), 'rb') as f:
            meta = pickle.load(f)

        self.data = dataset['data']
        if labels == 'int':
            self.targets = dataset['labels']
        elif labels == 'vector':
            self.targets = [meta['vectors'][l] for l in dataset['labels']]
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
    """Reads and combined multiple datasets in the form of pickles lists of PNG bytes.

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
