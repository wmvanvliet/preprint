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
import tarfile
import pandas as pd
import struct

from torchvision.datasets import VisionDataset
from PIL import Image
import numpy as np
import example_pb2


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
        elif labels == 'vector':
            self.targets = np.array([meta['vectors'][l] for l in dataset['labels']], dtype=np.float32)
            self.vectors = np.array(meta['vectors'], dtype=np.float32)
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

class Tar(VisionDataset):
    """Reads datasets in the form of a tar file of image files.

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

        base_fname = 'train' if train else 'test'
        self.file = tarfile.open(op.join(root, f'{base_fname}.tar'))
        self.items = self.file.getmembers()
        self.meta = pd.read_csv(op.join(root, f'{base_fname}.csv'), index_col=0)
        self.vectors = np.atleast_2d(np.loadtxt(op.join(root, 'vectors.csv'), delimiter=',', skiprows=1, usecols=np.arange(1, 301), encoding='utf8', comments=None))

        if labels == 'int':
            self.targets = [l + label_offset for l in self.meta['label']]
        elif labels == 'vector':
            self.targets = np.array([self.vectors[l] for l in self.meta['label']], dtype=np.float32)
        else:
            raise ValueError('`labels` should be either "int" or "vector"')

        self.classes = self.meta.groupby('label').agg('first')['text']
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = Image.open(self.file.extractfile(self.items[index]))
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.items)

class TFRecord(VisionDataset):
    """Reads datasets in the form of tfrecord

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

        self.labels = labels
        base_fname = 'train' if train else 'test'
        self.file = open(op.join(root, f'{base_fname}.tfrecord'), 'rb')
        self.file_index = np.loadtxt(op.join(root, f'{base_fname}.index'), dtype=np.int64)[:, 0]
        self.meta = pd.read_csv(op.join(root, f'{base_fname}.csv'), index_col=0)
        self.vectors = np.atleast_2d(np.loadtxt(op.join(root, 'vectors.csv'), delimiter=',', skiprows=1, usecols=np.arange(1, 301), encoding='utf8', dtype=np.float32, comments=None))
        self.classes = self.meta.groupby('label').agg('first')['text']
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        self.file.seek(self.file_index[index])

        length_bytes = bytearray(8)
        crc_bytes = bytearray(4)
        datum_bytes = bytearray(1024 * 1024)

        if self.file.readinto(length_bytes) != 8:
            raise RuntimeError("Failed to read the record size.")
        if self.file.readinto(crc_bytes) != 4:
            raise RuntimeError("Failed to read the start token.")
        length, = struct.unpack("<Q", length_bytes)
        if length > len(datum_bytes):
            datum_bytes = datum_bytes.zfill(int(length * 1.5))
        datum_bytes_view = memoryview(datum_bytes)[:length]
        if self.file.readinto(datum_bytes_view) != length:
            raise RuntimeError("Failed to read the record.")
        if self.file.readinto(crc_bytes) != 4:
            raise RuntimeError("Failed to read the end token.")

        example = example_pb2.Example()
        example.ParseFromString(datum_bytes_view)

        features = {}
        for key in ['image/encoded', 'image/class/label']:
            field = example.features.feature[key].ListFields()[0]
            inferred_typename, value = field[0].name, field[1].value

            # Decode raw bytes into respective data types
            if inferred_typename == "bytes_list":
                value = np.frombuffer(value[0], dtype=np.uint8)
            elif inferred_typename == "float_list":
                value = np.array(value, dtype=np.float32)
            elif inferred_typename == "int64_list":
                value = np.array(value, dtype=np.int32)
            features[key] = value

        img = Image.open(BytesIO(features['image/encoded'])).convert('RGB')
        target = int(features['image/class/label'][0])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.labels == 'vector':
            target = self.vectors[target]

        return img, target

    def __len__(self):
        return len(self.file_index)
