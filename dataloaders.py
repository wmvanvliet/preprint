from io import BytesIO
import os.path as op
import pickle

from PIL import Image
import pandas as pd
from torchvision.datasets.vision import VisionDataset


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
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None):
        super().__init__(root, transform=transform,
                         target_transform=target_transform)

        with open(op.join(root, 'train' if train else 'test'), 'rb') as f:
            dataset = pickle.load(f)
            self.data = dataset['data']
            self.targets = dataset['labels']

        with open(op.join(root, 'meta'), 'rb') as f:
            meta = pickle.load(f)
            self.classes = meta['label_names']
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

        self.metadata = pd.read_csv(
            op.join(root, 'train.csv' if train else 'test.csv'), index_col=0)


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
