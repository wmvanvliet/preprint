import h5py
import numpy as np
from torch import Tensor
from torch.utils.data.dataset import Dataset 


class H5File(Dataset):
    def __init__(self, fname, transform=None, target_transform=None):
        self.f = h5py.File(fname, 'r')
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.f['images'].shape[0]

    def __getitem__(self, i):
        img, label = self.f['images'][i], self.f['labels'][i]
        img = Tensor(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __del__(self):
        if hasattr(self, 'f'):
            self.f.close()

    @property
    def classes(self):
        return np.unique(self.f['labels'])
