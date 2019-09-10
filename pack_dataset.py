import argparse
import h5py
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Pack image folder into HDF5 file')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('output', metavar='filename',
                    help='File to write to. Should end in .h5')
args = parser.parse_args()

dataset = ImageFolder(args.data, transform=ToTensor())
dataloader = DataLoader(dataset, batch_size=256)

f = h5py.File(args.output, 'w')

print('Reading from ', args.data)
img_shape = dataset[0][0].shape
X = np.zeros((len(dataset),) + img_shape)
Y = np.zeros(len(dataset), dtype=np.int)
for i, (img, label) in tqdm(enumerate(dataset), total=len(dataset), unit='img'):
   X[i] = img 
   Y[i] = label

print('Writing to', args.output)
f.create_dataset("images", data=X)
f.create_dataset("labels", data=Y)
f.close()
