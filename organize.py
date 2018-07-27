from pathlib import Path
from shutil import copyfile
from os import makedirs
import pandas as pd

from_path = Path('/l/vanvlm1/tiny-imagenet-200')
to_path = Path('/l/vanvlm1/tiny-imagenet')

# Training images
train = from_path / 'train'
for d in train.glob('n????????'):
    to_train_path = to_path / 'train' / d.name 
    makedirs(to_train_path, exist_ok=True)
    for image in (d / 'images').glob('*.JPEG'):
        print('%s -> %s' % (image, to_train_path / image.name))
        copyfile(image, to_train_path / image.name)

# Validation images
val = from_path / 'val'
df = pd.read_table(val / 'val_annotations.txt', usecols=[0, 1],
                   header=None, names=['file', 'wnid'])
df = df.set_index('file')
for image in (val / 'images').glob('*.JPEG'):
    wnid = df.at[image.name, 'wnid']
    to_val_path = to_path / 'val' / wnid
    makedirs(to_val_path, exist_ok=True)
    print('%s -> %s' % (image, to_val_path / image.name))
    copyfile(image, to_val_path / image.name)
