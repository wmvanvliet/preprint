"""
Construct a dataset containing 128x128 pixel images of random symbol strings.  Uses the
11 symbols used in the "epasana" study.
"""
# encoding: utf-8
import argparse
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib import font_manager as fm
import pandas as pd
from os import makedirs
from tqdm import tqdm
import pickle
from PIL import Image
from io import BytesIO
import random
import seaborn as sb
import matplotlib.pyplot as plt

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate epasana-symbols dataset')
parser.add_argument('path', type=str, help='The path to write the dataset to.')
parser.add_argument('set', type=str, help='Specify either "train" to generate the training set and to "test" to generate the test set.')
args = parser.parse_args()

# Limits
noise_levels = [0.2, 0.35, 0.5]
lengths = [7,8]
rotations = np.linspace(-20, 20, 11)

# Loading array of symbol bitmaps from pickle object. The pickle object is generated with symbol_bitmaps.py
with open('symbol-bitmaps', 'rb') as f:
    symbol_bitmaps = pickle.load(f)
          
rng = np.random.RandomState(0)

chosen_rotations = []
chosen_noise_levels = []
chosen_lengths = []

n = 100_000 if args.set == 'train' else 10_000

data = []
labels = np.zeros(n, dtype=np.int)

dpi = 96.
f = Figure(figsize=(128 / dpi, 128 / dpi), dpi=dpi)
canvas = FigureCanvasAgg(f)


for i in tqdm(range(n), total=n):

    f.clf()
    ax = f.add_axes([0, 0, 1, 1])
    rotation = rng.choice(rotations)
    noise_level = rng.choice(noise_levels)
    length = rng.choice(lengths)

    background_noise = rng.rand(128, 128)
 
    ax.imshow(background_noise, extent=[0, 1, 0, 1], cmap='gray', alpha=noise_level)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # Creating symbol image as numpy array
    chosen_symbols = []
    for a in range(length):
        chosen_symbols.append(random.choice(symbol_bitmaps))

    img_array = np.concatenate(chosen_symbols,axis=1)

    # Separating black pixels in symbols from black pixels that append when rotating symbol image
    img_array = img_array + 1

    # Creating image object from the numpy array
    img = Image.fromarray(img_array.astype(np.uint8))
    
    # Rotating the image
    img = img.rotate(rotation, expand='True')

    # Converting image back to numpy array
    rotated = np.asarray(img,dtype='float64').copy()

    # Changing black background pixels to white
    rotated[rotated == 0] = 106

    # Returning other pixels to their original values
    rotated = rotated - 1

    # Constructing subplot noise array
    rand_noise = rng.rand(rotated.shape[0],rotated.shape[1])

    # Making a sublot within the excisting figure
    ax1 = f.add_subplot(111)
    ax1.axis('off')
    ax1.imshow(rotated,cmap = 'gray')

    # Applying noise on top of the symbol image
    ax3 = f.add_subplot(111)
    ax3.axis('off')
    ax3.imshow(rand_noise,cmap = 'gray',alpha=noise_level)

    #f.savefig('training_imgs/symbols%03d.JPEG' % i)

    canvas.draw()
    buffer, (width, height) = canvas.print_to_buffer()
    image = np.frombuffer(buffer, np.uint8).reshape((height, width, 4))[:, :, :3]

    chosen_rotations.append(rotation)
    chosen_noise_levels.append(noise_level)
    chosen_lengths.append(length)

    buf = BytesIO()
    Image.fromarray(image.astype(np.uint8)).save(buf, format='png')
    img_compressed = buf.getvalue()
    data.append(img_compressed)


df = pd.DataFrame(dict(rotation=chosen_rotations,
                       label=labels,
                       noise_level=chosen_noise_levels,lenght=chosen_lengths))

makedirs(args.path, exist_ok=True)

df.to_csv(f'{args.path}/{args.set}.csv')
with open(f'{args.path}/{args.set}', 'wb') as f:
    pickle.dump(dict(data=data, labels=labels), f)

#with open(f'{args.path}/meta', 'wb') as f:
#    pickle.dump(dict(label_names=['symbols'], vectors=np.zeros((1, 300))), f)
