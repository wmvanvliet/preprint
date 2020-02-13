import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm
import rsa
import mne
import numpy as np
import sys
import os
import pickle
from matplotlib import pyplot as plt
import mkl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib import font_manager as fm
mkl.set_num_threads(4)

os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'

import networks

model_name = 'vgg_first_imagenet64_then_tiny-words_tiny-imagenet'

epochs = mne.read_epochs('data/pilot_data/pilot2/pilot2_epo.fif', preload=False)
epochs = epochs[['word', 'symbols', 'consonants']]
stimuli = epochs.metadata.groupby('text').agg('first').sort_values('event_id')
stimuli['y'] = np.arange(len(stimuli))
metadata = pd.merge(epochs.metadata, stimuli[['y']], left_on='text', right_index=True).sort_index()
assert np.array_equal(metadata.event_id.values.astype(int), epochs.events[:, 2])

with open('data/datasets/tiny-words/meta', 'rb') as f:
    meta = pickle.load(f)
labels = meta['label_names'].reset_index()
labels = labels.set_index('text')
metadata = metadata.join(labels)
order = np.argsort(stimuli[:180, :200]['class_index'])
order = np.hstack([order, np.arange(180, 360)])

preproc = transforms.Compose([
    transforms.CenterCrop(208),
    transforms.Resize(64),
    transforms.CenterCrop(60),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

unnormalize = transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                   std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

images = []
for fname in tqdm(stimuli['filename'], desc='Reading images'):
    with Image.open(f'data/pilot_data/pilot2/stimuli/{fname}') as image:
        image = image.convert('RGB')
        image = preproc(image).unsqueeze(0)
        #image = transforms.ToTensor()(image)
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        images.append(image)
images = torch.cat(images, 0)

plt.figure(figsize=(10, 10))
plt.imshow(make_grid(images/5 + 0.5, nrow=20).numpy().transpose(1, 2, 0))
plt.axis('off')
plt.tight_layout()

checkpoint = torch.load('data/models/%s.pth.tar' % model_name, map_location='cpu')
num_classes = checkpoint['state_dict']['classifier.6.weight'].shape[0]
classifier_size = checkpoint['state_dict']['classifier.6.weight'].shape[1]
#num_classes = checkpoint['state_dict']['classifier.weight'].shape[0]
model = networks.vgg(num_classes=num_classes, classifier_size=classifier_size)
#model = networks.TwoLayerNet(num_classes=num_classes)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

feature_outputs = []
out = images
for i, layer in enumerate(model.features):
    layer_out = []
    for j in range(0, len(out), 128):
        layer_out.append(layer(out[j:j + 128]))
    out = torch.cat(layer_out, 0)
    del layer_out
    print('layer %02d, output=%s' % (i, out.shape))
    if i in [4, 10, 17, 24]:
        feature_outputs.append(out.detach().numpy().copy())
classifier_outputs = []
out = out.view(out.size(0), -1)
layer_out = []
for i, layer in enumerate(model.classifier):
    layer_out = []
    for j in range(0, len(out), 128):
        layer_out.append(layer(out[j:j + 128]))
    out = torch.cat(layer_out, 0)
    print('layer %02d, output=%s' % (i, out.shape))
    if i in [1, 4, 6]:
        classifier_outputs.append(out.detach().numpy().copy())

plt.figure()
plt.imshow(classifier_outputs[-1][order])
plt.axhline(180, color='black')
plt.axhline(180 + 90, color='black')
plt.axvline(200, color='black')

outputs = np.argmax(classifier_outputs[-1][order], axis=1)
test = np.zeros((360, 400))
test[(range(360), outputs)] = 1
plt.figure()
plt.imshow(test)
plt.axhline(180, color='black')
plt.axhline(180 + 90, color='black')
plt.axvline(200, color='black')
