import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import mne
import numpy as np
import os
from os.path import getsize
import pickle
from matplotlib import pyplot as plt
import rsa
import editdistance
import mkl
mkl.set_num_threads(4)

os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'

import networks

model_name = 'vgg_first_imagenet64_then_tiny-words-noisy_tiny-imagenet'

epochs = mne.read_epochs('data/pilot_data/pilot2/pilot2_epo.fif', preload=False)
epochs = epochs[['word', 'symbols', 'consonants']]
stimuli = epochs.metadata.groupby('text').agg('first').sort_values('event_id')
stimuli['y'] = np.arange(len(stimuli))

with open('data/datasets/tiny-words/meta', 'rb') as f:
    meta = pickle.load(f)
labels = meta['label_names'].reset_index()
labels.columns = ['class_index', 'text']
labels = labels.set_index('text')
stimuli = stimuli.join(labels)
order = np.argsort(stimuli[:180]['class_index'])
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

filesizes = []
images = []
for fname in tqdm(stimuli['filename'], desc='Reading images'):
    filesizes.append(getsize(f'data/pilot_data/pilot2/stimuli/{fname}'))
    with Image.open(f'data/pilot_data/pilot2/stimuli/{fname}') as image:
        image = image.convert('RGB')
        image = preproc(image).unsqueeze(0)
        #image = transforms.ToTensor()(image)
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        images.append(image)
images = torch.cat(images, 0)

stimuli['visual_complexity'] = filesizes

checkpoint = torch.load('data/models/%s.pth.tar' % model_name, map_location='cpu')
num_classes = checkpoint['state_dict']['classifier.6.weight'].shape[0]
classifier_size = checkpoint['state_dict']['classifier.6.weight'].shape[1]
model = networks.vgg(num_classes=num_classes, classifier_size=classifier_size)
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

def words_only(x, y):
    if (x != 'word' or y != 'word') and (x != y):
        return 1
    else:
        return 0

def letters_only(x, y):
    if (x == 'symbols' or y == 'symbols') and (x != y):
        return 1
    else:
        return 0

def str_not_equal(x, y):
    if x == y:
        return 0
    else:
        return 1

def str_dist(x, y):
    return editdistance.eval(x[0], y[0])

print('Computing DSMs...', end='', flush=True)
dsms_network = [
    rsa.compute_dsm(feature_outputs[0], metric='correlation'),
    rsa.compute_dsm(feature_outputs[1], metric='correlation'),
    rsa.compute_dsm(feature_outputs[2], metric='correlation'),
    rsa.compute_dsm(feature_outputs[3], metric='correlation'),
    rsa.compute_dsm(classifier_outputs[0], metric='correlation'),
    rsa.compute_dsm(classifier_outputs[1], metric='correlation'),
    rsa.compute_dsm(classifier_outputs[2], metric='correlation'),
]
dsms_model = [
    rsa.compute_dsm(stimuli[['type']], metric=words_only),
    rsa.compute_dsm(stimuli[['type']], metric=letters_only),
    rsa.compute_dsm(stimuli[['noise_level']], metric='sqeuclidean'),
    rsa.compute_dsm(stimuli[['visual_complexity']], metric='sqeuclidean'),
    rsa.compute_dsm(stimuli[['font']], metric=str_not_equal),
    rsa.compute_dsm(stimuli[['rotation']], metric='sqeuclidean'),
    rsa.compute_dsm(stimuli[['fontsize']], metric='sqeuclidean'),
    rsa.compute_dsm(stimuli.index.tolist(), metric=str_dist),
    rsa.compute_dsm(images.numpy().reshape(len(images), -1), metric='sqeuclidean'),
    rsa.compute_dsm(images.numpy().reshape(len(images), -1).sum(axis=1), metric='sqeuclidean'),
]
dsms_names = ['Words only', 'Letters only', 'Noise level', 'Visual complexity', 'Font', 'Rotation', 'Fontsize', 'Edit distance', 'Pixel distance', 'Pixel count']

rsa_results = rsa.rsa(dsms_model, dsms_network, metric='kendall-tau-a')
n_models, n_layers = rsa_results.shape

f = plt.figure(figsize=(8, 5))
axs = f.subplots(int(np.ceil(n_models / 5)), 5, sharex=True, sharey=True)
for i, (name, result) in enumerate(zip(dsms_names, rsa_results)):
    ax = axs[i // 5, i % 5]
    ax.bar(np.arange(n_layers), result)
    ax.axhline(0, color='black')
    ax.set_title(name)
    if (i // 5) == len(axs) - 1:
        ax.set_xlabel('Network layer')
        ax.set_xticks(np.arange(n_layers))
        ax.set_xticklabels(['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2', 'output'], rotation=90)
plt.tight_layout()
