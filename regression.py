import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from scipy.io import loadmat
import rsa
import mne
import numpy as np
import sys
import os
from matplotlib import pyplot as plt
import mkl
from sklearn import linear_model
mkl.set_num_threads(4)

os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'

import networks

model_name = sys.argv[1]

preproc = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(60),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

print('Reading epochs...', end='', flush=True)
epochs = mne.read_epochs('/l/vanvlm1/redness2/all-epo.fif')
epochs = epochs['written']
epochs.shift_time(-0.044)  # Compensate for projector delay
print('done.')
stimuli = epochs.metadata.groupby('label').aggregate('first')

images = []
for label in stimuli.index:
    image = Image.open('/l/vanvlm1/redness2/images/%s.JPEG' % label)
    image = 2.64 - preproc(image).unsqueeze(0)
    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1)
    images.append(image)
images = torch.cat(images, 0)
pixels = images.sum(dim=(1,2,3))[:, None].numpy()
#images = images.repeat((1, 3, 1, 1))

checkpoint = torch.load('models/%s.pth.tar' % model_name, map_location='cpu')
#checkpoint = torch.load('%s.pth.tar' % model_name)
num_classes = checkpoint['state_dict']['classifier.6.weight'].shape[0]
#num_classes = checkpoint['state_dict']['classifier.weight'].shape[0]
model = networks.vgg(num_classes=num_classes)
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
#classifier_outputs.append(model.classifier(out).detach().numpy().copy())

def pack_epochs(data):
    return mne.EpochsArray(data, info=epochs.info, tmin=epochs.times[0])

def pack_evoked(data, comment=''):
    return mne.EvokedArray(data, info=epochs.info, tmin=epochs.times[0], comment=comment)

Ws = []
brain = rsa.folds._create_folds(epochs.get_data(), epochs.metadata.label, 1)[0]
for i in [0, 1, 2, 3]:
    model = feature_outputs[i].sum(axis=(1, 2, 3))[:, None]
    m = linear_model.LinearRegression().fit(model, brain.reshape(len(brain), -1))
    W = pack_evoked(m.coef_.reshape(brain.shape[1:]), 'layer %d' % (i + 1))
    Ws.append(W)
