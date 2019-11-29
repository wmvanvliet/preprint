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
epochs = mne.read_epochs('/l/vanvlm1/redness2/all-epo.fif', preload=False)
epochs = epochs['written']
epochs.load_data()
epochs.shift_time(-0.044)  # Compensate for projector delay
print('done.')

stimuli = epochs.metadata.query("modality=='wri'").groupby('label').aggregate('first')
#stimuli = epochs.metadata.query("modality=='pic'").groupby('label').aggregate('first')
#stimuli = epochs.metadata.query("modality=='wri' or modality=='pic'").groupby('label').aggregate('first')

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


# Read in the word2vec vectors
m = loadmat('/m/nbe/scratch/redness2/semantic_features/Ginter-300-5+5.mat')
word2vec_words = [w[0][0] for w in m['sorted']['wordsNoscand'][0, 0]]
#order = stimuli.index.get_indexer_for(word2vec_words)
word2vec = m['sorted']['mat'][0, 0]
sel = [word2vec_words.index(w) for w in stimuli.Noscand if w in word2vec_words]
word2vec = word2vec[sel]
word2vec_words = [word2vec_words[s] for s in sel]

print('Computing model DSMs...', end='', flush=True)
dsm_models = [
    rsa.compute_dsm(feature_outputs[0], metric='correlation'),
    rsa.compute_dsm(feature_outputs[1], metric='correlation'),
    rsa.compute_dsm(feature_outputs[2], metric='correlation'),
    rsa.compute_dsm(feature_outputs[3], metric='correlation'),
    rsa.compute_dsm(classifier_outputs[0], metric='correlation'),
    rsa.compute_dsm(classifier_outputs[1], metric='correlation'),
    rsa.compute_dsm(classifier_outputs[2], metric='correlation'),
    rsa.compute_dsm(pixels, metric='euclidean'),
    #rsa.compute_dsm(stimuli['letters'].values, metric='euclidean'),
    rsa.compute_dsm(word2vec, metric='cosine'),
    #np.ones(44850),
]
print('done.')

print('Starting RSA compution', flush=True)
rsa_epochs = rsa.rsa_epochs(
    epochs,
    dsm_models,
    y=epochs.metadata.label,
    spatial_radius=0.001,
    temporal_radius=0.05,
    epochs_dsm_metric='sqeuclidean',
    rsa_metric='spearman',
    n_folds=5,
    n_jobs=2,
    verbose=True,
)
for e, comment in zip(rsa_epochs, ['feature layer 1', 'feature layer 2', 'feature layer 3', 'feature layer 4', 'classifier layer 1', 'classifier layer 2', 'classifier output', 'pixels', 'word2vec']):
    e.comment = comment

#np.save('models/rsa_%s.npy' % model_name, [e._data for e in rsa_epochs])
np.save('rsa_%s.npy' % model_name, [e._data for e in rsa_epochs])
mne.write_evokeds('rsa_%s-ave.fif' % model_name, rsa_epochs)

fig = mne.viz.plot_evoked_topo(rsa_epochs, layout_scale=1)
fig.set_size_inches(14, 12, forward=True)
plt.savefig('figures/%s.pdf' % model_name)

def pack_data(data, comment):
    return mne.EvokedArray(data, info=evoked_template.info, tmin=evoked_template.times[4], comment=comment)

