import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from scipy.spatial import distance
from scipy.stats import zscore
import rsa
import mne
import numpy as np
import sys
from matplotlib import pyplot as plt

import networks

model_name = sys.argv[1]

preproc = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(60),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

epochs = mne.read_epochs('/l/vanvlm1/redness2/all-epo.fif')

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

checkpoint = torch.load('models/%s.pth.tar' % model_name)
num_classes = checkpoint['state_dict']['classifier.6.weight'].shape[0]
model = networks.vgg(num_classes=num_classes)
#model = networks.TwoLayerNet(num_classes=200)
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
    #if i in [6, 13, 20, 27]:
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

word2vec = np.load('/l/vanvlm1/redness2/word2vec.npy')

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
    #rsa.compute_dsm(word2vec, metric='cosine'),
]

epochs = epochs['written']
rsa_epochs = rsa.rsa_epochs(
    epochs,
    dsm_models,
    y=epochs.metadata.label,
    spatial_radius=0.001,
    temporal_radius=0.05,
    epochs_dsm_metric='sqeuclidean',
    n_folds=5,
    n_jobs=4,
    verbose=True,
)
for e, comment in zip(rsa_epochs, ['feature layer 1', 'feature layer 2', 'feature layer 3', 'feature layer 4', 'classifier layer 1', 'classifier layer 2', 'classifier output', 'pixels']):
    e.comment = comment

np.save('models/rsa_%s.npy' % model_name, [e._data for e in rsa_epochs])

fig = mne.viz.plot_evoked_topo(rsa_epochs, layout_scale=1)
fig.set_size_inches(14, 12, forward=True)
plt.savefig('figures/%s.pdf' % model_name)

def pack_data(data, comment):
    return mne.EvokedArray(data, info=evoked_template.info, tmin=evoked_template.times[4], comment=comment)
