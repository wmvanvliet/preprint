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

stimuli = pd.read_csv('/l/vanvlm1/redness1/stimuli.csv', index_col=0)

images = []
for word in stimuli.index:
    image = Image.open('/l/vanvlm1/redness1/images/%s.JPEG' % word)
    images.append(2.64 - preproc(image).unsqueeze(0))
images = torch.cat(images, 0)
pixels = images.sum(dim=(1,2,3))[:, None].numpy()

checkpoint = torch.load('models/%s.pth.tar' % model_name)
num_classes = checkpoint['state_dict']['classifier.6.weight'].shape[0]
model = networks.vgg(num_classes=num_classes)
#model = networks.TwoLayerNet(num_classes=200)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

feature_outputs = []
out = images
for i, layer in enumerate(model.features):
    out = layer(out)
    print('Layer %02d, output=%s' % (i, out.shape))
    feature_outputs.append(out.detach().numpy().copy())
classifier_outputs = []
out = out.view(out.size(0), -1)
for i, layer in enumerate(model.classifier):
    out = layer(out)
    print('Layer %02d, output=%s' % (i, out.shape))
    classifier_outputs.append(out.detach().numpy().copy())
#classifier_outputs.append(model.classifier(out).detach().numpy().copy())

evoked_template = mne.read_evokeds('/l/vanvlm1/redness1/ga-ave.fif', 0)
locs = np.vstack([ch['loc'][:3] for ch in evoked_template.info['chs']])
dist = distance.squareform(distance.pdist(locs))
evokeds = np.load('/l/vanvlm1/redness1/all_evokeds.npy')
labels = np.load('/l/vanvlm1/redness1/all_evokeds_labels.npy')
word2vec = np.load('/l/vanvlm1/redness1/word2vec.npy')

def compute_rsa(layer_output, comment):
    dsm_model = rsa.compute_dsm(layer_output, pca=False, metric='correlation')
    rsa_layer = rsa.rsa_spattemp(
        evokeds,
        dsm_model,
        dist,
        spatial_radius=0.001,
        temporal_radius=5,
        y=labels,
        data_dsm_metric='sqeuclidean',
        verbose=True
    )
    return mne.EvokedArray(rsa_layer, info=evoked_template.info, tmin=evoked_template.times[4], comment=comment)

def pack_data(data, comment):
    return mne.EvokedArray(data, info=evoked_template.info, tmin=evoked_template.times[4], comment=comment)

dsm_models = [
    rsa.compute_dsm(feature_outputs[6], metric='correlation'),
    rsa.compute_dsm(feature_outputs[13], metric='correlation'),
    rsa.compute_dsm(feature_outputs[20], metric='correlation'),
    rsa.compute_dsm(feature_outputs[27], metric='correlation'),
    rsa.compute_dsm(classifier_outputs[1], metric='correlation'),
    rsa.compute_dsm(classifier_outputs[4], metric='correlation'),
    rsa.compute_dsm(classifier_outputs[6], metric='correlation'),
    rsa.compute_dsm(pixels, metric='euclidean'),
    rsa.compute_dsm(stimuli['letters'].values, metric='euclidean'),
    rsa.compute_dsm(word2vec, metric='cosine'),
]
#dsm_models = [
#    rsa.compute_dsm(feature_outputs[3], metric='correlation'),
#    rsa.compute_dsm(feature_outputs[7], metric='correlation'),
#    rsa.compute_dsm(classifier_outputs[0], metric='correlation'),
#]
rsa_evokeds = rsa.rsa_spattemp(
    evokeds,
    dsm_models,
    dist,
    spatial_radius=0.001,
    temporal_radius=5,
    y=labels,
    data_dsm_metric='sqeuclidean',
    #data_dsm_metric='correlation',
    n_jobs=4,
    verbose=True,
)
rsa_evokeds = rsa_evokeds.transpose(2, 0, 1)
rsa_evokeds = [
    mne.EvokedArray(ev, info=evoked_template.info, tmin=evoked_template.times[4], comment=c)
    for ev, c in zip(rsa_evokeds, ['layer1', 'layer2', 'layer3', 'layer4', 'classifier1', 'classifier2', 'classifier3', 'pixels', 'letters', 'word2vec'])
    #for ev, c in zip(rsa_evokeds, ['layer1', 'layer2', 'classifier1'])
]

np.save('models/rsa_%s.npy' % model_name, rsa_evokeds)

fig = mne.viz.plot_evoked_topo(rsa_evokeds, layout_scale=1)
fig.set_size_inches(14, 12, forward=True)
plt.savefig('figures/%s.pdf' % model_name)
