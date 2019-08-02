import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from scipy.spatial import distance
import rsa
import mne
import numpy as np
import sys

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

checkpoint = torch.load('models/%s.pth.tar' % model_name)
model = networks.vgg(num_classes=200)
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

def compute_rsa(layer_output, comment):
    dsm_model = rsa.compute_dsm(layer_output, metric='correlation')
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

rsa_evokeds = [
    compute_rsa(feature_outputs[6], 'layer1'),
    compute_rsa(feature_outputs[13], 'layer2'),
    compute_rsa(feature_outputs[20], 'layer3'),
    compute_rsa(feature_outputs[27], 'layer4'),
    compute_rsa(classifier_outputs[1], 'classifier1'),
    compute_rsa(classifier_outputs[4], 'classifier2'),
    compute_rsa(classifier_outputs[6], 'classifier3'),
]
#rsa_evokeds = [
#    compute_rsa(feature_outputs[3], 'layer1'),
#    compute_rsa(feature_outputs[7], 'layer2'),
#    compute_rsa(classifier_outputs[0], 'classifier'),
#]

np.save('models/rsa_%s.npy' % model_name, rsa_evokeds)

fig = mne.viz.plot_evoked_topo(rsa_evokeds, layout_scale=1)
fig.set_size_inches(14, 12, forward=True)
fig.save('figures/%s.pdf' % model_name)

#rsa_layer1 = rsa.rsa_evokeds(evokeds, feature_outputs[3].reshape(123, -1), evoked_dsm_metric='euclidean', spatial_radius=0.001, temporal_radius=0.05, n_jobs=4, verbose=True)
#rsa_layer2 = rsa.rsa_evokeds(evokeds, feature_outputs[7].reshape(123, -1), evoked_dsm_metric='euclidean', spatial_radius=0.001, temporal_radius=0.05, n_jobs=4, verbose=True)
#rsa_classifier = rsa.rsa_evokeds(evokeds, classifier_outputs[0].reshape(123, -1), evoked_dsm_metric='euclidean', spatial_radius=0.001, temporal_radius=0.05, n_jobs=4, verbose=True)
#
#rsa_layer1.comment = 'Layer 1'
#rsa_layer2.comment = 'Layer 2'
#rsa_classifier.comment = 'Classifier'
#
#fig = mne.viz.plot_evoked_topo([rsa_layer1, rsa_layer2, rsa_classifier], layout_scale=1)
#fig.set_size_inches(14, 12, forward=True)
