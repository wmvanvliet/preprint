import mne
import numpy as np
from matplotlib import pyplot as plt

info = mne.io.read_info('template_info.fif')
tmin = -0.15

def load_model(model_name):
    evokeds = []
    m = np.load('models/%s.npy' % model_name)
    for layer in range(4):
        evokeds.append(mne.EvokedArray(m[layer], info, tmin, 'layer %d' % (layer + 1)))
    return evokeds

rsa_redness1 = load_model('rsa_vgg_redness1')
rsa_first_images_then_redness1 = load_model('rsa_first_images_then_redness1')
rsa_tiny_redness1_image = load_model('rsa_vgg_tiny_redness1_image')

#fig = mne.viz.plot_evoked_topo(rsa_tiny_redness1_image, layout_scale=1)
#fig.set_size_inches(14, 12, forward=True)

selections = [
    'Left-temporal',
    'Right-temporal',
    'Left-parietal',
    'Right-parietal',
    'Left-occipital',
    'Right-occipital',
    #'Left-frontal',
    #'Right-frontal'
]
ch_names = [ch['ch_name'] for ch in info['chs']]

fig, axes = plt.subplots(nrows=len(selections) // 2, ncols=2)
fig.set_size_inches(10, 10, forward=True)
for sel, ax in zip(selections, [a for sublist in axes for a in sublist]):
    picks = np.intersect1d(ch_names, mne.selection.read_selection(sel, None, info))
    mne.viz.plot_compare_evokeds(
        #dict(redness1=rsa_redness1[3], tiny_redness1_images=rsa_tiny_redness1_image[3]),
        {r.comment: r.copy().crop(0, 0.6) for r in rsa_tiny_redness1_image},
        picks=picks, combine='mean', title=sel, show_sensors=True, axes=ax, ylim=dict(grad=[-5E11, 5E11]))
plt.tight_layout()
