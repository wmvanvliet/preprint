import mne
import numpy as np
import sys
from matplotlib import pyplot as plt

model_name = sys.argv[1]

info = mne.io.read_info('template_info.fif')
tmin = -0.15

def load_model(model_name):
    evokeds = []
    #m = np.load('models/rsa_%s.npy' % model_name)
    m = np.load('rsa_%s.npy' % model_name)
    for layer, comment in zip([0, 1, 2, 3], ['conv 1', 'conv 2', 'conv 3', 'conv 4']):
        evokeds.append(mne.EvokedArray(m[layer], info, tmin, comment))
    return evokeds

#rsa_redness1 = load_model('rsa_vgg_redness1')
#rsa_tiny_imagenet = load_model('rsa_vgg_tiny_imagenet')
#rsa_first_images_then_redness1 = load_model('rsa_first_images_then_redness1')
#rsa_tiny_redness1_image = load_model('rsa_vgg_tiny_redness1_image')
rsa_model = load_model(model_name)

#fig = mne.viz.plot_evoked_topo(rsa_tiny_redness1_image, layout_scale=1)
#fig.set_size_inches(14, 12, forward=True)

# selections = [
#     #'Left-temporal',
#     #'Right-temporal',
#     'Left-parietal',
#     #'Right-parietal',
#     'Left-occipital',
#     #'Right-occipital',
#     #'Left-frontal',
#     #'Right-frontal'
# ]
selections = ['Occipital', 'Left-occipital', 'Left-temporal']
ch_names = [ch['ch_name'] for ch in info['chs']]

for r in rsa_model:
    r.info['projs'] = []
    r.set_channel_types({ch: 'misc' for ch in r.ch_names})
    r.shift_time(-0.045)

fig, axes = plt.subplots(nrows=1, ncols=3)
fig.set_size_inches(12, 2.5, forward=True)
for sel, ax in zip(selections, axes): #[a for sublist in axes for a in sublist]):
    picks = np.intersect1d(ch_names, mne.selection.read_selection(sel, './roi.sel', info))
    mne.viz.plot_compare_evokeds(
        {r.comment: r.copy().crop(0, 0.6) for r in rsa_model},
        picks=picks, combine='mean', title=sel, show_sensors=True, axes=ax, ylim=dict(misc=[-0.05, 0.08]))
#plt.tight_layout()
