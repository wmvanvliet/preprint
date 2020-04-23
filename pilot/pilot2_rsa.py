import rsa
import mne
import numpy as np
import pickle
from tqdm import tqdm

#model_name = 'n400'
#model_name = 'vgg_first_imagenet64_then_tiny-words_tiny-consonants_w2v'
#model_name = 'vgg_first_imagenet64_then_tiny-words-noisy_tiny-imagenet'
#model_name = 'vgg_first_imagenet64_then_tiny-words_tiny-imagenet_then_w2v'
model_name = 'vgg_first_imagenet64_then_tiny-words_tiny-imagenet'
with open(f'../data/dsms/pilot2_{model_name}_dsms.pkl', 'rb') as f:
    dsm_models = pickle.load(f)
    dsms = dsm_models['dsms']
    dsm_names = dsm_models['dsm_names']

def load_meg():
    dsms_meg = np.load(f'../data/pilot_data/pilot2/pilot2_dsms_{model_name}.npy')
    for dsm in tqdm(dsms_meg):
        yield dsm

rsa_results = rsa.rsa(
    load_meg(),
    dsms,
    metric='kendall-tau-a',
)

info = mne.io.read_info('../template_info.fif')
rsa_evokeds = []
for r, name in zip(rsa_results, dsm_names):
    ev = mne.EvokedArray(r, info, tmin=0)
    ev.comment = name
    rsa_evokeds.append(ev)

mne.write_evokeds(f'../data/pilot_data/pilot2/pilot2_rsa_{model_name}-ave.fif', rsa_evokeds)
