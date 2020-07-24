"""
Perform RSA between the model DSMs (computed with epasana_model_dsms.py) and
the MEG DSMs (computed with epasana_dsms.py).
"""
import mne_rsa
import mne
import numpy as np
import pickle

print('Loading Epasana DSMs...', end='', flush=True)
meg_data = np.load('../data/dsms/epasana-dsms.npz')
ch_names = meg_data['ch_names']
times = meg_data['times']
dsms_meg = meg_data['dsms']
print('done.')

model_name = 'vgg11_first_imagenet_then_epasana-1kwords_epasana-nontext_imagenet256'
with open(f'../data/dsms/epasana_{model_name}_dsms.pkl', 'rb') as f:
    model_data = pickle.load(f)
    dsms_model = model_data['dsms']
    dsms_model_names = model_data['dsm_names']

rsa_results = mne_rsa.rsa(
    list(dsms_meg.reshape(102 * 70, 172578)),
    dsms_model,
    metric='kendall-tau-a',
    verbose=True,
    n_jobs=4,
)

info = mne.io.read_info('info_102.fif')
info['sfreq'] = 100
evokeds = []
for data, name in zip(rsa_results.T, dsms_model_names):
    data = data.reshape(102, 70)
    evokeds.append(mne.EvokedArray(data, info, tmin=0.05, comment=name))

mne.write_evokeds(f'../data/epasana/epasana_rsa_{model_name}-ave.fif', evokeds)
