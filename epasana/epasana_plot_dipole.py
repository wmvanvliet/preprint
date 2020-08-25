import mne
import mne_bids
from config import fname, n_jobs
from mayavi import mlab
from matplotlib import pyplot as plt
import numpy as np

subject = 1

epochs = mne.read_epochs(fname.epochs(subject=subject))
epochs.pick_types(meg='grad')
trans = mne_bids.get_head_mri_trans(fname.raw(subject=subject), fname.bids_root)
cov = mne.compute_covariance(epochs) # read_cov(fname.cov(subject=subject))
bem = mne.read_bem_solution(fname.bem(subject=subject))

##
#evoked = epochs['type=="noisy word"'].average()
#evoked.comment = 'noise'
#evoked = epochs[epochs.metadata.type.isin(['word', 'pseudoword', 'consonants'])].average()
#evoked.comment = 'letter'
evoked = epochs[epochs.metadata.type.isin(['word', 'pseudoword'])].average()
evoked.comment = 'word'
#time_roi = (0.27, 0.54)

# Read a dipole
dips = mne.read_dipole(fname.dip(subject=subject))

# Make a fancy plot showing the realtionship between the dipole and the field lines

##
sel_dip = 6
time = 0.2
fig = mlab.figure(size=(1000, 1000))
mne.viz.plot_alignment(
    evoked.info,
    trans,
    f'sub-{subject:02d}',
    fname.subjects_dir,
    surfaces=['pial', 'outer_skin'],
    meg=False,
    fig=fig,
    bem=bem,
    coord_frame='mri'
)
dips[sel_dip].plot_locations(
    trans,
    f'sub-{subject:02d}',
    fname.subjects_dir,
    mode='arrow',
    fig=fig,
    coord_frame='mri'
)
maps = mne.make_field_map(
    evoked,
    trans,
    f'sub-{subject:02d}',
    fname.subjects_dir,
    ch_type='meg'
)
evoked.plot_field(maps, time=time, fig=fig)

# Tweak the surfaces in the plot so you can see everything
a = fig.children[1].children[0].children[0].children[0].actor.property.opacity = 0.2
a = fig.children[2].children[0].children[0].children[0].actor.property.opacity = 0.2
fig.children[5].children[0].children[0].children[0].actor.visible = False

act, _ = mne.fit_dipole(
    evoked,
    cov,
    bem,
    trans,
    pos=dips.pos[sel_dip],
    ori=dips.ori[sel_dip],
    verbose=False,
    min_dist=0,
)
act.plot()

## 
# Get dipole activity for each epoch
#
#proj = np.array([mne.dipole.project_dipole(dip, epochs, cov, bem, trans, verbose=True)
#                 for dip in dips])
proj = np.load(fname.dip_timecourses(subject=subject))['proj']
n_dip = len(proj)
fig, axes = plt.subplots(n_dip//3+1, 3, sharex=True, sharey=False)
for p, ax in zip(proj, axes.flat):
    for cl in ['noisy word', 'consonants', 'pseudoword', 'word', 'symbols']:
        w = p[epochs.metadata.type == cl].mean(axis=0)
        ax.plot(epochs.times, w, label=cl)
    ax.legend()

##
plt.figure()
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.plot(1000 * epochs.times, ls[metadata.query("type=='word'").index].mean(axis=0), color='C0', label='Word')
plt.plot(1000 * epochs.times, ls[metadata.query("type=='pseudoword'").index].mean(axis=0), color='black', label='Pseudoword')
plt.plot(1000 * epochs.times, ls[metadata.query("type=='consonants'").index].mean(axis=0), color='pink', label='Consonant string')
plt.plot(1000 * epochs.times, ls[metadata.query("type=='symbols'").index].mean(axis=0), color='red', label='Symbol string')
plt.plot(1000 * epochs.times, ls[metadata.query("type=='noisy word'").index].mean(axis=0), color='green', label='Noisy word')
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.legend(loc='upper right')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (nAm)')
plt.title('Left occipital-temporal cortex  at ~150 ms (n=14)')

