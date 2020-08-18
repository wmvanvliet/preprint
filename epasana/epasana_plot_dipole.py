import mne
import mne_bids
from config import fname, n_jobs
from mayavi import mlab
from matplotlib import pyplot as plt
import numpy as np

subject=3

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
dip = mne.read_dipole(fname.dip(subject=subject, landmark='word'))

# Make a fancy plot showing the realtionship between the dipole and the field lines
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
dip.plot_locations(
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
evoked.plot_field(maps, time=dip.times[0], fig=fig)

# Tweak the surfaces in the plot so you can see everything
a = fig.children[1].children[0].children[0].children[0].actor.property.opacity = 0.2
a = fig.children[2].children[0].children[0].children[0].actor.property.opacity = 0.2
fig.children[5].children[0].children[0].children[0].actor.visible = False

act, _ = mne.fit_dipole(
    evoked,
    cov,
    bem,
    trans,
    pos=dip.pos[0],
    ori=dip.ori[0],
    verbose=False,
    min_dist=0,
)
act.plot()

## 
# Get dipole activity for each epoch
#
#proj = mne.dipole.project_dipole(dip, epochs, cov, bem, trans, verbose=True)
proj = np.load(f'sub-{subject:02d}-proj.npz')['letters']

plt.figure()
for cl in ['noisy word', 'consonants', 'pseudoword', 'word', 'symbols']:
    w = proj[epochs.metadata.type == cl].mean(axis=0)
    plt.plot(epochs.times, w, label=cl)
plt.legend()
