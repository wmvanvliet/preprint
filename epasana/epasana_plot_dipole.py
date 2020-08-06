import mne
import mne_bids
from config import fname, n_jobs
from mayavi import mlab
from tqdm import tqdm

subject=1

epochs = mne.read_epochs(fname.epochs(subject=subject))
epochs.pick_types(meg='grad')
trans = mne_bids.get_head_mri_trans(fname.raw(subject=subject), fname.bids_root)
cov = mne.compute_covariance(epochs) # read_cov(fname.cov(subject=subject))
bem = mne.read_bem_solution(fname.bem(subject=subject))

##
evoked = epochs['type=="noisy word"'].average()
evoked.comment = 'noise'
#evoked = epochs[epochs.metadata.type.isin(['word', 'pseudoword', 'consonants'])].average()
#evoked.comment = 'letter'
#evoked = epochs[epochs.metadata.type.isin(['word', 'pseudoword'])].average()
#time_roi = (0.27, 0.54)

# Read a dipole
dip = mne.read_dipole('sub-01-ave.bdip')

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
)
act.plot()

# ## 
# # Get dipole activity for each epoch (WTF MNE-Python diple API???)
# acts = []
# for i in tqdm(range(len(epochs))):
#     act, _ = mne.fit_dipole(
#         epochs[i].average(),
#         cov,
#         bem,
#         trans,
#         pos=dip.pos[best],
#         ori=dip.ori[best],
#         rank=dict(grad=69),
#         verbose=False,
#     )
#     acts.append(act.data[0])
