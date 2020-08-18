import mne
import mne_bids
from config import fname, n_jobs
from mayavi import mlab
from matplotlib import pyplot as plt

subject=3

epochs = mne.read_epochs(fname.epochs(subject=subject))
epochs.pick_types(meg='grad')
#trans = mne.read_trans(f'../data/epasana/sub-{subject:02d}/sub-{subject:02}_trans.fif')
trans = mne_bids.get_head_mri_trans(fname.raw(subject=subject), fname.bids_root)
cov = mne.compute_covariance(epochs) # read_cov(fname.cov(subject=subject))
bem = mne.read_bem_solution(fname.bem(subject=subject))

src = mne.setup_volume_source_space(
    pos=5.0,
    subject=f'sub-{subject:02d}',
    subjects_dir=fname.subjects_dir,
    bem=bem,
)
fwd = mne.make_forward_solution(epochs.info, trans, src, bem=fname.bem(subject=subject))

##
# evoked = epochs['type=="noisy word"'].average()
# evoked.comment = 'noise'
# time_roi = (0.065, 0.135)
# evoked = epochs[epochs.metadata.type.isin(['word', 'pseudoword', 'consonants'])].average()
# evoked.comment = 'letter'
# time_roi = (0.135, 0.2)
evoked = epochs[epochs.metadata.type.isin(['word', 'pseudoword'])].average()
evoked.comment = 'word'
time_roi = (0.35, 0.45)
#spat_roi = ([''.join(ch.split()) for ch in mne.selection.read_selection('Left-temporal')] +
#            [''.join(ch.split()) for ch in mne.selection.read_selection('Left-frontal')] +
#            [''.join(ch.split()) for ch in mne.selection.read_selection('Left-parietal')])

# Fit a dipole
sel = evoked.copy().crop(*time_roi) #.pick_channels(spat_roi)

# Find the slope of the onset
_, peak_time = sel.get_peak('grad')
sel.crop(peak_time - 0.01, peak_time + 0.01)
print(sel)

# dip, _ = mne.fit_dipole(sel, cov, bem, trans, n_jobs=n_jobs)
# best = dip.gof.argmax()
dips = mne.beamformer.rap_music(sel, fwd, cov, n_dipoles=15)
best = dips[0].gof.argmax()

# Plot the result in 3D brain with the MRI image.
# dip.plot_locations(trans, f'sub-{subject:02d}', fname.subjects_dir, mode='orthoview')

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
for dip in dips:
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
evoked.plot_field(maps, time=dips[0].times[best], fig=fig)

# Tweak the surfaces in the plot so you can see everything
a = fig.children[1].children[0].children[0].children[0].actor.property.opacity = 0.2
a = fig.children[2].children[0].children[0].children[0].actor.property.opacity = 0.2
fig.children[-3].children[0].children[0].children[0].actor.visible = False

# Plot the dipole activities
plt.figure()
for i, dip in enumerate(dips):
    act, _ = mne.fit_dipole(
        evoked,
        cov,
        bem,
        trans,
        pos=dip.pos[0],
        ori=dip.ori[0],
        verbose=False,
    )
    plt.plot(act.times, act.data[0], label=f'dip {i}')
plt.legend()

##
# Compute epoch-specific activation of a single dipole
dip = dips[8]
proj = mne.dipole.project_dipole(dip, epochs, cov, bem, trans, verbose=True)

plt.figure()
for cl in ['noisy word', 'consonants', 'pseudoword', 'word', 'symbols']:
    w = proj[epochs.metadata.type == cl].mean(axis=0)
    plt.plot(epochs.times, w, label=cl)
plt.legend()
