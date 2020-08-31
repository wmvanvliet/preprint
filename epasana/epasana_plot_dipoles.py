import mne
import mne_bids
from config import fname, n_jobs, subjects
from mayavi import mlab
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

foci = []
for subject in subjects[:1]:
    info = mne.io.read_info(fname.raw(subject=subject))
    trans = mne_bids.get_head_mri_trans(fname.raw(subject=subject), fname.bids_root)
    dips = mne.read_dipole(fname.dip(subject=subject))
    dip_selection = pd.read_csv(fname.dip_selection(subject=subject), sep='\t', index_col=0)
    if 'LeftOcci1' in dip_selection.index:
        foci.append(
            mne.head_to_mri(
                dips.pos[dip_selection.loc['LeftOcci1'].dipole],
                f'sub-{subject:02d}',
                trans,
                subjects_dir=fname.subjects_dir
            )
        )
foci = np.array(foci)

ga_stc = mne.read_source_estimate(fname.ga_stc(condition='word'))
fig = mlab.figure(size=(1000, 1000))
brain = ga_stc.copy().crop(0.065, 0.11).mean().plot(
    'fsaverage',
    subjects_dir=fname.subjects_dir,
    hemi='lh',
    background='white',
    cortex='low_contrast',
    surface='inflated',
    figure=fig,
)
brain.add_foci(
    foci,
    map_surface='pial'
)
brain.scale_data_colormap(2.7, 3, 7, True)

##
fig = mlab.figure(size=(1000, 1000))
brain = ga_stc.copy().crop(0.14, 0.2).mean().plot(
    'fsaverage',
    subjects_dir=fname.subjects_dir,
    hemi='lh',
    background='white',
    cortex='low_contrast',
    surface='inflated',
    figure=fig,
)
brain.scale_data_colormap(3.5, 3.8, 7, True)

##
fig = mlab.figure(size=(1000, 1000))
brain = ga_stc.copy().crop(0.3, 0.5).mean().plot(
    'fsaverage',
    subjects_dir=fname.subjects_dir,
    hemi='lh',
    background='white',
    cortex='low_contrast',
    surface='inflated',
    figure=fig,
)
brain.scale_data_colormap(4, 4.5, 6, True)
