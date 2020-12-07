import mne
from config import fname, subjects
from mayavi import mlab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

views = [
    (180, 90, 440, [0, 0, 0]),
    (-90, 180, 440, [0, 0, 0]),
]
view_names = ['lateral', 'ventral']

##
foci = []
for subject in subjects:
    dips = mne.read_dipole(fname.dip_tal(subject=subject))
    dip_selection = pd.read_csv(fname.dip_selection(subject=subject), sep='\t', index_col=0)
    if 'LeftOcci1' in dip_selection.index:
        foci.append(dips.pos[dip_selection.loc['LeftOcci1'].dipole] * 1000)
foci = np.array(foci)

# to_colin27 = mne.Transform('fs_tal', 'mri', [
#     [1.03061319e+00, 1.75597081e-03, 1.32446652e-02, -2.93965741e-01],
#     [-7.64009722e-03, 1.02982571e+00, -5.51759364e-02, 1.96450828e+01],
#     [-1.56219187e-02, -3.15304730e-02, 1.05012865e+00, -1.86331397e+01],
#     [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
# ])
# foci = mne.transforms.apply_trans(to_colin27, foci)

ga_stc = mne.read_source_estimate(fname.ga_stc(condition='word'))
fig1 = mlab.figure(1, size=(1000, 1000))
mne.viz.set_3d_backend('mayavi')
brain = ga_stc.copy().crop(0.065, 0.11).mean().plot(
    'fsaverage',
    subjects_dir=fname.subjects_dir,
    hemi='lh',
    background='white',
    cortex='low_contrast',
    surface='inflated',
    figure=fig1,
)
brain.scale_data_colormap(3.5, 3.8, 7, True)

# Save images without dipoles
for view, view_name in zip(views, view_names):
    mlab.view(*view)
    # Save with a transparent background (must route through mlab.screenshot to obtain this)
    plt.imsave(f'figures/landmark1_{view_name}_no_dipoles.png', mlab.screenshot(mode='rgba', antialiased=True))

# Save images with dipoles
brain.add_foci(foci, map_surface='white', hemi='lh', scale_factor=0.2)
for view, view_name in zip(views, view_names):
    mlab.view(*view)
    plt.imsave(f'figures/landmark1_{view_name}_with_dipoles.png', mlab.screenshot(mode='rgba', antialiased=True))

##
foci = []
for subject in subjects:
    dips = mne.read_dipole(fname.dip_tal(subject=subject))
    dip_selection = pd.read_csv(fname.dip_selection(subject=subject), sep='\t', index_col=0)
    if 'LeftOcciTemp2' in dip_selection.index:
        foci.append(dips.pos[dip_selection.loc['LeftOcciTemp2'].dipole] * 1000)
foci = np.array(foci)
fig2 = mlab.figure(2, size=(1000, 1000))
brain = ga_stc.copy().crop(0.14, 0.2).mean().plot(
    'fsaverage',
    subjects_dir=fname.subjects_dir,
    hemi='lh',
    background='white',
    cortex='low_contrast',
    surface='inflated',
    figure=fig2,
)
brain.scale_data_colormap(3.5, 3.8, 7, True)

# Save images without dipoles
for view, view_name in zip(views, view_names):
    mlab.view(*view)
    # Save with a transparent background (must route through mlab.screenshot to obtain this)
    plt.imsave(f'figures/landmark2_{view_name}_no_dipoles.png', mlab.screenshot(mode='rgba', antialiased=True))

# Save images with dipoles
brain.add_foci(foci, map_surface='white', hemi='lh', scale_factor=0.2)
for view, view_name in zip(views, view_names):
    mlab.view(*view)
    plt.imsave(f'figures/landmark2_{view_name}_with_dipoles.png', mlab.screenshot(mode='rgba', antialiased=True))

##
foci = []
for subject in subjects:
    dips = mne.read_dipole(fname.dip_tal(subject=subject))
    dip_selection = pd.read_csv(fname.dip_selection(subject=subject), sep='\t', index_col=0)
    if 'LeftTemp3' in dip_selection.index:
        foci.append(dips.pos[dip_selection.loc['LeftTemp3'].dipole] * 1000)
foci = np.array(foci)
fig = mlab.figure(3, size=(1000, 1000))
brain = ga_stc.copy().crop(0.3, 0.5).mean().plot(
    'fsaverage',
    subjects_dir=fname.subjects_dir,
    hemi='lh',
    background='white',
    cortex='low_contrast',
    surface='inflated',
    figure=fig,
)
brain.scale_data_colormap(3.5, 3.8, 7, True)

# Save images without dipoles
for view, view_name in zip(views, view_names):
    mlab.view(*view)
    # Save with a transparent background (must route through mlab.screenshot to obtain this)
    plt.imsave(f'figures/landmark3_{view_name}_no_dipoles.png', mlab.screenshot(mode='rgba', antialiased=True))

# Save images with dipoles
brain.add_foci(foci, map_surface='white', hemi='lh', scale_factor=0.2)
for view, view_name in zip(views, view_names):
    mlab.view(*view)
    plt.imsave(f'figures/landmark3_{view_name}_with_dipoles.png', mlab.screenshot(mode='rgba', antialiased=True))
