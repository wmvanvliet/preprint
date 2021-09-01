"""
===========
Config file
===========

Configuration parameters for the study.
"""

import os
import getpass
from socket import getfqdn
from fnames import FileNames

###############################################################################
# Determine which user is running the scripts on which machine and set the path
# where the data is stored and how many CPU cores to use.

user = getpass.getuser()  # Username of the user running the scripts
host = getfqdn()  # Hostname of the machine running the scripts

# You want to add your machine to this list
if host == 'nbe-024.org.aalto.fi' and user == 'vanvlm1':
    # My workstation
    orig_epasana_dir = '/m/nbe/scratch/epasana/bids'
    reading_models_dir = '/m/nbe/scratch/reading_models'
    n_jobs = 4  # My workstation has 4 cores
elif user == 'wmvan':
    # My laptop
    orig_epasana_dir = 'M:/scratch/epasana/bids'
    reading_models_dir = '../data'
    n_jobs = 6  # My laptop has 6 cores
elif user == 'vanvlm1':
    orig_epasana_dir = '/m/nbe/scratch/epasana/bids'
    reading_models_dir = '/m/nbe/scratch/reading_models'
    n_jobs = 1
else:
    raise ValueError('Please add your system to config.py')

# For BLAS to use the right amount of cores
os.environ['OMP_NUM_THREADS'] = str(n_jobs)


###############################################################################
# These are all the relevant parameters for the analysis.

# All subjects
subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

###############################################################################
# Templates for filenames
#
# This part of the config file uses the FileNames class. It provides a small
# wrapper around string.format() to keep track of a list of filenames.
# See fnames.py for details on how this class works.
fname = FileNames()

# Some directories
fname.add('orig_epasana_dir', orig_epasana_dir)
fname.add('reading_models_dir', reading_models_dir)
fname.add('derivatives_dir', '{orig_epasana_dir}/derivatives/marijn')
fname.add('dipoles_dir', '{orig_epasana_dir}/derivatives/dipoles')
fname.add('stimulus_dir', '{orig_epasana_dir}/stimulus_images')

# The data files that are used and produced by the analysis steps
fname.add('epochs', '{derivatives_dir}/sub-{subject:02d}/meg/sub-{subject:02d}-epo.fif')
fname.add('evokeds', '{derivatives_dir}/sub-{subject:02d}/meg/sub-{subject:02d}-ave.fif')
fname.add('cov', '{derivatives_dir}/sub-{subject:02d}/meg/sub-{subject:02d}-cov.fif')
fname.add('inv', '{derivatives_dir}/sub-{subject:02d}/meg/sub-{subject:02d}-inv.fif')
fname.add('fwd', '{derivatives_dir}/sub-{subject:02d}/meg/sub-{subject:02d}-fwd.fif')
fname.add('src', '{derivatives_dir}/sub-{subject:02d}/meg/sub-{subject:02d}-src.fif')
fname.add('morph', '{derivatives_dir}/sub-{subject:02d}/meg/sub-{subject:02d}-morph.h5')
fname.add('stc', '{derivatives_dir}/sub-{subject:02d}/meg/sub-{subject:02d}-{condition}')
fname.add('ga_stc', '{derivatives_dir}/grand_average/grand_average-{condition}')
fname.add('dsms', '{reading_models_dir}/epasana/sub-{subject:02d}/sub-{subject:02d}_dsms.npz')
fname.add('layer_corr', '{reading_models_dir}/epasana/sub-{subject:02d}/sub-{subject:02d}_layer_corr-ave.fif')
fname.add('ga_layer_corr', '{reading_models_dir}/epasana/grand_average/ga_layer_corr-ave.fif')
fname.add('stc_layer_corr', '{reading_models_dir}/epasana/sub-{subject:02d}/sub-{subject:02d}_layer_corr_stc.npz')
fname.add('ga_stc_layer_corr', '{reading_models_dir}/epasana/grand_average/ga_layer_corr_stc.npz')
fname.add('contrasts', '{reading_models_dir}/epasana/sub-{subject:02d}/sub-{subject:02d}_contrasts-ave.fif')
fname.add('ga_contrasts', '{reading_models_dir}/epasana/grand_average/ga_contrasts-ave.fif')
fname.add('model', '{reading_models_dir}/models/{name}_pth.tar')
fname.add('model_dsms', '{reading_models_dir}/dsms/epasana_{name}_dsms.pkl')
fname.add('ga_epochs', '{reading_models_dir}/epasana/grand_average/items-epo.fif')
fname.add('stimulus_selection', 'stimulus_selection.csv')

# Original BIDS files
fname.add('bids_root', '{orig_epasana_dir}')
fname.add('raw', '{bids_root}/sub-{subject:02d}/meg/sub-{subject:02d}_task-epasana_meg.fif')
fname.add('events', '{bids_root}/sub-{subject:02d}/meg/sub-{subject:02d}_task-epasana_events.tsv')
fname.add('channels', '{bids_root}/sub-{subject:02d}/meg/sub-{subject:02d}_task-epasana_channels.tsv')

# Dipoles
fname.add('dip', '{dipoles_dir}/sub-{subject:02d}/meg/sub-{subject:02d}_task-epasana_dipoles.bdip')
fname.add('dip_tal', '{dipoles_dir}/sub-{subject:02d}/meg/sub-{subject:02d}_task-epasana_dipoles_talairach.bdip')
fname.add('dip_timecourses', '{dipoles_dir}/sub-{subject:02d}/meg/sub-{subject:02d}_task-epasana_dipole_timecourses.npz')
fname.add('dip_selection', '{dipoles_dir}/sub-{subject:02d}/meg/sub-{subject:02d}_task-epasana_dipole_selection.tsv')
fname.add('dip_layer_corr', '{reading_models_dir}/epasana/sub-{subject:02d}/sub-{subject:02d}_dip_layer_corr.csv')
fname.add('dip_activation', '{dipoles_dir}/epasana_dipole_activation.csv')

# Freesurfer stuff
fname.add('subjects_dir', '{orig_epasana_dir}/derivatives/freesurfer')
fname.add('bem', '{subjects_dir}/sub-{subject:02d}/bem/sub-{subject:02d}_5120-5120-5120_bem-sol.fif')
fname.add('trans_tal', '{subjects_dir}/sub-{subject:02d}/mri/transforms/talairach.xfm')

# Brain <-> Model comparison
fname.add('brain_model_comparison', '{reading_models_dir}/epasana/brain_model_comparison.csv')
fname.add('layer_activaty', '{reading_models_dir}/epasana/{model}_layer_activity.pth')

# Reports
fname.add('report', 'reports/sub{subject:02d}_report.h5')
fname.add('report_html', 'reports/sub{subject:02d}_report.html')

# Filenames in BIDSPath format
def bids_raw(fname, subject):
    from mne_bids import BIDSPath
    return BIDSPath(subject=f'{subject:02d}', task='epasana', suffix='meg', extension='fif', root=fname.bids_root)
fname.add('bids_raw', bids_raw)
