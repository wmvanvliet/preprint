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
fname.add('stimulus_dir', '{orig_epasana_dir}/stimulus_images')

# The data files that are used and produced by the analysis steps
fname.add('epochs', '{derivatives_dir}/sub-{subject:02d}/meg/sub-{subject:02d}-epo.fif')
fname.add('evokeds', '{derivatives_dir}/sub-{subject:02d}/meg/sub-{subject:02d}-ave.fif')
fname.add('dsms', '{reading_models_dir}/epasana/sub-{subject:02d}/sub-{subject:02d}_dsms.npz')
fname.add('layer_corr', '{reading_models_dir}/epasana/sub-{subject:02d}/sub-{subject:02d}_layer_corr-ave.fif')
fname.add('ga_layer_corr', '{reading_models_dir}/epasana/grand_average/ga_layer_corr-ave.fif')
fname.add('model', '{reading_models_dir}/models/{name}_pth.tar')
fname.add('model_dsms', '{reading_models_dir}/dsms/epasana_{name}_dsms.pkl')
fname.add('ga_epochs', '{reading_models_dir}/epasana/grand_average/items-epo.fif')
fname.add('stimulus_selection', 'stimulus_selection.csv')
fname.add('info_102', 'info_102.fif')
