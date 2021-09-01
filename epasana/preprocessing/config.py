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
    orig_epasana_dir = '/m/nbe/scratch/epasana'
    reading_models_epasana_dir = '/m/nbe/scratch/reading_models/epasana_data'
    n_jobs = 4  # My workstation has 4 cores
else:
    raise ValueError('Please add your system to config.py')

# For BLAS to use the right amount of cores
os.environ['OMP_NUM_THREADS'] = str(n_jobs)


###############################################################################
# These are all the relevant parameters for the analysis.

# All subjects
subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

event_id = dict(word=10, consonants=20, symbols=30, question=40)

###############################################################################
# Templates for filenames
#
# This part of the config file uses the FileNames class. It provides a small
# wrapper around string.format() to keep track of a list of filenames.
# See fnames.py for details on how this class works.
fname = FileNames()

# Some directories
fname.add('orig_epasana_dir', orig_epasana_dir)
fname.add('derivatives_dir', '{orig_epasana_dir}/bids/derivatives/marijn')
fname.add('stimulus_dir', '{orig_epasana_dir}/stimulus_images')

# The data files that are used and produced by the analysis steps
fname.add('epochs', '{derivatives_dir}/sub-{subject:02d}/meg/sub-{subject:02d}-epo.fif')
fname.add('dsms', '{reading_models_epasana_dir}/sub-{subject:02d}/sub-{subject:02d}_dsms.npz')
