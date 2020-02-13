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
if user == 'wmvan':
    # My laptop
    raw_data_dir = '../../data/pilot_data'
    n_jobs = 8  # My laptop has 8 cores
elif host == 'nbe-024.org.aalto.fi' and user == 'vanvlm1':
    # My workstation
    raw_data_dir = '/m/nbe/scratch/reading_models/pilot_data'
    n_jobs = 8  # My workstation has 8 cores
elif host == 'ECIT01472U':
    raw_data_dir = '../../data/pilot_data'
    n_jobs = 4
else:
    raise ValueError('Please add your system to config.py')

# For BLAS to use the right amount of cores
os.environ['OMP_NUM_THREADS'] = str(n_jobs)


###############################################################################
# These are all the relevant parameters for the analysis.

# Bandpass filter
fmin = 0.1  # Hz
fmax = 40.0  # Hz

# All subjects
subjects = [1, 2]
n_runs = {1: 3, 2: 4}

# Bad channels
bads = {
    1: ['MEG2233', 'MEG1842', 'MEG2621'],
    2: ['MEG2233', 'MEG1842', 'MEG2621'],
}

event_id = dict(word=10, consonants=20, symbols=30, question=40)

###############################################################################
# Templates for filenames
#
# This part of the config file uses the FileNames class. It provides a small
# wrapper around string.format() to keep track of a list of filenames.
# See fnames.py for details on how this class works.
fname = FileNames()

# Some directories
fname.add('raw_data_dir', raw_data_dir)
fname.add('stimulus_dir', '{raw_data_dir}/pilot{subject:d}/stimuli')

# The data files that are used and produced by the analysis steps
fname.add('raw', '{raw_data_dir}/pilot{subject:d}/pilot{subject:d}_run{run:d}_raw.fif')
fname.add('log', '{raw_data_dir}/pilot{subject:d}/pilot{subject:d}-run{run:d}.log')
fname.add('stimuli', '{raw_data_dir}/pilot{subject:d}/run{run:d}.csv')
fname.add('stimulus_image', '{stimulus_dir}/{stim_fname}')
fname.add('raw_filt', '{raw_data_dir}/pilot{subject:d}/pilot{subject:d}_run{run:d}_filt_raw.fif')
fname.add('raw_detrend', '{raw_data_dir}/pilot{subject:d}/pilot{subject:d}_run{run:d}_detrend_raw.fif')
fname.add('eog_events', '{raw_data_dir}/pilot{subject:d}/pilot{subject:d}_run{run:d}_eog-eve.fif')
fname.add('ecg_events', '{raw_data_dir}/pilot{subject:d}/pilot{subject:d}_run{run:d}_ecg-eve.fif')
fname.add('ica', '{raw_data_dir}/pilot{subject:d}/pilot{subject:d}_run{run:d}_ica.fif')
fname.add('epochs', '{raw_data_dir}/pilot{subject:d}/pilot{subject:d}_epo.fif')
fname.add('evoked', '{raw_data_dir}/pilot{subject:d}/pilot{subject:d}_ave.fif')

# Filenames for MNE reports
fname.add('reports_dir', './reports/')
fname.add('report', '{reports_dir}/pilot{subject:d}-report.h5')
fname.add('report_html', '{reports_dir}/pilot{subject:d}-report.html')

# File produced by check_system.py
fname.add('system_check', './system_check.txt')
