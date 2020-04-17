"""
Do-it script to execute the entire pipeline using the doit tool:
http://pydoit.org

All the filenames are defined in config.py
"""
from config import fname, subjects, n_runs
# Configuration for the "doit" tool.
DOIT_CONFIG = dict(
    # While running scripts, output everything the script is printing to the
    # screen.
    verbosity=2,

    # When the user executes "doit list", list the tasks in the order they are
    # defined in this file, instead of alphabetically.
    sort='definition',
)


def task_check():
    """Check the system dependencies."""
    return dict(
        file_dep=['check_system.py'],
        targets=[fname.system_check],
        actions=['python check_system.py']
    )


def task_filter():
    """Step 01: Bandpass filter the data"""
    for subject in subjects:
        runs = range(1, n_runs[subject] + 1)
        for run in runs:
            yield dict(
                name=f'{subject}:{run}',
                file_dep=[fname.raw(subject=subject, run=run), '01_filter.py'],
                targets=[fname.raw_filt(subject=subject, run=run)],
                actions=[f'python 01_filter.py {subject} {run}'],
            )


def task_ica():
    """Step 02: Fit ICA"""
    for subject in subjects:
        runs = range(1, n_runs[subject] + 1)
        for run in runs:
            yield dict(
                name=f'{subject}:{run}',
                file_dep=[fname.raw_filt(subject=subject, run=run), '02_ica.py'],
                targets=[fname.ica(subject=subject, run=run)],
                actions=[f'python 02_ica.py {subject} {run}'],
            )
         
 
def task_epochs():
    """Step 03: Cut epochs, apply ICA and make evokeds"""
    for subject in subjects:
        runs = range(1, n_runs[subject] + 1)
        yield dict(
            name=str(subject),
            file_dep=(
                [fname.raw_filt(subject=subject, run=run) for run in runs] +
                [fname.ica(subject=subject, run=run) for run in runs] +
                ['03_epochs.py']),
            targets=[fname.epochs(subject=subject), fname.evoked(subject=subject)],
            actions=[f'python 03_epochs.py {subject}'],
        )


def task_dsms():
    """Step 04: Compute DSMs in a searchlight pattern"""
    for subject in subjects:
        yield dict(
            name=str(subject),
            file_dep=[fname.epochs(subject=subject), '04_dsms.py'],
            targets=[fname.dsms(subject=subject)],
            actions=[f'python 03_epochs.py {subject}'],
        )
