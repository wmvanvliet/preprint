def task_compare_activity():
    for subject in range(1, 16):
        yield dict(
            name=f'sub-{subject:02d}',
            file_dep = [f'm:/scratch/epasana/bids/derivatives/marijn/sub-{subject:02d}/meg/sub-{subject:02d}-epo.fif', 'epasana_compare_activity.py'],
            targets = [f'm:/scratch/reading_models/epasana/sub-{subject:02}_layer_corr-ave.fif'],
            actions = [f'python epasana_compare_activity.py {subject}']
        )
