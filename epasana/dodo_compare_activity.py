from config import fname

def task_compare_activity():
    for subject in range(1, 16):
        yield dict(
            name=f'sub-{subject:02d}',
            file_dep = [fname.epochs(subject=subject), 'epasana_compare_activity.py'],
            targets = [fname.layer_corr(subject=subject)],
            actions = [f'python epasana_compare_activity.py {subject}']
        )
    yield dict(
        name = 'ga',
        file_dep = [fname.ga_epochs, 'epasana_compare_activity.py'],
        targets = [fname.ga_layer_corr],
        actions = ['python epasana_compare_activity.py']
    )

def task_compare_activity_source():
    for subject in range(1, 16):
        yield dict(
            name=f'sub-{subject:02d}',
            file_dep = [fname.epochs(subject=subject), 'epasana_compare_activity_source.py'],
            targets = [fname.stc_layer_corr(subject=subject)],
            actions = [f'python epasana_compare_activity_source.py {subject}']
        )

def task_compare_activity_dipoles():
    for subject in range(1, 16):
        yield dict(
            name=f'sub-{subject:02d}',
            file_dep = [fname.dip_timecourses(subject=subject),
                        fname.dip_selection(subject=subject),
                        'epasana_compare_activity_dipoles.py'],
            targets = [fname.dip_layer_corr(subject=subject)],
            actions = [f'python epasana_compare_activity_dipoles.py {subject}']
        )
