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

def task_contrasts():
    for subject in range(1, 16):
        yield dict(
            name=f'sub-{subject:02d}',
            file_dep = [fname.epochs(subject=subject), 'epasana_functional_localizers.py'],
            targets = [fname.contrasts(subject=subject)],
            actions = [f'python epasana_functional_localizers.py {subject}']
        )
    yield dict(
        name = 'ga',
        file_dep = [fname.ga_epochs, 'epasana_functional_localizers.py'],
        targets = [fname.ga_contrasts],
        actions = ['python epasana_functional_localizers.py']
    )
