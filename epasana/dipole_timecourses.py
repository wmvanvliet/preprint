"""
Compute dipole activity for each epoch.
"""
import mne
import mne_bids
from config import fname, n_jobs
import numpy as np
import sys

subject = int(sys.argv[1])

epochs = mne.read_epochs(fname.epochs(subject=subject))
trans = mne_bids.get_head_mri_trans(fname.raw(subject=subject), fname.bids_root)
bem = mne.read_bem_solution(fname.bem(subject=subject))
dips = mne.read_dipole(fname.dip(subject=subject))

# We need an empirical version of the covariance matrix, because the rank
# detection bugs out on the version with shrinkage
cov = mne.compute_covariance(epochs)

# Calling of this function happens way below at the end of this script
@mne.utils.verbose
def project_dipole(dip, epochs, cov, bem, trans=None, free_ori=False,
                   rank=None, verbose=None):
    """Project sensor data onto a dipole to estimate the source timecourse.

    This function was lifted from MNE-Python and hacked upon to work on Epochs
    objects.

    Parameters
    ----------
    dip : instance of Dipole | instance of DipoleFixed
        The source dipole to project the sensor data onto.
    epochs : instance of Epochs
        The sensor data to project onto the dipole.
    cov : str | instance of Covariance
        The noise covariance.
    bem : str | instance of ConductorModel
        The BEM filename (str) or conductor model.
    trans : str | None
        The head<->MRI transform filename. Must be provided unless BEM
        is a sphere model.
    free_ori : bool
        Whether to allow the orientation of the dipole to change over time to
        optimize the goodness of fit, or should remain fixed. Defaults to
        False.
    %(rank_None)s
    %(n_jobs)s
        Number of CPU cores to use.
    %(verbose)s

    Returns
    -------
    source_timecourse : ndarray, shape (n_times,)
        The source timecourse.

    See Also
    --------
    Dipole
    DipoleFixed
    fit_dipole
    read_dipole
    """
    from mne.io.proj import _needs_eeg_average_ref_proj
    from mne.io.pick import pick_types
    from mne.utils import logger, _pl
    from mne.forward._make_forward import (_get_trans, _setup_bem,
                                           _prep_meg_channels,
                                           _prep_eeg_channels)
    from mne.forward._compute_forward import _prep_field_computation
    from mne.bem import _bem_find_surface, _fit_sphere
    from mne.transforms import _print_coord_trans, apply_trans
    from mne.cov import read_cov, compute_whitener
    from mne.surface import transform_surface_to
    from mne.dipole import _dipole_forwards, _fit_dipole_fixed, _fit_dipoles
    from mne.parallel import parallel_func
    from scipy import linalg

    # Determine if a list of projectors has an average EEG ref
    if _needs_eeg_average_ref_proj(epochs.info):
        raise ValueError('EEG average reference is mandatory for dipole '
                         'fitting.')

    data = epochs.get_data()
    if not np.isfinite(data).all():
        raise ValueError('Evoked data must be finite')
    info = epochs.info
    times = epochs.times.copy()

    # Figure out our inputs
    neeg = len(pick_types(info, meg=False, eeg=True, ref_meg=False,
                          exclude=[]))
    if isinstance(bem, str):
        bem_extra = bem
    else:
        bem_extra = repr(bem)
        logger.info('BEM               : %s' % bem_extra)
    mri_head_t, trans = _get_trans(trans)
    logger.info('MRI transform     : %s' % trans)
    bem = _setup_bem(bem, bem_extra, neeg, mri_head_t, verbose=False)
    if not bem['is_sphere']:
        # Find the best-fitting sphere
        inner_skull = _bem_find_surface(bem, 'inner_skull')
        inner_skull = inner_skull.copy()
        R, r0 = _fit_sphere(inner_skull['rr'], disp=False)
        # r0 back to head frame for logging
        r0 = apply_trans(mri_head_t['trans'], r0[np.newaxis, :])[0]
        inner_skull['r0'] = r0
        logger.info('Head origin       : '
                    '%6.1f %6.1f %6.1f mm rad = %6.1f mm.'
                    % (1000 * r0[0], 1000 * r0[1], 1000 * r0[2], 1000 * R))
        del R, r0
    else:
        r0 = bem['r0']
        if len(bem.get('layers', [])) > 0:
            R = bem['layers'][0]['rad']
            kind = 'rad'
        else:  # MEG-only
            # Use the minimum distance to the MEG sensors as the radius then
            R = np.dot(linalg.inv(info['dev_head_t']['trans']),
                       np.hstack([r0, [1.]]))[:3]  # r0 -> device
            R = R - [info['chs'][pick]['loc'][:3]
                     for pick in pick_types(info, meg=True, exclude=[])]
            if len(R) == 0:
                raise RuntimeError('No MEG channels found, but MEG-only '
                                   'sphere model used')
            R = np.min(np.sqrt(np.sum(R * R, axis=1)))  # use dist to sensors
            kind = 'max_rad'
        logger.info('Sphere model      : origin at (% 7.2f % 7.2f % 7.2f) mm, '
                    '%s = %6.1f mm'
                    % (1000 * r0[0], 1000 * r0[1], 1000 * r0[2], kind, R))
        inner_skull = dict(R=R, r0=r0)  # NB sphere model defined in head frame
        del R, r0
    accurate = False  # can be an option later (shouldn't make big diff)

    pos = dip.pos[0].astype(np.float)
    logger.info('Dipole position    : %6.1f %6.1f %6.1f mm'
                % tuple(1000 * pos))

    if free_ori:
        logger.info('Free orientation   : <time-varying>')
    else:
        ori = dip.ori[0].astype(np.float)
        logger.info('Dipole orientation  : %6.4f %6.4f %6.4f mm'
                    % tuple(ori))

    if isinstance(cov, str):
        logger.info('Noise covariance  : %s' % (cov,))
        cov = read_cov(cov, verbose=False)
    logger.info('')

    _print_coord_trans(mri_head_t)
    _print_coord_trans(info['dev_head_t'])
    logger.info('%d bad channels total' % len(info['bads']))

    # Forward model setup (setup_forward_model from setup.c)
    ch_types = epochs.get_channel_types()

    megcoils, compcoils, megnames, meg_info = [], [], [], None
    eegels, eegnames = [], []
    if 'grad' in ch_types or 'mag' in ch_types:
        megcoils, compcoils, megnames, meg_info = \
            _prep_meg_channels(info, exclude='bads',
                               accurate=accurate, verbose=verbose)
    if 'eeg' in ch_types:
        eegels, eegnames = _prep_eeg_channels(info, exclude='bads',
                                              verbose=verbose)

    # Ensure that MEG and/or EEG channels are present
    if len(megcoils + eegels) == 0:
        raise RuntimeError('No MEG or EEG channels found.')

    # Whitener for the data
    logger.info('Decomposing the sensor noise covariance matrix...')
    picks = pick_types(info, meg=True, eeg=True, ref_meg=False)

    whitener, _, rank = compute_whitener(cov, info, picks=picks,
                                         rank=rank, return_rank=True)

    # Proceed to computing the fits
    guess_src = dict(nuse=1, rr=pos[np.newaxis], inuse=np.array([True]))
    logger.info('Compute forward for dipole location...')

    # inner_skull goes from mri to head frame
    if 'rr' in inner_skull:
        transform_surface_to(inner_skull, 'head', mri_head_t)

    # C code computes guesses w/sphere model for speed, don't bother here
    fwd_data = dict(coils_list=[megcoils, eegels], infos=[meg_info, None],
                    ccoils_list=[compcoils, None], coil_types=['meg', 'eeg'],
                    inner_skull=inner_skull)
    # fwd_data['inner_skull'] in head frame, bem in mri, confusing...
    _prep_field_computation(guess_src['rr'], bem, fwd_data, n_jobs=n_jobs,
                            verbose=False)
    guess_fwd, guess_fwd_orig, guess_fwd_scales = _dipole_forwards(
        fwd_data, whitener, guess_src['rr'], n_jobs=1)
    # decompose ahead of time
    guess_fwd_svd = [linalg.svd(fwd, overwrite_a=False, full_matrices=False)
                     for fwd in np.array_split(guess_fwd,
                                               len(guess_src['rr']))]
    guess_data = dict(fwd=guess_fwd, fwd_svd=guess_fwd_svd,
                      fwd_orig=guess_fwd_orig, scales=guess_fwd_scales)
    del guess_fwd, guess_fwd_svd, guess_fwd_orig, guess_fwd_scales  # destroyed
    logger.info('[done %d source%s]' % (guess_src['nuse'],
                                        _pl(guess_src['nuse'])))

    # Do actual fits
    data = data[:, picks]
    parallel, p_fun, _ = parallel_func(_fit_dipoles, n_jobs=n_jobs)
    timecourses = parallel(p_fun(_fit_dipole_fixed, 0, d, times,
                                 guess_src['rr'], guess_data, fwd_data,
                                 whitener, ori, 1, rank)
                           for d in data)
    timecourses = [out[1] for out in timecourses]
    return np.array(timecourses)

# Compute dipole timecourses
proj = np.array([mne.dipole.project_dipole(dips[i], epochs, cov, bem, trans, verbose=True)
                 for i in range(len(dips))])
proj = np.savez_compressed(fname.dip_timecourses(subject=subject), proj=proj)
