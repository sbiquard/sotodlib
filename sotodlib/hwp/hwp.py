import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import BSpline
from scipy import sparse
from scipy.linalg import LinAlgError, cholesky_banded, cho_solve_banded
from lmfit import Model as LmfitModel
from sotodlib import core, tod_ops
from sotodlib.tod_ops import filters, apodize, fft_ops
import logging

logger = logging.getLogger(__name__)


def get_hwpss(aman, signal=None, hwp_angle=None, bin_signal=True, bins=360,
              lin_reg=True, modes=[1, 2, 3, 4, 5, 6, 7, 8], apply_prefilt=True,
              prefilt_cfg=None, prefilt_detrend='linear', flags=None,
              apodize_edges=True, apodize_edges_samps=1600, 
              apodize_flags=True, apodize_flags_samps=200, apo_type='C1',
              merge_stats=True, hwpss_stats_name='hwpss_stats',
              merge_model=True, hwpss_model_name='hwpss_model'):
    r"""
    Extracts HWP synchronous signal (HWPSS) from a time-ordered data (TOD)
    using linear regression or curve-fitting. The curve-fitting or linear
    regression are either run on the full time ordered data vs hwp angle
    or the time ordered data binned in hwp_angle. If the curve-fitting
    option is used it must be performed on the binned data.

    Parameters
    ----------
    aman : AxisManager object
        The TOD to extract HWPSS from.
    signal : str or None
        The field name in the axis manager to use for the TOD signal.
        If not provided, ``signal`` will be used.
    hwp_angle : array-like, optional
        The HWP angle for each sample in `aman`. If not provided, `aman.hwp_angle` will be used.
    bin_signal : bool, optional
        Whether to bin the TOD signal into HWP angle bins before extracting HWPSS. Default is `True`.
    bins : int, optional
        The number of HWP angle bins to use if `bin_signal` is `True`. Default is 360.
    lin_reg : bool, optional
        Whether to use linear regression to extract HWPSS from the binned signal. If `False`, curve-fitting will be used instead.
        Default is `True`.
    modes : list of int, optional
        The HWPSS harmonic modes to extract. Default is [1, 2, 3, 4, 5, 6, 7, 8].
    apply_prefilt : bool, optional
        Whether to apply a high-pass filter to signal before extracting HWPSS. Default is `True`.
        If run through preprocess and `signal` is not `aman.signal` then default to `False`.
    prefilt_cfg : dict, optional
        The configuration of the high-pass filter, in Hz. Only used if `apply_prefilt` is `True`.
        Default is sine2 filter of with cutoff frequency of 1.0 Hz and trans_width of 1.0 Hz.
    prefilt_detrend: str or None
        Method of detrending when you apply prefilter. Default is `linear`. If data is already detrended or you do not want to detrend,
        set it to `None`.
    flags : str or RangesMatrix or Ranges, optional
        Flags to be masked out before extracting HWPSS. If Default is None, and no mask will be applied.
        If provided by a string, `aman.flags.get(flags)` is used for the flags.
    apodize_edges : bool, optional
        If True, applies an apodization window to the edges of the signal. Defaults to True.
    apodize_edges_samps : int, optional
        The number of samples over which to apply the edge apodization window. Defaults to 1600.
    apodize_flags : bool, optional
        If True, applies an apodization window based on the flags. Defaults to True.
    apodize_flags_samps : int, optional
        The number of samples over which to apply the flags apodization window. Defaults to 200.
    apo_type: str, optional
        Type of apodization, default is C1. See tod_ops.apodize for all options.
    merge_stats : bool, optional
        Whether to add the extracted HWPSS statistics to `aman` as new axes. Default is `True`.
    hwpss_stats_name : str, optional
        The name to use for the new field containing the HWPSS statistics if `merge_stats` is `True`. Default is 'hwpss_stats'.
    merge_extract : bool, optional
        Whether to add the extracted HWPSS to `aman` as a new signal field. Default is `True`.
    hwpss_extract_name : str, optional
        The name to use for the new signal field containing the extracted HWPSS if `merge_extract` is `True`. Default is 'hwpss_extract'.

    Returns
    -------
    hwpss_stats : AxisManager object
        The extracted HWPSS and its statistics. The statistics include:

            - **coeffs** (n_dets x n_modes) : coefficients of the model

            .. math::
                \sum_n \mathrm{coeffs}[2n]\sin{(\mathrm{modes}[n] \chi_{\mathrm{hwp}})} + \mathrm{coeffs}[2n+1]\cos{(\mathrm{modes}[n] \chi_{\mathrm{hwp}})}

            where the sum on n range(len(modes)). **Note**: n_modes is 2*len(modes)

            - **covars** (n_dets x n_modes x n_modes) : variance covariance matrix of the fitted coefficients for each detector.
            - **redchi2** (n_dets) : reduced chi^2 of the fit for each detector.
        
            **In the binned case the following are returned:**
            
            - **binned_angle** (n_bins) : binned version of hwp_angle in range (0, 2pi] with number of bins set by bins argument.
            - **bin_counts** (n_dets x n_bins): sample counts of each bin for each detector.
            - **binned_signal** (n_dets x n_bins) : binned signal for each detector.
            - **sigma_bin** (n_dets) : average over all bins of the standard deviation of the signal within each bin.
        
            **In the non-binned case the following are returned:**

            - **sigma_tod** (n_dets) : estimate of the standard deviation of the signal using function ``estimate_sigma_tod``
    """
    if prefilt_cfg is None:
        prefilt_cfg = {'type': 'sine2', 'cutoff': 1.0, 'trans_width': 1.0}

    prefilt = filters.get_hpf(prefilt_cfg)
    if signal is None:
        #signal_name variable to be deleted when tod_ops.fourier_filter is updated
        signal_name = 'signal'
        signal = aman[signal_name]
    elif isinstance(signal, str):
        signal_name = signal
        signal = aman[signal_name]
    elif isinstance(signal, np.ndarray):
        raise TypeError("Currently ndarray not supported, need update to tod_ops.fourier_filter module to remove signal_name argument.")
    else:
        raise TypeError("Signal must be None, str, or ndarray")

    if apply_prefilt:
        # This requires signal to be a string.
        signal = np.array(tod_ops.fourier_filter(
                aman, prefilt, detrend=prefilt_detrend, signal_name=signal_name)
                )

    if hwp_angle is None:
        hwp_angle = aman.hwp_angle

    # define hwpss_stats
    mode_names = []
    for mode in modes:
        mode_names.append(f'S{mode}')
        mode_names.append(f'C{mode}')

    hwpss_stats = core.AxisManager(aman.dets, core.LabelAxis(
        name='modes', vals=np.array(mode_names, dtype='<U3')))
    if bin_signal:
        hwp_angle_bin_centers, bin_counts, binned_hwpss, hwpss_sigma_bin = get_binned_hwpss(
            aman, signal, hwp_angle=None, bins=bins, flags=flags, 
            apodize_edges=apodize_edges, apodize_edges_samps=apodize_edges_samps, 
            apodize_flags=apodize_flags, apodize_flags_samps=apodize_flags_samps, apo_type=apo_type)
        
        # check bin count
        num_invalid_bins = np.count_nonzero(np.isnan(binned_hwpss[0][:]))
        if num_invalid_bins > 0:
            logger.warning(f'There are {num_invalid_bins} bins with zero samples. ' + 
                             'You maybe using simulation data whose hwp speed is perfectly constant, ' + 
                             'or your specification of number of bins is too large.')
        
        # wrap
        hwpss_stats.wrap('binned_angle', hwp_angle_bin_centers, [
                       (0, core.IndexAxis('bin_hwp_samps', count=bins))])
        hwpss_stats.wrap('bin_counts', bin_counts, [
                       (0, 'dets'), (1, 'bin_hwp_samps')])
        hwpss_stats.wrap('binned_signal', binned_hwpss, [
                       (0, 'dets'), (1, 'bin_hwp_samps')])
        hwpss_stats.wrap('sigma_bin', hwpss_sigma_bin, [(0, 'dets')])

        if lin_reg:
            fitsig_binned, coeffs, covars, redchi2s = hwpss_linreg(
                x=hwp_angle_bin_centers, ys=binned_hwpss, yerrs=hwpss_sigma_bin, modes=modes)
        else:
            params_init = guess_hwpss_params(
                x=hwp_angle_bin_centers, ys=binned_hwpss, modes=modes)
            fitsig_binned, coeffs, covars, redchi2s = hwpss_curvefit(x=hwp_angle_bin_centers, ys=binned_hwpss, yerrs=hwpss_sigma_bin,
                                                                     modes=modes, params_init=params_init)
        # tod template
        fitsig_tod = harms_func(hwp_angle, modes, coeffs)

        # wrap the optimal values and stats
        hwpss_stats.wrap('binned_model', fitsig_binned,
                       [(0, 'dets'), (1, 'bin_hwp_samps')])
        hwpss_stats.wrap('coeffs', coeffs, [(0, 'dets'), (1, 'modes')])
        hwpss_stats.wrap('covars', covars, [
                       (0, 'dets'), (1, 'modes'), (2, 'modes')])
        hwpss_stats.wrap('redchi2s', redchi2s, [(0, 'dets')])

    else:
        if isinstance(flags, str):
            flags = aman.flags.get(flags)
        if flags is None:
            m = np.ones([aman.dets.count, aman.samps.count], dtype=bool)
        else:
            m = ~flags.mask()

        hwpss_sigma_tod = estimate_sigma_tod(signal, hwp_angle)
        hwpss_stats.wrap('sigma_tod', hwpss_sigma_tod, [(0, 'dets')])

        if lin_reg:
            fitsig_tod, coeffs, covars, redchi2s = hwpss_linreg(
                x=hwp_angle, ys=signal, yerrs=hwpss_sigma_tod, modes=modes)

        else:
            raise ValueError('Curve-fitting for TOD are specified.' +
                             'It will take too long time and return meaningless result.' +
                             'Specify (bin_signal, lin_reg) = (True, True) or (True, False) or (False, True)')

        hwpss_stats.wrap('coeffs', coeffs, [(0, 'dets'), (1, 'modes')])
        hwpss_stats.wrap('covars', covars, [
                       (0, 'dets'), (1, 'modes'), (2, 'modes')])
        hwpss_stats.wrap('redchi2s', redchi2s, [(0, 'dets')])
    
    if merge_stats:
        aman.wrap(hwpss_stats_name, hwpss_stats)
    if merge_model:
        aman.wrap(hwpss_model_name, fitsig_tod, [(0, 'dets'), (1, 'samps')])
    return hwpss_stats


def get_binned_hwpss(aman, signal=None, hwp_angle=None,
                     bins=360, flags=None, 
                     apodize_edges=True, apodize_edges_samps=1600,
                     apodize_flags=True, apodize_flags_samps=200, apo_type='C1'):
    """
    Bin time-ordered data by the HWP angle and return the binned signal and its standard deviation.

    Parameters
    ----------
    aman : TOD
        The Axismanager object to be binned.
    signal : str, optional
        The name of the signal to be binned. Defaults to aman.signal if not specified.
    hwp_angle : str, optional
        The name of the timestream of hwp_angle. Defaults to aman.hwp_angle if not specified.
    bins : int, optional
        The number of HWP angle bins to use. Default is 360.
    flags : str or RangesMatrix or Ranges. optional
        Flag indicating whether to exclude flagged samples when binning the signal.
        If provided by a string, `aman.flags.get(flags)` is used for the flags.
        Default is no mask applied.
    apodize_edges : bool, optional
        If True, applies an apodization window to the edges of the signal. Defaults to True.
    apodize_edges_samps : int, optional
        The number of samples over which to apply the edge apodization window. Defaults to 1600.
    apodize_flags : bool, optional
        If True, applies an apodization window based on the flags. Defaults to True.
    apodize_flags_samps : int, optional
        The number of samples over which to apply the flags apodization window. Defaults to 200.
    apo_type: str, optional
        Type of apodization, default is C1. See tod_ops.apodize for all options.

    Returns
    -------
    aman_proc:
        The AxisManager object which contains
        * center of each bin of hwp_angle
        * binned hwp synchrounous signal
        * estimated sigma of binned signal
    """
    if signal is None:
        signal = aman.signal
    if hwp_angle is None:
        hwp_angle = aman['hwp_angle']

    if isinstance(flags, str):
        flags = aman.flags.get(flags)

    weight_for_signal = None
    if apodize_flags and (flags is not None):
        weight_for_signal = apodize.get_apodize_window_from_flags(
            aman, flags=flags, apodize_samps=apodize_flags_samps, apo_type=apo_type)

    if apodize_edges:
        edges_apodizer = apodize.get_apodize_window_for_ends(
            aman, apodize_samps=apodize_edges_samps, apo_type=apo_type)
        if weight_for_signal is None:
            weight_for_signal = edges_apodizer
        else:
            weight_for_signal *= edges_apodizer

    binning_dict = tod_ops.bin_signal(aman, bin_by=hwp_angle, range=[0, 2*np.pi],
                              bins=bins, signal=signal, flags=flags, weight_for_signal=weight_for_signal)
    
    bin_centers = binning_dict['bin_centers']
    bin_counts = binning_dict['bin_counts']
    binned_hwpss = binning_dict['binned_signal']
    binned_hwpss_sigma = binning_dict['binned_signal_sigma']
    
    # use median of sigma of each bin as uniform sigma for a detector
    hwpss_sigma = np.nanmedian(binned_hwpss_sigma, axis=-1)
    
    return bin_centers, bin_counts, binned_hwpss, hwpss_sigma


def hwpss_linreg(x, ys, yerrs, modes):
    """
    Performs a linear regression of the input data ys as a function of x, using a set of sine and cosine
    basis functions defined by the input modes. Returns the fitted signal, the coefficients of the
    basis functions, their covariance matrix, and the reduced chi-square.

    Parameters
    -----------
    x : numpy.ndarray
        The independent variable values of the data points to fit.
    ys : numpy.ndarray
        The dependent variable values of the data points to fit.
    yerrs : numpy.ndarray
        The error estimates of the dependent variable values.
    modes : list of int
        The frequencies of the sine and cosine basis functions to use.

    Returns
    -------
    fitsig : numpy.ndarray
        The fitted signal, obtained by evaluating the model with the optimal coefficients.
    coeffs : numpy.ndarray
        The coefficients of the sine and cosine basis functions that best fit the data.
    covars : numpy.ndarray
        The covariance matrix of the coefficients, estimated from the data errors.
    redchi2s : numpy.ndarray
        The reduced chi-square statistic of the fit, computed for each data point.
    """
    m = np.isnan(ys[0][:])
    xn = np.copy(x)
    x = x[~m]
    ys = ys[:, ~m]

    vects = np.zeros([2*len(modes), x.shape[0]], dtype='float32')
    for i, mode in enumerate(modes):
        vects[2*i, :] = np.sin(mode*x)
        vects[2*i+1, :] = np.cos(mode*x)

    I = np.linalg.inv(np.tensordot(vects, vects, (1, 1)))
    coeffs = np.matmul(ys, vects.T)
    coeffs = np.dot(I, coeffs.T).T
    fitsig = harms_func(x, modes, coeffs)

    # covariance of coefficients
    covars = np.zeros((ys.shape[0], 2*len(modes), 2*len(modes)))
    for det_idx in range(ys.shape[0]):
        covars[det_idx, :, :] = I * yerrs[det_idx]**2

    # reduced chi-square
    redchi2s = np.sum(
        ((ys - fitsig)/yerrs[:, np.newaxis])**2, axis=-1) / (x.shape[0] - 2*len(modes))

    fitsig = harms_func(xn, modes, coeffs)
    return fitsig, coeffs, covars, redchi2s


def harms_func(x, modes, coeffs):
    """
    calculates the harmonics function given the input values, modes and coefficients.

    Parameters
    ----------
    x (numpy.ndarray): Input values
    modes (list): List of modes to be used in the harmonics function
    coeffs (numpy.ndarray): Coefficients of the harmonics function

    Returns
    -------
    numpy.ndarray: The calculated harmonics function.
    """
    vects = np.zeros([2*len(modes), x.shape[0]], dtype=coeffs.dtype if coeffs is not None else 'float32')
    for i, mode in enumerate(modes):
        vects[2*i, :] = np.sin(mode*x)
        vects[2*i+1, :] = np.cos(mode*x)

    if coeffs is None:
        return vects
    else:
        harmonics = np.matmul(coeffs, vects, casting='no')
        return harmonics


def guess_hwpss_params(x, ys, modes):
    """
    Compute initial guess for the coefficients of a harmonics-based fit to data.

    Parameters
    ----------
    x : array-like of shape (nsamps,)
    ys : array-like of shape (ndets, nsamps)
    modes : array-like of shape (nmodes,)
        List of modes to use in the fit.

    Returns
    -------
    Params_init : ndarray of shape (m, 2*p)
        Initial guess for the coefficients of a harmonics-based fit to the data.
    """
    m = np.isnan(ys[0][:])
    x = x[~m]
    ys = ys[:, ~m]

    vects = harms_func(x, modes, coeffs=None)
    for i, mode in enumerate(modes):
        vects[2*i, :] = np.sin(mode*x)
        vects[2*i+1, :] = np.cos(mode*x)
    params_init = 2 * np.matmul(ys, vects.T) / x.shape[0]
    return params_init

def wrapper_harms_func(x, modes, *args):
    """
    A wrapper function for the harmonics function to be used for fitting data using Scipy's curve-fitting algorithm.
    Parameters
    ----------
    x : array-like
        The x-values of the data points to be fitted.
    modes : array-like
        An array of integers representing the modes of the harmonics function.
    *args : tuple
        A tuple of arguments. The first argument should be an array of coefficients used to calculate the harmonics function.
    Returns
    -------
    y : array-like
        An array of the same length as x representing the values of the harmonics function evaluated at x using the given 
        modes and coefficients.
    """
    coeffs = np.array(args[0])
    return harms_func(x, modes, coeffs)

def hwpss_curvefit(x, ys, yerrs, modes, params_init=None):
    """
    Fit harmonics to input data using scipy's curve_fit method.

    Parameters
    ----------
    x : array_like
        1-D array of x values.
    ys : array_like
        2-D array of y values for each detector.
    yerrs : array_like
        1-D array of the standard deviation of the y values for each detector.
    modes : array_like
        1-D array of mode numbers to be fitted.
    params_init : array_like, optional
        2-D array of initial parameter values for each detector. Default is None.

    Returns
    -------
    fitsig : ndarray
        2-D array of the fitted values for each detector.
    coeffs : ndarray
        2-D array of the fitted coefficients for each detector.
    covars : ndarray
        3-D array of the covariance matrix of the fitted coefficients for each detector.
    redchi2s : ndarray
        1-D array of the reduced chi-square values for each detector.

    Notes
    -----
    This function fits a set of harmonic functions to the input data using scipy's curve_fit method.
    The `modes` parameter specifies the mode numbers to be fitted.
    The `params_init` parameter can be used to provide initial guesses for the fit parameters.
    """
    N_dets = ys.shape[0]
    N_samps = ys.shape[-1]
    N_modes = len(modes)
    
    # Handle binned data w/ 0 counts in a bin.
    m = np.isnan(ys[0][:])
    xn = np.copy(x)
    x = x[~m]
    ys = ys[:, ~m]
  
    if params_init is None:
        params_init = np.zeros((N_dets, 2*N_modes))

    coeffs = np.zeros((N_dets, 2*len(modes)))
    covars = np.zeros((N_dets, 2*len(modes), 2*len(modes)))
    redchi2s = np.zeros(N_dets)
    fitsig = np.zeros((N_dets, N_samps))

    for det_idx in range(N_dets):
        p0 = params_init[det_idx]
        coeff, covar = curve_fit(lambda x, *p0: wrapper_harms_func(x, modes, p0),
                                 x, ys[det_idx], p0=p0, sigma=yerrs[det_idx] *
                                 np.ones_like(ys[det_idx]),
                                 absolute_sigma=True)

        coeffs[det_idx, :] = coeff
        covars[det_idx, :] = covar

        yfit = harms_func(x, modes, coeff)
        fitsig[det_idx, :] = harms_func(xn, modes, coeff)
        redchi2s[det_idx] = np.sum(
            ((ys[det_idx] - yfit) / yerrs[det_idx])**2) / (x.shape[0] - 2*len(modes))

    return fitsig, coeffs, covars, redchi2s


def estimate_sigma_tod(signal, hwp_angle):
    """
    Estimate the noise level of a signal in a time-ordered data (TOD) using a half-wave plate (HWP) modulation.

    Parameters
    ----------
    signal : ndarray
        A 2D numpy array of shape (n_dets, n_samps) containing the TOD of each detector.
    hwp_angle : ndarray
        A 1D numpy array containing the HWP angles in degrees.

    Returns
    -------
    hwpss_sigma_tod : ndarray
        A 1D numpy array containing the estimated noise level for each detector.

    Notes
    -----
    This function computes the mean of the signal in each period of HWP rotation and multiplies it
    by the square root of the number of samples in that period. The standard deviation of the
    resulting values for all periods is then computed and returned as the estimated sigma of each data point.
    """
    hwp_zeros_idxes = np.where(np.abs(np.diff(hwp_angle)) > 5)[0][:] + 1
    hwpss_sigma_tod = np.zeros((signal.shape[0], hwp_zeros_idxes.shape[0] - 1))

    for i, (init_idx, end_idx) in enumerate(zip(hwp_zeros_idxes[:-1], hwp_zeros_idxes[1:])):
        hwpss_sigma_tod[:, i] = np.mean(
            signal[:, init_idx:end_idx], axis=-1) * np.sqrt(end_idx - init_idx)
    hwpss_sigma_tod = np.std(hwpss_sigma_tod, axis=-1)
    return hwpss_sigma_tod


def subtract_hwpss(aman, signal='signal', hwpss_template_name='hwpss_model',
                   subtract_name='hwpss_remove', in_place=False, remove_template=True):
    """
    Subtract the half-wave plate synchronous signal (HWPSS) template from the
    signal in the given axis manager.

    Parameters
    ----------
    aman : AxisManager
        The axis manager containing the signal and the HWPSS template.
    signal : str, optional
        The name of the field in the axis manager containing the signal to be processed.
        Defaults to 'signal'.
    hwpss_template_name : str, optional
        The name of the field in the axis manager containing the HWPSS template.
        Defaults to 'hwpss_model'.
    subtract_name : str, optional
        The name of the field in the axis manager that will store the HWPSS-subtracted signal.
        Only used if in_place is False. Defaults to 'hwpss_remove'.
    in_place : bool, optional
        If True, the subtraction is done in place, modifying the original signal in the axis manager.
        If False, the result is stored in a new field specified by subtract_name. Defaults to False.
    remove_template : bool, optional
        If True, the HWPSS template field is removed from the axis manager after subtraction.
        Defaults to True.

    Returns
    -------
    None
    """
    if signal is None:
        signal_name = 'signal'
        signal = aman[signal_name]
    elif isinstance(signal, str):
        signal_name = signal
        signal = aman[signal_name]
    elif isinstance(signal, np.ndarray):
        if np.shape(signal) != (aman.dets.count, aman.samps.count):
            raise ValueError("When passing signal as ndarray shape must match (n_dets x n_samps).")
        signal_name = None
    else:
        raise TypeError("Signal must be None, str, or ndarray")

    if in_place:
        if signal_name is None:
            signal -= aman[hwpss_template_name].astype(signal.dtype)
        else:
            aman[signal_name] -= aman[hwpss_template_name].astype(aman[signal_name].dtype)
    else:
        if subtract_name in aman._fields:
            aman[subtract_name] = np.subtract(signal, aman[hwpss_template_name], dtype='float32')
        else:
            aman.wrap(subtract_name, np.subtract(signal,
                      aman[hwpss_template_name], dtype='float32'),
                      [(0, 'dets'), (1, 'samps')])
    
    if remove_template:
        aman.move(hwpss_template_name, None)


def _build_knot_vector(x, n_knots, degree=3):
    """
    Build a clamped (open-uniform) knot vector for a degree-`degree` B-spline
    over the domain [x[0], x[-1]], with `n_knots` interior breakpoints.

    The number of resulting B-spline basis functions is `n_knots + degree + 1`.
    """
    x_min, x_max = x[0], x[-1]
    interior = np.linspace(x_min, x_max, n_knots + 2)[1:-1]
    return np.pad(interior, degree + 1, mode='constant', constant_values=(x_min, x_max))


def get_bspline_design_matrix(x, n_knots=None, samples_per_knot=4000, degree=3,
                               min_knots=2, extrapolate=False):
    """
    Build the sparse local-support B-spline design matrix for sample
    locations `x` (typically `aman.timestamps`).

    Parameters
    ----------
    x : array-like
        Non-decreasing 1-D sample locations (e.g. timestamps).
    n_knots : int, optional
        Number of interior knot breakpoints. If None, computed from
        `samples_per_knot` as ``max(min_knots, len(x) // samples_per_knot)``.
    samples_per_knot : int, optional
        Used to auto-compute `n_knots` when not given explicitly. Default 4000.
    degree : int, optional
        B-spline degree. Default 3 (cubic).
    min_knots : int, optional
        Minimum number of interior knots when auto-computing `n_knots`. Default 2.
    extrapolate : bool, optional
        Whether to allow evaluating the basis outside [x[0], x[-1]]. Default
        False, which raises if this is ever attempted (e.g. when rebuilding a
        template on samples outside the domain the spline was originally fit on).

    Returns
    -------
    B : sparse matrix, shape (len(x), n_bases)
        The B-spline design matrix. Each row has at most `degree + 1` nonzero
        entries (local support).
    knots : ndarray, shape (n_bases + degree + 1,)
        The full clamped knot vector, needed to re-evaluate the basis later.
    """
    x = np.asarray(x)
    if n_knots is None:
        n_knots = max(min_knots, len(x) // samples_per_knot)
    knots = _build_knot_vector(x, n_knots, degree)
    B = BSpline.design_matrix(x, knots, degree, extrapolate=extrapolate)
    return B, knots


def hwpss_spline_func(timestamps, hwp_angle, modes, coeffs, knots, degree=3,
                       extrapolate=False):
    """
    Evaluate the B-spline x HWP-harmonic HWPSS template given per-detector,
    per-(harmonic, spline-basis) coefficients.

    Parameters
    ----------
    timestamps : array-like
        Sample locations the spline basis is evaluated at.
    hwp_angle : array-like
        HWP angle at each sample.
    modes : list of int
        Harmonic modes, as in `harms_func`.
    coeffs : ndarray, shape (n_dets, 2*len(modes), n_bases)
        Per-detector spline coefficients for each sin/cos harmonic component.
        `n_bases = len(knots) - degree - 1`.
    knots : ndarray
        Clamped knot vector, as returned by `get_bspline_design_matrix`.
    degree : int, optional
        B-spline degree used to build `knots`. Default 3.
    extrapolate : bool, optional
        See `get_bspline_design_matrix`. Default False.

    Returns
    -------
    template : ndarray, shape (n_dets, n_samps)
    """
    B = BSpline.design_matrix(timestamps, knots, degree, extrapolate=extrapolate)
    H = harms_func(hwp_angle, modes, coeffs=None)  # (2*n_modes, n_samps)
    n_dets = coeffs.shape[0]
    n_samps = len(timestamps)
    template = np.zeros((n_dets, n_samps))
    for c in range(2 * len(modes)):
        Bc = B @ coeffs[:, c, :].T  # (n_samps, n_dets)
        template += (Bc * H[c, :, None]).T
    return template


def _knot_major_design_matrix(B, H):
    """
    Build the B-spline x harmonic design matrix in "knot-major" column order
    (index = j*n_harm + c, for basis index j and harmonic sin/cos index c).
    """
    n_harm = H.shape[0]
    n_bases = B.shape[1]
    Bc = B.tocoo()
    t_idx, j_idx, v = Bc.row, Bc.col, Bc.data
    data = (v[None, :] * H[:, t_idx]).T.ravel()  # (nnz, n_harm), harmonic fastest
    rows = np.repeat(t_idx, n_harm)
    cols = np.repeat(j_idx, n_harm) * n_harm + np.tile(np.arange(n_harm), len(t_idx))
    return sparse.csr_array((data, (rows, cols)), shape=(B.shape[0], n_harm * n_bases))


def _solve_spline_coeffs(A, w, y, bandwidth, n_coeffs):
    """
    Solve the weighted normal equations ``(A^T diag(w) A) x = A^T diag(w) y``
    for one or more detectors at once (`y` can be 1-D, for a single
    detector, or (n_samps, n_dets), for several sharing the same weights
    `w`), exploiting the banded structure of the normal matrix. Falls back
    to a dense pseudo-inverse solve if the system is degenerate (e.g. a
    knot span with no unflagged samples).

    Parameters
    ----------
    A : sparse matrix, shape (n_samps, n_coeffs)
        Design matrix in knot-major column order (see `_knot_major_design_matrix`).
    w : ndarray, shape (n_samps,)
    y : ndarray, shape (n_samps,) or (n_samps, n_dets)
    bandwidth : int
    n_coeffs : int

    Returns
    -------
    coeffs : ndarray, shape (n_coeffs,) or (n_coeffs, n_dets)
    n_degenerate : int
        Number of coefficients with ~zero data support.
    """
    Aw = A.multiply(w[:, None])
    N = (A.T @ Aw).tocoo()
    RHS = np.asarray(Aw.T @ y)

    mask = N.col >= N.row
    rows, cols, data = N.row[mask], N.col[mask], N.data[mask]
    offsets = cols - rows
    if offsets.size and offsets.max() > bandwidth:
        msg = (
            f'normal matrix has entries {offsets.max()} columns off-diagonal, '
            f'exceeding the assumed bandwidth {bandwidth}'
        )
        raise ValueError(msg)
    ab = np.zeros((bandwidth + 1, n_coeffs))
    ab[bandwidth - offsets, cols] = data

    diag = ab[bandwidth, :]
    valid = diag > 0
    thresh = 1e-8 * np.median(diag[valid]) if valid.any() else 0.0
    n_degenerate = int(np.count_nonzero(diag <= thresh))

    def _pinv_solve():
        N_dense = sparse.coo_matrix((N.data, (N.row, N.col)), shape=(n_coeffs, n_coeffs)).toarray()
        return np.linalg.pinv(N_dense, rcond=1e-10) @ RHS

    if n_degenerate > 0:
        coeffs = _pinv_solve()
    else:
        try:
            cb = cholesky_banded(ab, lower=False)
            coeffs = cho_solve_banded((cb, False), RHS)
        except LinAlgError:
            coeffs = _pinv_solve()

    return coeffs, n_degenerate


def _get_hwpss_weights(aman, flags, apodize_edges, apodize_edges_samps,
                        apodize_flags, apodize_flags_samps, apo_type):
    """
    Build the per-sample fit weight from edge/flag apodization, shared by
    `get_hwpss_spline` and `get_hwpss_gain_spline`.

    Returns
    -------
    W : ndarray, shape (n_samps,) or (n_dets, n_samps)
        (dets, samps) if `flags` genuinely differ across detectors, else
        (samps,) shared -- see `tod_ops.apodize.get_apodize_window_from_flags`.
    """
    if isinstance(flags, str):
        flags = aman.flags.get(flags)

    W = None
    if apodize_flags and (flags is not None):
        W = apodize.get_apodize_window_from_flags(
            aman, flags=flags, apodize_samps=apodize_flags_samps, apo_type=apo_type
        )

    if apodize_edges:
        edges_apodizer = apodize.get_apodize_window_for_ends(
            aman, apodize_samps=apodize_edges_samps, apo_type=apo_type
        )
        W = edges_apodizer if W is None else W * edges_apodizer

    if W is None:
        W = np.ones(aman.samps.count)

    return W


def get_hwpss_spline(aman, signal=None, hwp_angle=None, timestamps=None,
                      modes=[1, 2, 3, 4, 5, 6, 7, 8],
                      degree=3, n_knots=None, samples_per_knot=4000,
                      apply_prefilt=True, prefilt_cfg=None, prefilt_detrend='linear',
                      flags=None,
                      apodize_edges=True, apodize_edges_samps=1600,
                      apodize_flags=True, apodize_flags_samps=200, apo_type='C1',
                      merge_stats=True, hwpss_stats_name='hwpss_stats_spline',
                      merge_model=True, hwpss_model_name='hwpss_model'):
    r"""
    Extracts a HWPSS template whose per-harmonic sin/cos amplitudes vary
    slowly in time, expanded on a shared B-spline knot grid built from
    `timestamps`:

    .. math::
        \mathrm{signal}(t) = \sum_n a_n(t)\sin{(\mathrm{modes}[n]\,\chi_\mathrm{hwp}(t))}
            + b_n(t)\cos{(\mathrm{modes}[n]\,\chi_\mathrm{hwp}(t))}

    where each :math:`a_n(t)`/:math:`b_n(t)` is itself expanded on a B-spline
    basis. This is linear in the spline coefficients, so the fit is always a
    weighted linear least-squares solve at full sample resolution.

    We use a banded Cholesky factorization to exploit the B-spline's local support.
    If the apodization/flags weights are identical across detectors, all detectors
    are solved in one shot (shared design matrix, shared factorization, multi-column
    solve). If they differ (e.g. per-detector flags), each detector is solved against
    its own weighted normal matrix.

    Parameters
    ----------
    aman : AxisManager object
        The TOD to extract HWPSS from.
    signal : str or None
        The field name in the axis manager to use for the TOD signal.
        If not provided, ``signal`` will be used.
    hwp_angle : array-like, optional
        The HWP angle for each sample in `aman`. If not provided, `aman.hwp_angle` will be used.
    timestamps : array-like, optional
        The sample locations used to build the B-spline knot grid. If not
        provided, `aman.timestamps` will be used.
    modes : list of int, optional
        The HWPSS harmonic modes to extract. Default is [1, 2, 3, 4, 5, 6, 7, 8].
    degree : int, optional
        B-spline degree. Default is 3 (cubic).
    n_knots : int, optional
        Number of interior knot breakpoints for the B-spline grid. If None,
        computed from `samples_per_knot`.
    samples_per_knot : int, optional
        Used to auto-compute `n_knots` when not given explicitly, as
        ``max(2, n_samps // samples_per_knot)``. Default is 4000.
    apply_prefilt : bool, optional
        Whether to apply a high-pass filter to signal before extracting HWPSS. Default is `True`.
        IMPORTANT: the default high-pass cutoff (1.0 Hz) will suppress slow
        HWPSS amplitude drift if `samples_per_knot` implies a drift timescale
        near or above 1 Hz -- lower the cutoff (via `prefilt_cfg`) or disable
        prefiltering for typical (minutes-scale) drift.
    prefilt_cfg : dict, optional
        The configuration of the high-pass filter, in Hz. Only used if `apply_prefilt` is `True`.
        Default is sine2 filter with cutoff frequency of 1.0 Hz and trans_width of 1.0 Hz.
    prefilt_detrend : str or None
        Method of detrending when you apply prefilter. Default is `linear`.
    flags : str or RangesMatrix or Ranges, optional
        Flags to be masked out before extracting HWPSS. Default is None, and no mask will be applied.
        If provided by a string, `aman.flags.get(flags)` is used for the flags.
    apodize_edges : bool, optional
        If True, applies an apodization window to the edges of the signal. Defaults to True.
    apodize_edges_samps : int, optional
        The number of samples over which to apply the edge apodization window. Defaults to 1600.
    apodize_flags : bool, optional
        If True, applies an apodization window based on the flags. Defaults to True.
    apodize_flags_samps : int, optional
        The number of samples over which to apply the flags apodization window. Defaults to 200.
    apo_type : str, optional
        Type of apodization, default is C1. See tod_ops.apodize for all options.
    merge_stats : bool, optional
        Whether to add the extracted HWPSS statistics to `aman` as new axes. Default is `True`.
    hwpss_stats_name : str, optional
        The name to use for the new field containing the HWPSS statistics if `merge_stats` is `True`.
    merge_model : bool, optional
        Whether to add the extracted HWPSS template to `aman` as a new signal field. Default is `True`.
    hwpss_model_name : str, optional
        The name to use for the new signal field containing the HWPSS template if `merge_model` is
        `True`. Defaults to 'hwpss_model' so that the generic `subtract_hwpss` can be reused unchanged.

    Notes
    -----
    No per-detector coefficient covariance matrices are computed or stored.

    Returns
    -------
    hwpss_stats : AxisManager object
        The extracted HWPSS spline coefficients and statistics:

            - **coeffs** (n_dets x 2*n_modes x n_bases) : per-detector, per-harmonic
              spline coefficients.
            - **knots** (n_bases + degree + 1,) : the clamped B-spline knot vector.
            - **degree** : the B-spline degree used.
            - **sigma_tod** (n_dets) : estimate of the standard deviation of the signal,
              from `estimate_sigma_tod`.
            - **redchi2s** (n_dets) : reduced chi^2 of the fit for each detector.
    """
    if prefilt_cfg is None:
        prefilt_cfg = {'type': 'sine2', 'cutoff': 1.0, 'trans_width': 1.0}

    prefilt = filters.get_hpf(prefilt_cfg)
    if signal is None:
        signal_name = 'signal'
        signal = aman[signal_name]
    elif isinstance(signal, str):
        signal_name = signal
        signal = aman[signal_name]
    elif isinstance(signal, np.ndarray):
        raise TypeError("Currently ndarray not supported, need update to tod_ops.fourier_filter module to remove signal_name argument.")
    else:
        raise TypeError("Signal must be None, str, or ndarray")

    if apply_prefilt:
        signal = np.array(
            tod_ops.fourier_filter(aman, prefilt, detrend=prefilt_detrend, signal_name=signal_name)
        )

    if hwp_angle is None:
        hwp_angle = aman.hwp_angle
    if timestamps is None:
        timestamps = aman.timestamps

    W = _get_hwpss_weights(aman, flags, apodize_edges, apodize_edges_samps,
                            apodize_flags, apodize_flags_samps, apo_type)

    B, knots = get_bspline_design_matrix(
        timestamps, n_knots=n_knots, samples_per_knot=samples_per_knot, degree=degree
    )
    n_dets = signal.shape[0]
    n_bases = B.shape[1]
    n_modes = len(modes)
    n_harm = 2 * n_modes
    n_coeffs = n_harm * n_bases
    # Basis functions j1,j2 overlap iff |j1-j2| <= degree; combined with
    # differing harmonic indices c1,c2 (flat index = j*n_harm+c) the largest
    # possible index spread is degree*n_harm + (n_harm-1).
    bandwidth = n_harm * (degree + 1) - 1
    H = harms_func(hwp_angle, modes, coeffs=None)  # (n_harm, n_samps)
    A = _knot_major_design_matrix(B, H)

    if W.ndim == 1:
        # Weights shared across all detectors: solve all in one shot
        coeffs_flat, n_degenerate = _solve_spline_coeffs(A, W, signal.T, bandwidth, n_coeffs)
        n_valid_samps = int(np.count_nonzero(W > 0))
        dof_per_det = np.full(n_dets, max(n_valid_samps - n_coeffs, 1))
    else:
        # Flags differ per detector: fit each detector separately
        coeffs_flat = np.zeros((n_coeffs, n_dets))
        dof_per_det = np.zeros(n_dets, dtype=int)
        n_degenerate = 0
        for d in range(n_dets):
            c_d, n_deg_d = _solve_spline_coeffs(A, W[d], signal[d], bandwidth, n_coeffs)
            coeffs_flat[:, d] = c_d
            n_degenerate += n_deg_d
            dof_per_det[d] = max(int(np.count_nonzero(W[d] > 0)) - n_coeffs, 1)

    if n_degenerate > 0:
        logger.warning(
            f'{n_degenerate} spline-basis x harmonic coefficients have close to '
            'zero data support (fully flagged/apodized knot spans). Falling back '
            'to a pseudo-inverse solve for the affected detector(s).'
        )

    # coeffs_flat is (n_coeffs, n_dets) in knot-major order (index = j*n_harm + c).
    # Reshape to (n_dets, n_harm, n_bases), the mode-major order expected
    # by hwpss_spline_func and the public coeffs field.
    coeffs = coeffs_flat.T.reshape(n_dets, n_bases, n_harm).transpose(0, 2, 1)

    fitsig_tod = hwpss_spline_func(timestamps, hwp_angle, modes, coeffs, knots, degree=degree)

    sigma_tod = estimate_sigma_tod(signal, hwp_angle)
    resid = signal - fitsig_tod
    w_for_chi2 = W if W.ndim == 2 else W[None, :]
    redchi2s = np.sum(w_for_chi2 * resid**2, axis=1) / sigma_tod**2 / dof_per_det

    mode_names = []
    for mode in modes:
        mode_names.append(f'S{mode}')
        mode_names.append(f'C{mode}')

    hwpss_stats = core.AxisManager(
        aman.dets,
        core.LabelAxis(name='modes', vals=np.array(mode_names, dtype='<U3')),
        core.IndexAxis('spline_basis', count=n_bases),
        core.IndexAxis('knot_grid', count=len(knots)),
    )
    hwpss_stats.wrap('coeffs', coeffs, [(0, 'dets'), (1, 'modes'), (2, 'spline_basis')])
    hwpss_stats.wrap('knots', knots, [(0, 'knot_grid')])
    hwpss_stats.wrap('degree', degree)
    hwpss_stats.wrap('sigma_tod', sigma_tod, [(0, 'dets')])
    hwpss_stats.wrap('redchi2s', redchi2s, [(0, 'dets')])

    if merge_stats:
        aman.wrap(hwpss_stats_name, hwpss_stats)
    if merge_model:
        aman.wrap(hwpss_model_name, fitsig_tod, [(0, 'dets'), (1, 'samps')])
    return hwpss_stats


def hwpss_gain_spline_func(timestamps, hwp_angle, modes, template_coeffs, gain_coeffs,
                            knots, degree=3, extrapolate=False):
    """
    Evaluate the shared-gain-drift HWPSS template: a single time-varying
    gain g(t), expanded on a B-spline basis, multiplying a fixed
    constant-coefficient harmonic template.

    signal(t) = g(t) * harms_func(hwp_angle(t), modes, template_coeffs)

    Parameters
    ----------
    template_coeffs : ndarray, shape (n_dets, 2*len(modes))
        Fixed constant per-harmonic sin/cos coefficients (e.g. from `get_hwpss`).
    gain_coeffs : ndarray, shape (n_dets, n_bases)
        Per-detector B-spline coefficients of the gain g(t).
    knots : ndarray
        Clamped knot vector, as returned by `get_bspline_design_matrix`.

    Returns
    -------
    template : ndarray, shape (n_dets, n_samps)
    """
    template = harms_func(hwp_angle, modes, template_coeffs)  # (n_dets, n_samps)
    B = BSpline.design_matrix(timestamps, knots, degree, extrapolate=extrapolate)
    gain = (B @ gain_coeffs.T).T  # (n_dets, n_samps)
    return gain * template


def get_hwpss_gain_spline(aman, template_coeffs, modes, signal=None, hwp_angle=None,
                           timestamps=None, degree=3, n_knots=None, samples_per_knot=4000,
                           apply_prefilt=True, prefilt_cfg=None, prefilt_detrend='linear',
                           flags=None,
                           apodize_edges=True, apodize_edges_samps=1600,
                           apodize_flags=True, apodize_flags_samps=200, apo_type='C1',
                           merge_stats=True, hwpss_stats_name='hwpss_gain_stats_spline',
                           merge_model=True, hwpss_model_name='hwpss_model'):
    r"""
    Fits a single shared time-varying gain g(t), expanded on a B-spline
    basis, multiplying a FIXED constant-coefficient HWPSS template:

    .. math::
        \mathrm{signal}(t) = g(t) \cdot \sum_n a_n\sin{(\mathrm{modes}[n]\,\chi_\mathrm{hwp}(t))}
            + b_n\cos{(\mathrm{modes}[n]\,\chi_\mathrm{hwp}(t))}

    Unlike `get_hwpss_spline` (an independent time-varying amplitude per
    harmonic), this models a single common drift multiplying every harmonic
    identically -- appropriate when the HWPSS drift is a broadband gain-like
    effect (e.g. responsivity/loading drift) rather than harmonic-specific.
    Because there is only one spline "channel" (not one per harmonic), the
    fit has `n_bases` free parameters per detector instead of
    `2*n_modes*n_bases`, and the normal matrix is banded with a much
    smaller bandwidth (`degree`, vs `2*n_modes*(degree+1)-1` for
    `get_hwpss_spline`). The per-detector template also differs by
    construction (each detector has its own `template_coeffs`), so unlike
    `get_hwpss_spline` there is no shared-weight fast path: every detector
    is always solved individually (still cheap, since `n_bases` is small).

    Parameters
    ----------
    aman : AxisManager object
        The TOD to extract the gain-drift model from.
    template_coeffs : ndarray, shape (n_dets, 2*len(modes))
        Fixed constant per-harmonic sin/cos coefficients to scale by g(t),
        e.g. from `get_hwpss(...).coeffs` or an existing preprocess
        `hwpss_stats` field. Not refit here.
    modes : list of int
        The HWPSS harmonic modes corresponding to `template_coeffs`.
    signal, hwp_angle, timestamps, degree, n_knots, samples_per_knot,
    apply_prefilt, prefilt_cfg, prefilt_detrend, flags, apodize_edges,
    apodize_edges_samps, apodize_flags, apodize_flags_samps, apo_type,
    merge_stats, hwpss_stats_name, merge_model, hwpss_model_name :
        See `get_hwpss_spline` -- identical names/semantics/defaults.

    Returns
    -------
    hwpss_stats : AxisManager object
        - **coeffs** (n_dets x n_bases) : per-detector B-spline coefficients of g(t).
        - **template_coeffs** (n_dets x 2*n_modes) : the fixed template coefficients used.
        - **knots**, **degree** : as in `get_hwpss_spline`.
        - **sigma_tod**, **redchi2s** (n_dets) : as in `get_hwpss_spline`.
    """
    if prefilt_cfg is None:
        prefilt_cfg = {'type': 'sine2', 'cutoff': 1.0, 'trans_width': 1.0}

    prefilt = filters.get_hpf(prefilt_cfg)
    if signal is None:
        signal_name = 'signal'
        signal = aman[signal_name]
    elif isinstance(signal, str):
        signal_name = signal
        signal = aman[signal_name]
    elif isinstance(signal, np.ndarray):
        raise TypeError("Currently ndarray not supported, need update to tod_ops.fourier_filter module to remove signal_name argument.")
    else:
        raise TypeError("Signal must be None, str, or ndarray")

    if apply_prefilt:
        signal = np.array(
            tod_ops.fourier_filter(aman, prefilt, detrend=prefilt_detrend, signal_name=signal_name)
        )

    if hwp_angle is None:
        hwp_angle = aman.hwp_angle
    if timestamps is None:
        timestamps = aman.timestamps

    W = _get_hwpss_weights(aman, flags, apodize_edges, apodize_edges_samps,
                            apodize_flags, apodize_flags_samps, apo_type)

    B, knots = get_bspline_design_matrix(
        timestamps, n_knots=n_knots, samples_per_knot=samples_per_knot, degree=degree
    )
    n_dets = signal.shape[0]
    n_bases = B.shape[1]
    # Single spline "channel" (no per-harmonic interleaving): overlapping
    # basis functions differ by at most `degree`, so that's the bandwidth.
    bandwidth = degree
    template = harms_func(hwp_angle, modes, template_coeffs)  # (n_dets, n_samps)

    coeffs = np.zeros((n_dets, n_bases))
    dof_per_det = np.zeros(n_dets, dtype=int)
    n_degenerate = 0
    for d in range(n_dets):
        w_d = W if W.ndim == 1 else W[d]
        X_d = B.multiply(template[d][:, None])
        c_d, n_deg_d = _solve_spline_coeffs(X_d, w_d, signal[d], bandwidth, n_bases)
        coeffs[d] = c_d
        n_degenerate += n_deg_d
        dof_per_det[d] = max(int(np.count_nonzero(w_d > 0)) - n_bases, 1)

    if n_degenerate > 0:
        logger.warning(
            f'{n_degenerate} gain spline-basis coefficients have close to '
            'zero data support (fully flagged/apodized knot spans). Falling back '
            'to a pseudo-inverse solve for the affected detector(s).'
        )

    fitsig_tod = hwpss_gain_spline_func(timestamps, hwp_angle, modes, template_coeffs,
                                        coeffs, knots, degree=degree)

    sigma_tod = estimate_sigma_tod(signal, hwp_angle)
    resid = signal - fitsig_tod
    w_for_chi2 = W if W.ndim == 2 else W[None, :]
    redchi2s = np.sum(w_for_chi2 * resid**2, axis=1) / sigma_tod**2 / dof_per_det

    mode_names = []
    for mode in modes:
        mode_names.append(f'S{mode}')
        mode_names.append(f'C{mode}')

    hwpss_stats = core.AxisManager(
        aman.dets,
        core.LabelAxis(name='template_modes', vals=np.array(mode_names, dtype='<U3')),
        core.IndexAxis('spline_basis', count=n_bases),
        core.IndexAxis('knot_grid', count=len(knots)),
    )
    hwpss_stats.wrap('coeffs', coeffs, [(0, 'dets'), (1, 'spline_basis')])
    hwpss_stats.wrap('template_coeffs', np.asarray(template_coeffs), [(0, 'dets'), (1, 'template_modes')])
    hwpss_stats.wrap('knots', knots, [(0, 'knot_grid')])
    hwpss_stats.wrap('degree', degree)
    hwpss_stats.wrap('sigma_tod', sigma_tod, [(0, 'dets')])
    hwpss_stats.wrap('redchi2s', redchi2s, [(0, 'dets')])

    if merge_stats:
        aman.wrap(hwpss_stats_name, hwpss_stats)
    if merge_model:
        aman.wrap(hwpss_model_name, fitsig_tod, [(0, 'dets'), (1, 'samps')])
    return hwpss_stats


def get_hwp_freq(timestamps, hwp_angle):
    """
    Calculate the frequency of HWP rotation.

    Parameters:
    timestamps (array-like): An array of timestamps.
    hwp_angle (array-like): An array of HWP angles in radian

    Returns:
    float: The frequency of the HWP rotation in Hz.
    """
    hwp_freq = (np.sum(np.abs(np.diff(np.unwrap(hwp_angle)))) /
            (timestamps[-1] - timestamps[0])) / (2 * np.pi)
    return hwp_freq

def demod_tod(aman, signal=None, demod_mode=4,
              bpf_cfg=None, lpf_cfg=None, wrap=True):
    """
    Demodulate TOD based on HWP angle

    Parameters
    ----------
    aman : AxisManager
        The AxisManager object
    signal : str, optional
        Axis name of the signal to demodulate in aman. Default is 'signal'.
    demod_mode : int, optional
        Demodulation mode. Default is 4 (i.e. 4th harmonic of HWP).
    bpf_cfg : dict
        Configuration for Band-pass filter applied to the TOD data before demodulation.
        If not specified, a sine-squared bandwidth filter of
        (demod_mode * HWP speed) +/- 0.95*(HWP speed) is used with transition width 0.1.
        Example) bpf_cfg = {'type': 'sine2', 'center': 8.0, 'width': 3.8, 'trans_width': 0.1}
        or bpf_cfg = {'type': 'sine2', 'center': '4*f_HWP', 'width': '1.9*f_HWP', 'trans_width': 0.1}
        See filters.get_bpf for details.
    lpf_cfg : dict
        Configuration for Low-pass filter applied to the demodulated TOD data. If not specified,
        a sine-squared filter with a cutoff frequency of 0.95*(HWP speed) and transition width
        0.1 is used.
        Example) lpf_cfg = {'type': 'sine2', 'cutoff': 1.9, 'trans_width': 0.1}
        or lpf_cfg = {'type': 'sine2', 'cutoff': '0.95*f_HWP', 'trans_width': 0.1}
        See filters.get_lpf for details.
    wrap : bool, optional
        If True, the demodulated signal is wrapped and stored in the input aman container.
        If False, the demodulated signal is returned.

    Returns
    -------
    None
        The demodulated TOD data is added to the input `aman` container as new signals:
        'dsT' for the original signal filtered with `lpf`, 'demodQ' for the demodulated
        signal real component filtered with `lpf` and multiplied by 2, and 'demodU' for
        the demodulated signal imaginary component filtered with `lpf` and multiplied by 2.

    """
    if signal is None:
        #signal_name variable to be deleted when tod_ops.fourier_filter is updated
        signal_name = 'signal'
        signal = aman[signal_name]
    elif isinstance(signal, str):
        signal_name = signal
        signal = aman[signal_name]
    elif isinstance(signal, np.ndarray):
        raise TypeError("Currently ndarray not supported, need update to tod_ops.fourier_filter module to remove signal_name argument.")
    else:
        raise TypeError("Signal must be None, str, or ndarray")
    
    # HWP speed in Hz
    speed = get_hwp_freq(timestamps=aman.timestamps, hwp_angle=aman.hwp_angle)
    
    if bpf_cfg is None:
        bpf_center = demod_mode * speed
        bpf_width = speed * 2. * 0.95
        bpf_cfg = {'type': 'sine2',
                   'center': bpf_center,
                   'width': bpf_width,
                   'trans_width': 0.1}
    
    for k, v in bpf_cfg.items():
        if isinstance(v, str):
            if k in ['center', 'width']:
                bpf_cfg[k] = speed*float(v.split('*')[0])

    bpf = filters.get_bpf(bpf_cfg)
    
    if lpf_cfg is None:
        lpf_cutoff = speed * 0.95
        lpf_cfg = {'type': 'sine2',
                   'cutoff': lpf_cutoff,
                   'trans_width': 0.1}
    
    for k, v in lpf_cfg.items():
        if isinstance(v, str):
            if k == 'cutoff':
                lpf_cfg[k] = speed*float(v.split('*')[0])

    lpf = filters.get_lpf(lpf_cfg)

    # Create a RFFT object to reuse
    n = fft_ops.find_superior_integer(aman.samps.count)
    rfft = fft_ops.RFFTObj.for_shape(aman.dets.count, n, 'BOTH')

    phasor = np.empty_like(aman.hwp_angle, dtype=np.promote_types(signal.dtype, np.complex64))
    np.exp((demod_mode * 1j * aman.hwp_angle), out=phasor)
    demod = tod_ops.fourier_filter(aman, bpf, detrend=None,
                                   signal_name=signal_name, rfft=rfft) * 2. * phasor

    # Filter the demodulated signal
    demod_aman = core.AxisManager(aman.dets, aman.samps)
    demod_aman.wrap("timestamps", aman.timestamps, axis_map=[(0, 'samps')])
    demod_aman.wrap("dsT", aman[signal_name].copy(), axis_map=[(0, 'dets'), (1, 'samps')])
    demod_aman["dsT"][:] = tod_ops.fourier_filter(demod_aman, lpf, signal_name='dsT', detrend=None, rfft=rfft)
    demod_aman.wrap("demodQ", demod.real, axis_map=[(0, 'dets'), (1, 'samps')])
    demod_aman["demodQ"][:] = tod_ops.fourier_filter(demod_aman, lpf, signal_name="demodQ", detrend=None, rfft=rfft)
    demod_aman.wrap("demodU", demod.imag, axis_map=[(0, 'dets'), (1, 'samps')])
    demod_aman["demodU"][:] = tod_ops.fourier_filter(demod_aman, lpf, signal_name="demodU", detrend=None, rfft=rfft)

    # Destroy the RFFT object
    del rfft

    # Either wrap or return the demodulated signal
    if wrap:
        for fld in ['dsT', 'demodQ', 'demodU']:
            aman.wrap(fld, demod_aman[fld], axis_map=[(0, 'dets'), (1, 'samps')])
        del demod_aman
    else:
        return demod_aman['dsT'], demod_aman['demodQ'], demod_aman['demodU']


def tau_model(fhwp, tau, AQ, AU, mode):
    return (AQ + 1j*AU) / (1 - 1j* mode * 2 * np.pi * tau * fhwp )


def get_tau_hwp(
    aman,
    signal='signal',
    lpf_cfg=None,
    bpf_cfg=None,
    width=1000,
    apodize_samps=2000,
    apo_type='C1',
    trim_samps=2000,
    min_fhwp=1.,
    max_fhwp=2.,
    demod_mode=4,
    wn=None,
    flags=None,
    full_output=False,
    merge=False,
    name='tau_hwp'
):
    """
    Analyze observation with hwp spinning up or spinning down and
    compute the timeconstant of detectors.
    This demoulate tod and estimate timeconstant from hwp rotation
    speed depdndence of half-wave plate synchronous signal.

    Parameters
    ----------
    aman : AxisManager
        AxisManager object containing the TOD data.
    signal: std optional
        Name of signal to process. Default is `signal`.
    lpf_cfg: dict optional
        Configuration for Low-pass filter applied before demodulation.
    bpf_cfg: dict optional
        Configuration for Band-pass filter applied before demodulation.
    width: int optional
        width of single section of TOD.
    apodize_samps: int optional
        Number of samples on tod ends to apodize.
    apo_type: str, optional
        Type of apodization, default is C1. See tod_ops.apodize for all options.
    trim_samps: int optional
        Number of samples on tod ends to trim.
    min_fhwp: float optional
        Mininum rotation frequency of hwp to be used for calculation.
    max_fhwp: float optional
        Maximum rotation frequency of hwp to be used for calculation.
    demod_mode: int, optional
        Demodulation mode. Default is 4.
    flags: str optional
        Name of flags in `aman.flags` to use for fitting.
    wn: str or None
        Precomputed white noise level of signal to be used for weights of
        fitting. If none, the fitting weights will be 1.
    full_output: bool optional
        Whether to output all the statistics used for fitting.
    merge: bool optional
        Whether to merge calculated statistics to `aman`. Default is False.
    name: str optional
        Name under which to wrap the calculated statistics.
        Default is 'tau_hwp'.

    Returns
    -------
    result : AxisManager
        An AxisManager containing time constants, their errors, and reduced
        chi-squared statistics.
    """

    if lpf_cfg is None:
        lpf_cfg = {'type': 'sine2',
                   'cutoff': min_fhwp * 0.95,
                   'trans_width': 0.1}
    if bpf_cfg is None:
        # no band pass filter
        bpf_cfg = {'type': 'sine2',
                   'center': 0,
                   'width': 200,
                   'trans_width': 0.1}

    hwp_rate = np.gradient(np.unwrap(aman.hwp_angle) * 200 / 2 / np.pi)
    if 'hwp_rate' not in aman:
        aman.wrap('hwp_rate', hwp_rate, [(0, 'samps')])

    if wn is None:
        # Calculate white noise from section with low hwp speed.
        idx = np.where(abs(hwp_rate) < .5)[0]
        s = aman.samps.offset + idx[0]
        e = aman.samps.offset + idx[-1]
        aman_short = aman.restrict('samps', (s, e), in_place=False)
        freqs, Pxx, nseg = fft_ops.calc_psd(aman_short,
                                            signal=aman_short.get(signal),
                                            noverlap=0,
                                            full_output=True)
        wn = fft_ops.calc_wn(aman_short, pxx=Pxx, freqs=freqs, nseg=nseg)

    tod_ops.detrend_tod(aman, signal_name=signal, method="median")
    tod_ops.apodize.apodize_cosine(aman, signal_name=signal,
                                   apodize_samps=apodize_samps, apo_type=apo_type)
    demod_tod(aman, signal=signal, demod_mode=demod_mode,
              lpf_cfg=lpf_cfg, bpf_cfg=bpf_cfg, wrap=True)

    idx = np.where((min_fhwp < abs(hwp_rate)) & (abs(hwp_rate) < max_fhwp))[0]
    s = idx[0] if trim_samps < idx[0] else trim_samps
    e = idx[-1] if trim_samps < aman.samps.count - idx[-1] \
        else aman.samps.count - trim_samps
    s += aman.samps.offset
    e += aman.samps.offset

    aman.restrict('samps', (s, e), in_place=True)

    nsections = int(aman.samps.count / width)
    timestamps = np.zeros(nsections)
    hwp_rate = np.zeros(nsections)
    demodQ = np.zeros((nsections, aman.dets.count))
    demodU = np.zeros((nsections, aman.dets.count))
    weights = np.zeros((nsections, aman.dets.count))
    mask = np.ones((nsections, aman.dets.count), dtype=bool)
    if isinstance(flags, str) and (flags in aman.flags):
        flags = ~aman.flags[flags].mask()
        if flags.ndim == 1:
            for i in range(nsections):
                msk = np.zeros(aman.samps.count, dtype=bool)
                msk[i*width:(i+1)*width] = flags[i*width:(i+1)*width]
                timestamps[i] = np.median(aman.timestamps[msk])
                hwp_rate[i] = np.median(aman.hwp_rate[msk])
                demodQ[i] = np.median(aman.demodQ[:, msk], axis=1)
                demodU[i] = np.median(aman.demodU[:, msk], axis=1)
                count = np.sum(msk)
                weights[i] = np.sqrt(count/200)/wn
                mask[i, :] = False if count == 0 else True
        elif flags.ndim == 2:
            for i in range(nsections):
                timestamps[i] = np.median(aman.timestamps[i*width:(i+1)*width])
                hwp_rate[i] = np.median(aman.hwp_rate[i*width:(i+1)*width])
                for j in range(aman.dets.count):
                    msk = np.zeros(aman.samps.count, dtype=bool)
                    msk[i*width:(i+1)*width] = flags[j, i*width:(i+1)*width]
                    demodQ[i, j] = np.median(aman.demodQ[j, msk])
                    demodU[i, j] = np.median(aman.demodU[j, msk])
                    count = np.sum(msk)
                    weights[i, j] = np.sqrt(count/200)/wn[j]
                    mask[i, j] = False if count == 0 else True
    else:
        for i in range(nsections):
            timestamps[i] = np.median(aman.timestamps[i*width:(i+1)*width])
            hwp_rate[i] = np.median(aman.hwp_rate[i*width:(i+1)*width])
            demodQ[i] = np.median(aman.demodQ[:, i*width:(i+1)*width], axis=1)
            demodU[i] = np.median(aman.demodU[:, i*width:(i+1)*width], axis=1)
            weights[i] = np.sqrt(width/200)/wn

    logger.debug('Fit time constant')
    AQ = np.full(aman.dets.count, np.nan)
    AQ_error = np.full(aman.dets.count, np.nan)
    AU = np.full(aman.dets.count, np.nan)
    AU_error = np.full(aman.dets.count, np.nan)
    tau = np.full(aman.dets.count, np.nan)
    tau_error = np.full(aman.dets.count, np.nan)
    redchi2s = np.full(aman.dets.count, np.nan)
    for i in range(aman.dets.count):
        try:
            model = LmfitModel(tau_model, independent_vars=['fhwp'])
            model.set_param_hint('mode', vary=False)
            params = model.make_params(
                tau=1e-3,
                AQ=np.median(demodQ[:, i][mask[:, i]]),
                AU=np.median(demodU[:, i][mask[:, i]]),
                mode=demod_mode,
            )
            fit = model.fit(
                data=demodQ[:, i][mask[:, i]] + 1j * demodU[:, i][mask[:, i]],
                params=params,
                fhwp=hwp_rate[mask[:, i]],
                weights=weights[:, i][mask[:, i]]
            )
            AQ[i] = fit.params['AQ'].value
            AQ_error[i] = fit.params['AQ'].stderr
            AU[i] = fit.params['AU'].value
            AU_error[i] = fit.params['AU'].stderr
            tau[i] = fit.params['tau'].value
            tau_error[i] = fit.params['tau'].stderr
            redchi2s[i] = fit.redchi
        except Exception:
            logger.debug(f'Failed to fit {aman.dets.vals[i]}')

    if full_output:
        sections = core.OffsetAxis('sections', count=nsections,
                                   offset=0)
        result = core.AxisManager(aman.dets, sections)
        result.wrap('timestamps', timestamps, axis_map=[(0, 'sections')])
        result.wrap('hwp_rate', hwp_rate, axis_map=[(0, 'sections')])
        result.wrap('demodQ', np.transpose(demodQ),
                    axis_map=[(0, 'dets'), (1, 'sections')])
        result.wrap('demodU', np.transpose(demodU),
                    axis_map=[(0, 'dets'), (1, 'sections')])
        result.wrap('weights', np.transpose(weights),
                    axis_map=[(0, 'dets'), (1, 'sections')])
        result.wrap('mask', np.transpose(mask),
                    axis_map=[(0, 'dets'), (1, 'sections')])
    else:
        result = core.AxisManager(aman.dets)
    result.wrap('AQ', AQ, axis_map=[(0, 'dets')])
    result.wrap('AQ_error', AQ_error, axis_map=[(0, 'dets')])
    result.wrap('AU', AU, axis_map=[(0, 'dets')])
    result.wrap('AU_error', AU_error, axis_map=[(0, 'dets')])
    result.wrap('tau_hwp', tau, axis_map=[(0, 'dets')])
    result.wrap('tau_hwp_error', tau_error, axis_map=[(0, 'dets')])
    result.wrap('redchi2s', redchi2s, axis_map=[(0, 'dets')])

    if merge:
        aman.wrap(name, result)

    return result
