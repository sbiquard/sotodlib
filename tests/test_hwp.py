# Copyright (c) 2021 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Check tod_ops routines.

"""

import unittest
import numpy as np

from sotodlib import core
from sotodlib.hwp import hwp

INPUT_COEFFS = np.asarray(
                [[14.4086668, 10.57294344, 11.18964606, 12.27585103,
                 11.27456133, 11.7841095, 12.87044498, 13.25598597,
                 12.56675035, 12.06382477, 14.72865531, 12.62258284,
                 11.43530194, 11.92264568, 14.71371932, 11.8379756],
                [10.32748604, 11.82000625, 14.96949369, 14.18303644,
                 13.45385166, 12.81443907, 13.65328508, 13.78361639,
                 11.3964662, 13.06204199, 10.35769728, 10.20522442,
                 14.79385459, 11.24546604, 10.27690037, 10.04351162]])


def make_fake_tod(ts=np.arange(0, 60, 1/200), modes=np.arange(8)+1,
                  mult_fact=12.5*np.ones(8), ndets=2, input_coeffs=None):
    hwp_angle = (2*np.pi*2*ts) % (2*np.pi)
    fake_signal = np.zeros((ndets, len(hwp_angle)))
    if input_coeffs is None:
        input_coeffs = np.zeros((ndets, 2*len(mult_fact)))
    for nd in range(ndets):
        for i, md in enumerate(modes):
            if input_coeffs is None:
                input_coeffs[nd, 2*i] = mult_fact[i]*np.random.rand()
                input_coeffs[nd, 2*i+1] = mult_fact[i]*np.random.rand()
            fake_signal[nd] += input_coeffs[nd, 2*i]*np.sin(md*hwp_angle)
            fake_signal[nd] += input_coeffs[nd, 2*i + 1]*np.cos(md*hwp_angle)

    dets = ['det%i' % i for i in range(ndets)]
    mode_names = []
    for mode in modes:
        mode_names.append(f'S{mode}')
        mode_names.append(f'C{mode}')

    tod = core.AxisManager(core.LabelAxis('dets', vals=dets),
                           core.OffsetAxis('samps', count=len(hwp_angle)),
                           core.LabelAxis('modes', vals=mode_names))
    tod.wrap('timestamps', ts, axis_map=[(0, 'samps')])
    tod.wrap('hwp_angle', hwp_angle, axis_map=[(0, 'samps')])
    tod.wrap('signal', np.atleast_2d(fake_signal),
             axis_map=[(0, 'dets'), (1, 'samps')])
    tod.wrap('input_coeffs', np.atleast_2d(input_coeffs),
             axis_map=[(0, 'dets'), (1, 'modes')])
    return tod

def get_coeff_metric(tod):
    nm = tod.modes.count//2
    nd = tod.dets.count
    outmetric_num = np.zeros((nd, nm))
    outmetric_denom = np.zeros((nd, nm))
    for i in range(nm):
        outmetric_num[:, i] = (tod.hwpss_stats.coeffs[:, 2*i] - \
                               tod.input_coeffs[: , 2*i])**2
        outmetric_num[:, i] += (tod.hwpss_stats.coeffs[:, 2*i+1] - \
                                tod.input_coeffs[: , 2*i+1])**2
        outmetric_denom[:, i] = (tod.input_coeffs[:, 2*i])**2
        outmetric_denom[:, i] += (tod.input_coeffs[:, 2*i+1])**2
    return np.max(100*(outmetric_num/outmetric_denom))



class HwpssTest(unittest.TestCase):
    "Test the HWPSS fitting functions"
    def test_linregbin(self):
        lr, bn = True, True
        tod = make_fake_tod(input_coeffs=INPUT_COEFFS)
        _ = hwp.get_hwpss(tod, lin_reg=lr, bin_signal=bn,
                           bins=200)
        ommax = get_coeff_metric(tod)
        self.assertTrue(ommax < 0.1)

    def test_linregnobin(self):
        lr, bn = True, False
        tod = make_fake_tod(input_coeffs=INPUT_COEFFS)
        _ = hwp.get_hwpss(tod, lin_reg=lr, bin_signal=bn)
        ommax = get_coeff_metric(tod)
        self.assertTrue(ommax < 0.1)

    def test_fitbin(self):
        lr, bn = False, True
        tod = make_fake_tod(input_coeffs=INPUT_COEFFS)
        _ = hwp.get_hwpss(tod, lin_reg=lr, bin_signal=bn, bins=200)
        ommax = get_coeff_metric(tod)
        self.assertTrue(ommax < 0.1)
        # Checks that returned covariance matrix from fit is finite.
        # When not using wrapper function in lambda function we get
        # infinite covariance matrix returned. So this is a warning
        # to not change that even though it looks wrong.
        self.assertFalse(False in np.isfinite(tod.hwpss_stats.covars))

    def test_fitnobin(self):
        lr, bn = False, False
        tod = make_fake_tod(input_coeffs=INPUT_COEFFS)
        with self.assertRaises(ValueError):
            _ = hwp.get_hwpss(tod, lin_reg=lr, bin_signal=bn, modes=[2, 4])


def make_fake_tod_spline(ts=np.arange(0, 600, 1/200), modes=(2, 4), ndets=3,
                         degree=3, n_knots=6, coeff_scale=12.5, seed=0):
    """Build a synthetic AxisManager whose signal is exactly expressible in
    the B-spline x harmonic basis that `get_hwpss_spline` fits, by generating
    the signal via `hwp.hwpss_spline_func` itself.
    """
    rng = np.random.default_rng(seed)
    hwp_angle = (2*np.pi*2*ts) % (2*np.pi)
    _, knots = hwp.get_bspline_design_matrix(ts, n_knots=n_knots, degree=degree)
    n_bases = len(knots) - degree - 1
    true_coeffs = coeff_scale * (rng.random((ndets, 2*len(modes), n_bases)) - 0.5)
    signal = hwp.hwpss_spline_func(ts, hwp_angle, list(modes), true_coeffs, knots, degree=degree)

    dets = ['det%i' % i for i in range(ndets)]
    tod = core.AxisManager(core.LabelAxis('dets', vals=dets),
                           core.OffsetAxis('samps', count=len(ts)))
    tod.wrap('timestamps', ts, axis_map=[(0, 'samps')])
    tod.wrap('hwp_angle', hwp_angle, axis_map=[(0, 'samps')])
    tod.wrap('signal', signal, axis_map=[(0, 'dets'), (1, 'samps')])
    return tod, true_coeffs, knots


class HwpssSplineTest(unittest.TestCase):
    "Test the B-spline x HWP-harmonic HWPSS fitting functions"

    def test_spline_exact_recovery(self):
        """Noiseless signal built from the exact fitted basis: template and
        coefficients should recover to near machine precision."""
        tod, true_coeffs, _ = make_fake_tod_spline()
        modes = [2, 4]
        signal = tod.signal.copy()
        hwp.get_hwpss_spline(tod, modes=modes, degree=3, n_knots=6, apply_prefilt=False)
        np.testing.assert_allclose(tod.hwpss_model, signal, atol=1e-6)
        np.testing.assert_allclose(tod.hwpss_stats_spline.coeffs, true_coeffs, atol=1e-6)

    def test_spline_with_flags(self):
        """A fully-flagged span wiping out a knot should log a warning and
        fall back to a finite (pinv) solution, not NaN/inf."""
        from so3g.proj import Ranges
        tod, _, _ = make_fake_tod_spline(n_knots=100)
        r = Ranges.from_array(np.array([[20000, 100000]], dtype='int32'), tod.samps.count)
        flags = core.FlagManager.for_tod(tod)
        flags.wrap('cal_stop', r)
        tod.wrap('flags', flags)

        with self.assertLogs('sotodlib.hwp.hwp', level='WARNING'):
            hwp.get_hwpss_spline(tod, modes=[2, 4], degree=3, n_knots=100,
                                 flags='cal_stop', apodize_flags=True,
                                 apodize_flags_samps=50, apply_prefilt=False)
        self.assertTrue(np.all(np.isfinite(tod.hwpss_model)))

    def test_spline_per_det_flags(self):
        """Detectors flagged at different times: each should recover its own
        coefficients, unaffected by other detectors' flags."""
        from so3g.proj import Ranges, RangesMatrix
        modes = [2, 4]
        degree = 3
        n_knots = 10
        tod, true_coeffs, _ = make_fake_tod_spline(modes=modes, ndets=4, degree=degree, n_knots=n_knots)
        ndets = tod.dets.count
        ranges = []
        for d in range(ndets):
            start = 20000 + d * 30000
            ranges.append(Ranges.from_array(
                np.array([[start, start + 10000]], dtype='int32'), tod.samps.count))
        flags = core.FlagManager.for_tod(tod)
        flags.wrap('glitches', RangesMatrix(ranges))
        tod.wrap('flags', flags)

        hwp.get_hwpss_spline(tod, modes=modes, degree=degree, n_knots=n_knots,
                             flags='glitches', apodize_flags=True,
                             apodize_flags_samps=100, apply_prefilt=False)
        self.assertTrue(np.all(np.isfinite(tod.hwpss_model)))
        err_per_det = np.max(np.abs(tod.hwpss_stats_spline.coeffs - true_coeffs), axis=(1, 2))
        self.assertTrue(np.all(err_per_det < 1e-5))


class HwpssGainSplineTest(unittest.TestCase):
    "Test the shared gain-drift B-spline HWPSS fitting functions"

    def test_gain_spline_exact_recovery(self):
        """Noiseless signal built from a fixed template x a known gain
        spline: template/gain coefficients should recover to near machine
        precision, using far fewer parameters than the per-harmonic
        spline (one spline channel total, not one per harmonic)."""
        rng = np.random.default_rng(0)
        ts = np.arange(0, 600, 1/200)
        hwp_angle = (2*np.pi*2*ts) % (2*np.pi)
        modes = [1, 2, 3, 4, 5, 6, 7, 8]
        degree = 3
        n_knots = 10
        ndets = 3

        template_coeffs = 12.5 * (rng.random((ndets, 2*len(modes))) - 0.5)
        _, knots = hwp.get_bspline_design_matrix(ts, n_knots=n_knots, degree=degree)
        n_bases = len(knots) - degree - 1
        true_gain_coeffs = 1.0 + 0.3 * (rng.random((ndets, n_bases)) - 0.5)
        signal = hwp.hwpss_gain_spline_func(ts, hwp_angle, modes, template_coeffs,
                                            true_gain_coeffs, knots, degree=degree)

        dets = ['det%i' % i for i in range(ndets)]
        tod = core.AxisManager(core.LabelAxis('dets', vals=dets),
                               core.OffsetAxis('samps', count=len(ts)))
        tod.wrap('timestamps', ts, axis_map=[(0, 'samps')])
        tod.wrap('hwp_angle', hwp_angle, axis_map=[(0, 'samps')])
        tod.wrap('signal', signal, axis_map=[(0, 'dets'), (1, 'samps')])

        hwp.get_hwpss_gain_spline(tod, template_coeffs=template_coeffs, modes=modes,
                                  degree=degree, n_knots=n_knots, apply_prefilt=False)
        np.testing.assert_allclose(tod.hwpss_model, signal, atol=1e-6)
        np.testing.assert_allclose(tod.hwpss_gain_stats_spline.coeffs, true_gain_coeffs, atol=1e-6)

    def test_gain_spline_per_det_flags(self):
        """Detectors flagged at different times: each should recover its own
        gain coefficients, unaffected by other detectors' flags."""
        from so3g.proj import Ranges, RangesMatrix
        rng = np.random.default_rng(0)
        ts = np.arange(0, 600, 1/200)
        hwp_angle = (2*np.pi*2*ts) % (2*np.pi)
        modes = [1, 2, 3, 4, 5, 6, 7, 8]
        degree = 3
        n_knots = 10
        ndets = 4

        template_coeffs = 12.5 * (rng.random((ndets, 2*len(modes))) - 0.5)
        _, knots = hwp.get_bspline_design_matrix(ts, n_knots=n_knots, degree=degree)
        n_bases = len(knots) - degree - 1
        true_gain_coeffs = 1.0 + 0.3 * (rng.random((ndets, n_bases)) - 0.5)
        signal = hwp.hwpss_gain_spline_func(ts, hwp_angle, modes, template_coeffs,
                                            true_gain_coeffs, knots, degree=degree)

        dets = ['det%i' % i for i in range(ndets)]
        tod = core.AxisManager(core.LabelAxis('dets', vals=dets),
                               core.OffsetAxis('samps', count=len(ts)))
        tod.wrap('timestamps', ts, axis_map=[(0, 'samps')])
        tod.wrap('hwp_angle', hwp_angle, axis_map=[(0, 'samps')])
        tod.wrap('signal', signal, axis_map=[(0, 'dets'), (1, 'samps')])

        ranges = []
        for d in range(ndets):
            start = 20000 + d * 30000
            ranges.append(Ranges.from_array(
                np.array([[start, start + 10000]], dtype='int32'), tod.samps.count))
        flags = core.FlagManager.for_tod(tod)
        flags.wrap('glitches', RangesMatrix(ranges))
        tod.wrap('flags', flags)

        hwp.get_hwpss_gain_spline(tod, template_coeffs=template_coeffs, modes=modes,
                                  degree=degree, n_knots=n_knots, flags='glitches',
                                  apodize_flags=True, apodize_flags_samps=100,
                                  apply_prefilt=False)
        self.assertTrue(np.all(np.isfinite(tod.hwpss_model)))
        err_per_det = np.max(np.abs(tod.hwpss_gain_stats_spline.coeffs - true_gain_coeffs), axis=1)
        self.assertTrue(np.all(err_per_det < 1e-5))


if __name__ == '__main__':
    unittest.main()
