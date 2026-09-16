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

    def test_spline_explicit_knots_reproduce_fit(self):
        """Passing an explicit knot vector must reproduce the fit that built
        it. The sim branch of subtract_hwpss_spline relies on this to apply the
        same filter to a simulation as was applied to the data, since
        samples_per_knot is not recoverable from the archived stats.
        """
        tod, _, _ = make_fake_tod_spline(n_knots=6)
        kw = dict(modes=[2, 4], degree=3, apply_prefilt=False,
                  merge_stats=False, merge_model=False)
        a = hwp.get_hwpss_spline(tod, n_knots=6, **kw)
        b = hwp.get_hwpss_spline(tod, knots=a.knots, **kw)
        np.testing.assert_array_equal(a.knots, b.knots)
        np.testing.assert_array_equal(a.coeffs, b.coeffs)

        # a different grid really is a different fit, so the above is not
        # trivially true
        c = hwp.get_hwpss_spline(tod, n_knots=20, **kw)
        self.assertNotEqual(a.coeffs.shape[-1], c.coeffs.shape[-1])

    def test_spline_stats_coexist_with_hwpss_stats(self):
        """The spline stats must not collide with the standard HWPSS stats
        when both are wrapped into one AxisManager, as the preprocessing
        pipeline does. They carry different numbers of modes, so sharing an
        axis name silently truncates or fails to broadcast.
        """
        tod, _, _ = make_fake_tod_spline()
        hwp.get_hwpss(tod, modes=np.arange(8) + 1, bin_signal=False)
        hwp.get_hwpss_spline(tod, modes=[2, 4], degree=3, n_knots=6,
                             apply_prefilt=False,
                             hwpss_model_name='hwpss_model_spline')

        proc = core.AxisManager(tod.dets, tod.samps)
        proc.wrap('post_hwpss_stats', tod.hwpss_stats)
        proc.wrap('post_hwpss_stats_spline', tod.hwpss_stats_spline)

        self.assertEqual(proc.post_hwpss_stats.coeffs.shape[1], 16)
        self.assertEqual(proc.post_hwpss_stats_spline.coeffs.shape[1], 4)


class HWPSSSplinePipelineTest(unittest.TestCase):
    """The merged hwpss_spline preprocess step, across its three paths."""

    CFG = [{'name': 'hwpss_spline', 'skip_on_sim': False,
            'calc': {'modes': [2, 4], 'degree': 3, 'n_knots': 6,
                     'apply_prefilt': False,
                     'hwpss_stats_name': 'stats_spline', 'merge_model': False},
            'save': True,
            'process': {'subtract': True, 'subtract_name': 'signal',
                        'hwpss_model_name': 'hwpss_model_spline'}}]

    def _tod(self, **kw):
        tod, true_coeffs, _ = make_fake_tod_spline(n_knots=6, **kw)
        # Pipeline.run reads this to work out filter cutoffs
        tod.wrap('iir_params', core.AxisManager())
        return tod, true_coeffs

    def test_calc_pass_subtracts_and_archives(self):
        """On the archive-generating pass the step must both fit and subtract.
        Pipeline.run calls process() before calc_and_save(), so a step that
        only subtracts from archived stats would silently leave the HWPSS in
        for every later step of that run.
        """
        from sotodlib.preprocess.pcore import Pipeline
        tod, true_coeffs = self._tod()
        before = np.std(tod.signal)
        proc_aman, success = Pipeline(self.CFG).run(tod)
        self.assertEqual(success, 'end')
        self.assertIn('stats_spline', proc_aman)
        np.testing.assert_allclose(proc_aman.stats_spline.coeffs, true_coeffs,
                                   atol=1e-6)
        self.assertLess(np.std(tod.signal), 1e-6 * before)

    def test_load_pass_uses_archived_coeffs(self):
        """Re-running with saved stats must subtract without refitting."""
        from sotodlib.preprocess.pcore import Pipeline
        pipe = Pipeline(self.CFG)
        tod, _ = self._tod()
        proc_aman, _ = pipe.run(tod)

        tod2, _ = self._tod()
        before = np.std(tod2.signal)
        pipe.run(tod2, proc_aman)
        self.assertLess(np.std(tod2.signal), 1e-6 * before)

    def test_sim_pass_refits_on_the_sim(self):
        """On sims the archived coefficients describe the data, so the fit is
        redone on the sim -- a different realization must still be removed.
        """
        from sotodlib.preprocess.pcore import Pipeline
        pipe = Pipeline(self.CFG)
        tod, _ = self._tod()
        proc_aman, _ = pipe.run(tod)

        sim, _ = self._tod(seed=7)
        before = np.std(sim.signal)
        pipe.run(sim, proc_aman, sim=True)
        self.assertLess(np.std(sim.signal), 1e-6 * before)


if __name__ == '__main__':
    unittest.main()
