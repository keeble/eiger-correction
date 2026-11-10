"""Small synthetic tests: no beamline files or large detector reads required."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np

import short_scan_tools as tools


class ShortScanTests(unittest.TestCase):
    def test_fixed_step_grid(self):
        limits, npt, edges = tools.fixed_theta_grid(87.123, 0.05)
        self.assertEqual(npt, 1743)
        self.assertAlmostEqual(limits[1], 87.15)
        np.testing.assert_allclose(np.diff(edges), 0.05, rtol=0, atol=2e-14)
        with self.assertRaises(ValueError):
            tools.fixed_theta_grid(180, 0.07)

    def test_grouping_is_minimum_anchored(self):
        angles = np.array([0.0016, 0.0, 0.0008, 0.0, 2.0])
        groups = tools.angle_groups(angles, 0.001)
        self.assertEqual([g.tolist() for g in groups], [[1, 2, 3], [0], [4]])

    def test_selection_and_stop_jitter(self):
        angles = np.repeat(np.array([0., 2., 4.00003]), 88)
        groups = tools.angle_groups(angles)
        for mode in ("first", "spread", "random"):
            cut = tools.Cut("quarter", frame_fraction=0.25, frame_mode=mode, motor_max=4)
            chosen = tools.select_groups(angles, groups, cut)
            self.assertEqual([len(x) for x in chosen.values()], [22, 22, 22])
            for i, members in chosen.items():
                self.assertEqual(len(np.unique(members)), 22)
                self.assertTrue(np.isin(members, groups[i]).all())
                np.testing.assert_array_equal(members, tools.select_groups(angles, groups, cut)[i])
        stride = tools.select_groups(angles, groups, tools.Cut("odd", angle_stride=2, angle_offset=1))
        self.assertEqual(list(stride), [1])

    def test_low_motor_protection_preserves_stride_phase(self):
        angles = np.repeat([0., 8.75, 13.26, 16.91, 20., 23., 26.], 4)
        groups = tools.angle_groups(angles)
        for offset in range(3):
            cut = tools.Cut("protected", angle_stride=3, angle_offset=offset,
                            keep_motor_below=15., frame_fraction=0.5)
            selection = tools.select_groups(angles, groups, cut)
            expected = sorted(set(range(3)) | set(range(offset, len(groups), 3)))
            self.assertEqual(list(selection), expected)
            self.assertTrue(all(len(m) == 2 for m in selection.values()))
        bounded = tools.Cut("bounded", angle_stride=3, angle_offset=2,
                            keep_motor_below=15., motor_max=10.)
        self.assertEqual(list(tools.select_groups(angles, groups, bounded)), [0, 1])
        with self.assertRaises(ValueError):
            tools.Cut("invalid", keep_motor_below=np.nan)

    def test_streamed_counts_i0_geometry_and_invalid_union(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "raw.h5"
            frames = np.array([[[1., 2.], [3., 4.]], [[10., -1.], [30., 40.]]])
            with h5py.File(path, "x") as file:
                file["data"] = frames
            scan = SimpleNamespace(path=path, detector_path=path, detector_dataset="data",
                                   angles=np.array([0., 0.0008]), i0=np.array([1., 3.]),
                                   native_mask=np.zeros((2, 2), bool), shape=(2, 2))
            calibration = {"base_mask": scan.native_mask, "bad_pixel_mask": scan.native_mask,
                           "combined_divisor": np.ones((2, 2))}
            calls = []

            def fake_integrate(model, counts, angle, monitor, mask, divisor, npt, radial_range, **kwargs):
                calls.append((counts.copy(), angle, monitor, kwargs))
                return SimpleNamespace(radial=np.array([0.5, 1.5]), sum_signal=np.array([2., 4.]),
                                       sum_normalization=np.ones(2), sum_variance=np.array([2., 4.]),
                                       count=np.ones(2))

            cuts = [tools.Cut("full"), tools.Cut("same"), tools.Cut("half", frame_fraction=0.5)]
            with patch.object(tools.workflow, "integrate_frame", side_effect=fake_integrate):
                patterns, selections = tools.integrate_cuts(
                    scan, calibration, None, [np.array([0, 1])], cuts, (0., 2.), npt=2, progress=None)
            self.assertEqual(len(calls), 2)  # Identical subsets integrate only once.
            self.assertTrue(np.isnan(calls[0][0][0, 1]))
            self.assertAlmostEqual(calls[0][1], 0.0004)  # NOT I0-weighted.
            self.assertEqual(calls[0][2], 4.)
            np.testing.assert_equal(calls[1][0], frames[0])
            self.assertEqual(calls[1][2], 1.)
            self.assertEqual(calls[0][3]["frame_gain"], 1.)
            saved = tools.save_experiment(Path(temp) / "out", scan, path, cuts, patterns, selections, {})
            self.assertTrue((saved / "patterns.h5").exists())
            with self.assertRaises(FileExistsError):
                tools.save_experiment(saved, scan, path, cuts, patterns, selections, {})

    def test_gap_intervals(self):
        self.assertEqual(tools.missing_intervals(
            [True, True, False, True, True], [False, False, False, True, False], np.arange(6.)),
            [(0., 2.), (4., 5.)])
        self.assertEqual(tools.missing_intervals([False], [False], [0., 1.]), [])

    def test_coverage_uses_real_geometry_and_masks(self):
        scan = SimpleNamespace(angles=np.array([0., 3., 10.]),
                               shape=(1, 2), native_mask=np.array([[False, True]]))
        calibration = {"base_mask": np.zeros((1, 2), bool),
                       "bad_pixel_mask": np.zeros((1, 2), bool)}
        geometry = SimpleNamespace(get_ai=lambda angle: SimpleNamespace(
            center_array=lambda shape, unit: np.deg2rad([[angle + 0.5, 5.5]])))
        cuts = [tools.Cut("full"), tools.Cut("every_2", angle_stride=2)]
        coverage = tools.geometry_coverage(scan, calibration, SimpleNamespace(geometry=geometry),
            [np.array([i]) for i in range(3)], cuts, np.arange(12.))
        self.assertEqual(coverage["full"]["positions"].sum(), 3)
        self.assertEqual(coverage["every_2"]["positions"].sum(), 2)
        self.assertEqual(coverage["full"]["pixels"][5], 0)  # Masked, not envelope-filled.
        self.assertEqual(tools.missing_intervals(coverage["full"]["pixels"] > 0,
            coverage["every_2"]["pixels"] > 0, np.arange(12.)), [(3., 4.)])

    def test_self_calibration_uses_only_subset_and_own_gains(self):
        expected_gain = 1.02
        with TemporaryDirectory() as temp:
            path = Path(temp) / "raw.h5"
            frames = np.array([[[1., 2.], [3., 4.]], [[10., 20.], [30., 40.]]])
            with h5py.File(path, "x") as file:
                file["data"] = frames
            scan = tools.workflow.Scan(
                path, path, "data", np.array([0., 0.0008]), np.array([1., 3.]),
                np.zeros((2, 2), bool), 70., 0.001, "CdTe", 1, 1e-11, (2, 2),
            )
            fit = {
                "combined_divisor": np.full((2, 2), 1.03),
                "bad_pixel_mask": np.array([[True, False], [False, False]]),
                "frame_gains": np.array([1.02]),
            }

            def fake_cache(subset, model, base_mask, q_range, npt, saturation):
                with h5py.File(subset.detector_path, "r") as file:
                    np.testing.assert_equal(file["data"][0], frames[0])
                np.testing.assert_equal(subset.i0, [1.])
                np.testing.assert_equal(subset.angles, [0.])
                self.assertFalse(base_mask.any())
                return {0: {}}

            def fake_integrate(model, counts, angle, monitor, mask, divisor, npt, radial_range, **kwargs):
                np.testing.assert_equal(divisor, fit["combined_divisor"])
                np.testing.assert_equal(mask, fit["bad_pixel_mask"])
                self.assertEqual(kwargs["frame_gain"], expected_gain)
                return SimpleNamespace(
                    radial=np.array([0.5]), sum_signal=np.array([2.]),
                    sum_normalization=np.ones(1), sum_variance=np.array([2.]), count=np.ones(1),
                )

            with patch.object(tools.workflow, "detector_layout", return_value=(np.zeros((2, 2), bool), np.zeros((2, 2), int))), \
                 patch.object(tools.workflow, "build_cache", side_effect=fake_cache), \
                 patch.object(tools.workflow, "fit_response", return_value=fit) as fitter, \
                 patch.object(tools.workflow, "integrate_frame", side_effect=fake_integrate):
                result = tools.self_calibrate_cut(
                    scan, None, [np.array([0, 1])], tools.Cut("half", frame_fraction=0.5),
                    {"q_range": [200., 400.], "npt": 800}, (0., 1.), npt=1,
                )
                fitter.assert_called_once()
                self.assertIs(result["fit"], fit)
                self.assertIsNone(result["calibration_path"])
                self.assertIsNone(result["xy_path"])
                xy_directory = Path(temp) / "xy"
                destination = Path(temp) / "export"
                with patch.object(tools.workflow, "save_calibration", return_value=destination / "detector_corrections.h5"):
                    exported = tools.self_calibrate_cut(
                        scan, None, [np.array([0, 1])], tools.Cut("half", frame_fraction=0.5),
                        {"q_range": [200., 400.], "npt": 800}, (0., 1.), npt=1,
                        output_directory=destination, xy_directory=xy_directory,
                    )
                self.assertEqual(exported["xy_path"].parent, xy_directory)
                self.assertEqual(exported["xy_path"].name, "raw__half__gi_applied__self_corrected_2theta.xy")
                np.testing.assert_allclose(np.loadtxt(exported["xy_path"], ndmin=2), [[0.5, 2.]])
                self.assertTrue(exported["xy_path"].read_text().startswith("# 2theta_deg intensity"))
                expected_gain = 1.0
                destination_off = Path(temp) / "export_off"
                with patch.object(tools.workflow, "save_calibration", return_value=destination_off / "detector_corrections.h5"):
                    exported_off = tools.self_calibrate_cut(
                        scan, None, [np.array([0, 1])], tools.Cut("half", frame_fraction=0.5),
                        {"q_range": [200., 400.], "npt": 800}, (0., 1.), npt=1,
                        apply_source_gains=False, output_directory=destination_off, xy_directory=xy_directory,
                    )
                self.assertEqual(exported_off["xy_path"].name, "raw__half__gi_unapplied__self_corrected_2theta.xy")
                self.assertEqual(len(list(xy_directory.glob("*.xy"))), 2)
                self.assertFalse(list(destination.glob("*.xy")))
                self.assertFalse(list(destination_off.glob("*.xy")))
                with h5py.File(destination_off / "pattern.h5", "r") as file:
                    self.assertFalse(file.attrs["source_frame_gains_applied"])
                    np.testing.assert_equal(file["applied_frame_gains"][()], [1.])

    def test_metrics_and_missing_support(self):
        full = {"axis": np.arange(4.), "intensity": np.array([1., 2., 3., 4.]),
                "sigma": np.ones(4), "accumulators": np.full((4, 4), 200.)}
        short = {key: value.copy() for key, value in full.items()}
        short["intensity"] *= 2
        short["sigma"] *= 2
        short["accumulators"][1, -1] = 0
        metrics = tools.compare_patterns(full, short, (0, 4))
        self.assertAlmostEqual(metrics["relative_l2_percent"], 100.)
        self.assertAlmostEqual(metrics["median_bias_percent"], 100.)
        self.assertAlmostEqual(metrics["shape_log_rms_percent"], 0.)
        self.assertEqual(metrics["coverage_fraction"], 0.75)
        self.assertEqual(metrics["median_sigma_ratio"], 2.)


if __name__ == "__main__":
    unittest.main()