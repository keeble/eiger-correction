"""Fast synthetic regression checks; no raw scan data or facility writes."""

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

import eiger_cross_analysis as cross
import eiger_workflow as workflow


class CrossAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_root = Path(__file__).parent / "outputs" / "cross_analysis" / "test_tmp"
        cls.test_root.mkdir(parents=True, exist_ok=True)

    def test_geometry_json_and_legacy_group(self):
        config = {"wavelength": 1.6e-11, "detector_config": {"orientation": 3}, "param": [1., 2.]}
        with tempfile.TemporaryDirectory(dir=self.test_root) as directory:
            for kind in ("scalar", "nested"):
                path = Path(directory) / f"{kind}.h5"
                with h5py.File(path, "x") as file:
                    group = file.create_group("entry/plan_metadata")
                    if kind == "scalar":
                        group.create_dataset("geometry_calibration", data=json.dumps(config))
                    else:
                        geometry = group.create_group("geometry_calibration")
                        geometry.create_dataset("wavelength", data=config["wavelength"])
                        geometry.create_dataset("param", data=config["param"])
                        geometry.create_group("detector_config").create_dataset("orientation", data=3)
                self.assertEqual(workflow.geometry_from_scan(path), config)

    def test_overlap_separates_shape_and_scale(self):
        profile = np.linspace(10., 20., 100)
        intensity = np.array([profile, profile * 1.02, profile * .98])
        sigma = np.ones_like(intensity) * .01
        count = np.ones_like(intensity) * 200
        metrics = cross.overlap_metrics(intensity, sigma, count)
        self.assertLess(metrics["centered_overlap_percent"], 1e-12)
        self.assertGreater(metrics["absolute_overlap_percent"], 1)
        intensity[1] *= np.exp(np.linspace(-.02, .02, 100))
        changed = cross.overlap_metrics(intensity, sigma, count)
        self.assertGreater(changed["centered_overlap_percent"], 1)
        self.assertEqual(changed["overlap_pairs"], 2)
        with self.assertRaises(ValueError):
            cross.overlap_metrics(intensity, sigma, np.zeros_like(count))

    def test_rankings_exclude_diagonal_and_preserve_bootstrap_dimensions(self):
        numbers = list(range(100207, 100217))
        fingerprint = "synthetic_test"
        with tempfile.TemporaryDirectory(dir=self.test_root) as directory:
            root = Path(directory)
            (root / "comparisons").mkdir()
            for i, number in enumerate(numbers):
                metrics = []
                baseline = 2. + i * .05
                for j in range(11):
                    score = baseline if j == 0 else baseline * (.50 + .04 * (j - 1))
                    # A deliberately excellent self-score must never affect ranking.
                    if j == i + 1:
                        score = 1e-6
                    metrics.append({"centered_overlap_percent": score, "absolute_overlap_percent": score + .2,
                                    "frame_offset_percent": .1, "noise_scaled_rms": 1.,
                                    "overlap_pairs": 4, "overlap_bins": 100})
                with h5py.File(root / "comparisons" / f"target_{number}.h5", "x") as file:
                    file.attrs.update(complete=True, run_fingerprint=fingerprint, metrics_json=json.dumps(metrics))
                    file.create_dataset("q_nm", data=np.linspace(200, 400, 50))
                    file.create_dataset("merged_intensity", data=np.ones((11, 50)) * (i + 10))
                    file.create_dataset("merged_sigma", data=np.ones((11, 50)) * .01)
            result = cross.analyse(dict.fromkeys(numbers), root, fingerprint, bootstrap_samples=300)
            self.assertEqual(len(result["rows"]), 100)
            self.assertEqual(sum(not r["self_application"] for r in result["rows"]), 90)
            self.assertEqual(result["best_number"], numbers[0])
            for j, row in enumerate(result["ranking"]):
                self.assertEqual(row["independent_targets"], 9)
                expected = 100 * (1 - (.50 + .04 * j))
                self.assertAlmostEqual(row["median_improvement_percent"], expected)
                self.assertAlmostEqual(row["bootstrap_low_percent"], expected)
                self.assertAlmostEqual(row["bootstrap_high_percent"], expected)
            self.assertTrue(all(row["shared_independent_targets"] == 8 for row in result["pairwise"]))
            self.assertTrue(all(row["median_other_minus_best_percentage_points"] > 0 for row in result["pairwise"]))
            (root / "figures").mkdir()
            calibrations = {number: {"base_mask": np.zeros((4, 8), dtype=bool),
                                     "combined_divisor": np.ones((4, 8)) * (1 + j * .001),
                                     "sensor_gains": np.ones(8) * (1 + j * .001)}
                            for j, number in enumerate(numbers)}
            with patch.object(workflow, "load_calibration", side_effect=lambda number: calibrations[number]):
                figures = cross.plot_analysis(result, dict(zip(numbers, numbers)), root)
            self.assertEqual(len(figures), 4)
            self.assertEqual(len(list((root / "figures").glob("*.png"))), 4)
            import matplotlib.pyplot as plt
            for _, figure in figures:
                plt.close(figure)
            with self.assertRaises(ValueError):
                cross.analyse(dict.fromkeys(numbers), root, "wrong_fingerprint", bootstrap_samples=300)

    def test_output_path_guard(self):
        with self.assertRaises(ValueError):
            cross.local_path("/dls/i15-1/data/2026/cm44163-4/not_an_output.h5", self.test_root)
        self.assertEqual(cross.local_path(self.test_root / "allowed.h5", self.test_root), self.test_root / "allowed.h5")


if __name__ == "__main__":
    unittest.main()