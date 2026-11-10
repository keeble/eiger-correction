"""Scan reduction experiments using the existing EIGER fitter and integrator.

Beamline sources remain read-only. self_calibrate_cut fits each subset anew;
integrate_cuts retains the earlier frozen-map experiment as an explicit API.
Selections refer to original raw-frame indices in frozen full-scan groups.
"""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import tempfile

import h5py
import numpy as np

import eiger_workflow as workflow


def fixed_theta_grid(maximum, step=0.05):
    """Start at zero and round the upper edge up, never stretch the step."""
    if not np.isfinite(maximum) or not 0 < maximum <= 180:
        raise ValueError("maximum must lie in (0, 180] degrees")
    if not np.isfinite(step) or step <= 0 or step > maximum:
        raise ValueError("step must be finite, positive and no larger than maximum")
    npt = int(np.ceil(maximum / step))
    upper = float(npt * step)
    if upper > 180 + 1e-10:
        raise ValueError("Rounded upper edge exceeds 180 degrees; choose a different step")
    edges = np.arange(npt + 1, dtype=float) * step
    return (0.0, upper), npt, edges


@dataclass(frozen=True)
class Cut:
    name: str
    angle_stride: int = 1
    angle_offset: int = 0
    frame_fraction: float = 1.0
    frame_mode: str = "first"
    motor_max: float | None = None
    seed: int = 20261028
    keep_motor_below: float | None = None

    def __post_init__(self):
        if not self.name or not isinstance(self.angle_stride, int) or self.angle_stride < 1:
            raise ValueError("A name and positive integer angle_stride are required")
        if not isinstance(self.angle_offset, int) or not 0 <= self.angle_offset < self.angle_stride:
            raise ValueError("angle_offset must lie in [0, angle_stride)")
        if not np.isfinite(self.frame_fraction) or not 0 < self.frame_fraction <= 1:
            raise ValueError("frame_fraction must lie in (0, 1]")
        if self.frame_mode not in {"first", "spread", "random"}:
            raise ValueError("frame_mode must be first, spread, or random")
        if self.motor_max is not None and not np.isfinite(self.motor_max):
            raise ValueError("motor_max must be finite or None")
        if self.keep_motor_below is not None and not np.isfinite(self.keep_motor_below):
            raise ValueError("keep_motor_below must be finite or None")


def angle_groups(angles, tolerance=0.001):
    """Exactly the workflow's sorted, minimum-anchored grouping rule."""
    angles = np.asarray(angles, dtype=float)
    if angles.ndim != 1 or not len(angles) or not np.isfinite(angles).all():
        raise ValueError("Expected a nonempty finite vector of motor angles")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    unique, inverse = np.unique(angles, return_inverse=True)
    labels = np.zeros(len(unique), dtype=int)
    group, minimum = 0, unique[0]
    for i, angle in enumerate(unique):
        if angle - minimum > tolerance:
            group += 1
            minimum = angle
        labels[i] = group
    membership = labels[inverse]
    return [np.flatnonzero(membership == i) for i in range(group + 1)]


def select_groups(angles, groups, cut, tolerance=0.001):
    """Keep original groups; thin their chronological raw-frame membership.

    Motor stops act on full-group mean angles. A tolerance allowance avoids
    losing the nominal 60/70 degree group due to encoder jitter; all retained
    raw angles are bounded by motor_max + 2*tolerance. This is a set of motor
    positions, NOT a chronological prefix of a multi-pass acquisition.
    keep_motor_below unions all low-motor groups with the stride selection;
    stride phase remains anchored to the original full-scan group indices.
    Frame thinning and motor_max still apply to these protected positions.
    """
    selected = {}
    for group_index, members in enumerate(groups):
        mean = float(np.mean(angles[members]))
        protected = cut.keep_motor_below is not None and mean <= cut.keep_motor_below + tolerance
        if not protected and group_index % cut.angle_stride != cut.angle_offset:
            continue
        if cut.motor_max is not None and mean > cut.motor_max + tolerance:
            continue
        n = max(1, int(np.floor(len(members) * cut.frame_fraction)))
        if cut.frame_mode == "first":
            chosen = members[:n]
        elif cut.frame_mode == "spread":
            chosen = members[np.linspace(0, len(members) - 1, n, dtype=int)]
        else:
            rng = np.random.default_rng(np.random.SeedSequence([cut.seed, group_index]))
            chosen = np.sort(rng.choice(members, n, replace=False))
        selected[group_index] = chosen.copy()
    if not selected:
        raise ValueError(f"Cut {cut.name!r} contains no frames")
    return selected


def duration_seconds(path):
    """Read explicit duration units, never silently assume seconds."""
    with h5py.File(path, "r") as file:
        for key in ("entry/duration", "entry/diamond_scan/duration"):
            if key not in file:
                continue
            ds = file[key]
            units = ds.attrs.get("units", "")
            if isinstance(units, bytes):
                units = units.decode()
            factors = {"ms": 0.001, "s": 1.0, "sec": 1.0, "seconds": 1.0}
            if str(units) not in factors:
                raise ValueError(f"Unknown duration units: {units!r}")
            value = float(np.asarray(ds[()]).item()) * factors[str(units)]
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Duration must be finite and positive")
            return value
    return float("nan")


def selection_summary(scan, groups, cuts, tolerance=0.001):
    rows = []
    duration = duration_seconds(scan.path)
    for cut in cuts:
        selection = select_groups(scan.angles, groups, cut, tolerance)
        indices = np.concatenate(list(selection.values()))
        fraction = len(indices) / len(scan.angles)
        rows.append({
            "scan": scan.path.stem, "cut": cut.name, "groups": len(selection),
            "frames": len(indices), "frame_fraction": fraction,
            "i0_fraction": float(scan.i0[indices].sum() / scan.i0.sum()),
            "motor_min_deg": float(scan.angles[indices].min()),
            "motor_max_deg": float(scan.angles[indices].max()),
            "full_wall_minutes": duration / 60,
            "proportional_time_proxy_minutes": duration * fraction / 60,
        })
    return rows


def missing_intervals(reference_support, candidate_support, edges):
    """Contiguous lost-reference bins as half-open degree intervals."""
    reference_support = np.asarray(reference_support, dtype=bool)
    candidate_support = np.asarray(candidate_support, dtype=bool)
    edges = np.asarray(edges, dtype=float)
    if (reference_support.ndim != 1 or candidate_support.shape != reference_support.shape
            or edges.shape != (len(reference_support) + 1,)
            or not np.isfinite(edges).all() or not np.all(np.diff(edges) > 0)):
        raise ValueError("Expected matching support vectors and increasing bin edges")
    missing = reference_support & ~candidate_support
    transitions = np.diff(np.r_[False, missing, False].astype(int))
    return [(float(edges[start]), float(edges[stop]))
            for start, stop in zip(np.flatnonzero(transitions == 1),
                                   np.flatnonzero(transitions == -1))]


def geometry_coverage(scan, calibration, model, groups, cuts, edges, tolerance=0.001):
    """Pixel-centre coverage after fixed masks, without detector-count reads.

    Counts are pixels, not photons or fractional integration weights. A group
    contributes once regardless of exposure; >=2 positions is a redundancy
    diagnostic, not statistical independence. Actual integration is decisive.
    """
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or not np.all(np.diff(edges) > 0):
        raise ValueError("Expected finite increasing bin edges")
    if len({cut.name for cut in cuts}) != len(cuts):
        raise ValueError("Cut names must be unique")
    selections = {cut.name: select_groups(scan.angles, groups, cut, tolerance) for cut in cuts}
    mask = (calibration["base_mask"].astype(bool)
            | calibration["bad_pixel_mask"].astype(bool) | scan.native_mask)
    output = {cut.name: {"pixels": np.zeros(len(edges) - 1, dtype=np.int64),
                         "positions": np.zeros(len(edges) - 1, dtype=np.int64)} for cut in cuts}
    for group_index in range(len(groups)):
        histograms = {}
        for cut in cuts:
            selection = selections[cut.name]
            if group_index not in selection:
                continue
            angle = float(scan.angles[selection[group_index]].mean())
            if angle not in histograms:
                integrator = model.geometry.get_ai(angle)
                theta = np.rad2deg(integrator.center_array(scan.shape, unit="2th_rad"))
                histograms[angle] = np.histogram(theta[~mask & np.isfinite(theta)], bins=edges)[0]
            counts = histograms[angle]
            output[cut.name]["pixels"] += counts
            output[cut.name]["positions"] += counts > 0
    return output


def integrate_cuts(scan, calibration, model, groups, cuts, theta_range,
                   npt=1000, saturation=1e7, tolerance=0.001, progress=print):
    """Stream each raw frame once; integrate each distinct group subset once.

    Whole-angle cuts reuse the full-group integration. Frame cuts recompute
    counts, summed I0, invalid-pixel union, and arithmetic-mean motor angle.
    Always uses frame_gain=1: source-only fitted gains are not transferred.
    """
    if len({cut.name for cut in cuts}) != len(cuts):
        raise ValueError("Cut names must be unique")
    if not isinstance(npt, int) or npt < 1:
        raise ValueError("npt must be a positive integer")
    if not np.isfinite(saturation) or saturation <= 0:
        raise ValueError("saturation must be positive and finite")
    selections = {cut.name: select_groups(scan.angles, groups, cut, tolerance) for cut in cuts}
    results = {cut.name: {} for cut in cuts}
    mask = (calibration["base_mask"].astype(bool)
            | calibration["bad_pixel_mask"].astype(bool) | scan.native_mask)
    with h5py.File(scan.detector_path, "r") as file:
        source = file[scan.detector_dataset]
        for group_index, members in enumerate(groups):
            # A key contains original raw-frame indices, never reduced-array indices.
            consumers = {}
            for name, selection in selections.items():
                if group_index in selection:
                    key = tuple(int(i) for i in selection[group_index])
                    consumers.setdefault(key, []).append(name)
            if not consumers:
                continue
            totals = {key: np.zeros(scan.shape, dtype=np.float64) for key in consumers}
            invalid = {key: np.zeros(scan.shape, dtype=bool) for key in consumers}
            destinations = {}
            for key in consumers:
                for index in key:
                    destinations.setdefault(index, []).append(key)
            for index in members:
                index = int(index)
                if index not in destinations:
                    continue
                frame = np.asarray(source[index], dtype=np.float64)
                bad = ~np.isfinite(frame) | (frame < 0) | (frame >= saturation)
                valid_counts = np.where(bad, 0.0, frame)
                for key in destinations[index]:
                    totals[key] += valid_counts
                    invalid[key] |= bad
            for key, names in consumers.items():
                counts = totals[key]
                counts[invalid[key]] = np.nan
                indices = np.asarray(key, dtype=int)
                result = workflow.integrate_frame(
                    model, counts, float(scan.angles[indices].mean()),
                    float(scan.i0[indices].sum()), mask, calibration["combined_divisor"],
                    npt, theta_range, unit="2th_deg", frame_gain=1.0, saturation=np.inf,
                )
                for name in names:
                    results[name][group_index] = result
            if progress is not None:
                progress(f"{scan.path.stem}: group {group_index + 1}/{len(groups)}; "
                         f"{len(consumers)} distinct frame selections")
    return {name: workflow.merge_results(values) for name, values in results.items()}, selections


def compare_patterns(reference, candidate, interval, min_count=100):
    """Descriptive matched-bin metrics, never an independent chi-square test."""
    np.testing.assert_allclose(reference["axis"], candidate["axis"], rtol=0, atol=1e-10)
    axis = reference["axis"]
    in_range = (axis >= interval[0]) & (axis < interval[1])
    reference_ok = (in_range & (reference["accumulators"][1] > 0)
                    & (reference["accumulators"][3] >= min_count)
                    & np.isfinite(reference["intensity"]))
    candidate_ok = ((candidate["accumulators"][1] > 0)
                    & (candidate["accumulators"][3] >= min_count)
                    & np.isfinite(candidate["intensity"]))
    common = reference_ok & candidate_ok
    n_reference, n_common = int(reference_ok.sum()), int(common.sum())
    row = {"reference_bins": n_reference, "common_bins": n_common,
           "coverage_fraction": n_common / n_reference if n_reference else np.nan,
           "missing_reference_bins": n_reference - n_common}
    for name in ("relative_l2_percent", "median_bias_percent", "shape_log_rms_percent",
                 "median_sigma_ratio", "max_supported_2theta_deg"):
        row[name] = np.nan
    supported = in_range & candidate_ok
    if supported.any():
        row["max_supported_2theta_deg"] = float(axis[supported].max())
    if not n_common:
        return row
    full, short = reference["intensity"][common], candidate["intensity"][common]
    denom = np.sum(full**2)
    if denom > 0:
        row["relative_l2_percent"] = float(100 * np.sqrt(np.sum((short - full)**2) / denom))
    positive = (full > 0) & (short > 0)
    if positive.any():
        log_ratio = np.log(short[positive] / full[positive])
        center = np.median(log_ratio)
        row["median_bias_percent"] = float(100 * np.expm1(center))
        row["shape_log_rms_percent"] = float(100 * np.sqrt(np.mean((log_ratio - center)**2)))
    full_sigma, short_sigma = reference["sigma"][common], candidate["sigma"][common]
    good_sigma = np.isfinite(full_sigma) & np.isfinite(short_sigma) & (full_sigma > 0)
    if good_sigma.any():
        row["median_sigma_ratio"] = float(np.median(short_sigma[good_sigma] / full_sigma[good_sigma]))
    return row


def self_calibrate_cut(scan, model, groups, cut, fit_settings, theta_range,
                       npt=1000, saturation=1e7, tolerance=0.001,
                       apply_source_gains=True, output_directory=None, xy_directory=None):
    """Fit this subset from scratch, never accepting a previous calibration.

    One temporary summed image per selected group bounds memory. Each cut
    rereads its selected raw frames; outputs are exclusive-create. Fitter
    support failures propagate rather than falling back to an old divisor or
    silently weakening minimum_observations. Source gains belong to this cut.
    """
    if not np.isfinite(saturation) or saturation <= 0:
        raise ValueError("saturation must be finite and positive")
    settings = dict(fit_settings)
    q_range = tuple(settings.pop("q_range"))
    selection = select_groups(scan.angles, groups, cut, tolerance)
    members_by_group = list(selection.values())
    destination = Path(output_directory) if output_directory is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="eiger_self_fit_") as temporary:
        path = Path(temporary) / "selected_group_sums.h5"
        with h5py.File(scan.detector_path, "r") as source, h5py.File(path, "x") as target:
            data = target.create_dataset("data", shape=(len(selection), *scan.shape),
                                         dtype=np.float64, chunks=(1, *scan.shape), compression="lzf")
            for position, members in enumerate(members_by_group):
                total = np.zeros(scan.shape, dtype=np.float64)
                invalid = np.zeros(scan.shape, dtype=bool)
                for index in members:
                    frame = np.asarray(source[scan.detector_dataset][int(index)], dtype=np.float64)
                    bad = ~np.isfinite(frame) | (frame < 0) | (frame >= saturation)
                    total += np.where(bad, 0., frame)
                    invalid |= bad
                total[invalid] = np.nan
                data[position] = total
        subset = replace(scan, detector_path=path, detector_dataset="data",
                         angles=np.array([scan.angles[m].mean() for m in members_by_group]),
                         i0=np.array([scan.i0[m].sum() for m in members_by_group]))
        boundary, sensor_ids = workflow.detector_layout(scan.shape)
        base_mask = scan.native_mask.astype(bool) | boundary
        sensor_ids[base_mask] = -1
        cache = workflow.build_cache(subset, model, base_mask, q_range,
                                     settings["npt"], saturation=np.inf)
        indices = np.array(sorted(cache), dtype=int)
        fit = workflow.fit_response(cache, indices, sensor_ids, len(subset.angles), **settings)
        # The cache can be large; it is not needed for integration or export.
        del cache
        mask = base_mask | fit["bad_pixel_mask"].astype(bool)
        gains = fit["frame_gains"] if apply_source_gains else np.ones(len(subset.angles))
        results = {}
        with h5py.File(path, "r") as file:
            for index, angle in enumerate(subset.angles):
                results[index] = workflow.integrate_frame(
                    model, file["data"][index], angle, subset.i0[index], mask,
                    fit["combined_divisor"], npt, theta_range, unit="2th_deg",
                    frame_gain=gains[index], saturation=np.inf)
        pattern = workflow.merge_results(results)
        export_settings = {**fit_settings, "angle_aggregation": {
            "enabled": True, "method": "sum", "angle_tolerance_deg": tolerance,
            "source_frame_count": len(scan.angles), "retained_frame_count": sum(map(len, members_by_group)),
            "analysis_position_count": len(selection), "group_sizes": list(map(len, members_by_group))},
            "short_scan_cut": asdict(cut), "self_calibrated": True}
        calibration_path = None
        xy_path = None
        if destination is not None:
            calibration_path = workflow.save_calibration(
                destination / "detector_corrections.h5", subset, model, fit,
                base_mask, sensor_ids, export_settings, [])
            with h5py.File(destination / "pattern.h5", "x") as file:
                file.attrs["schema"] = "eiger_self_calibrated_short_scan_v1"
                file.attrs["calibration_path"] = str(calibration_path)
                file.attrs["source_scan"] = str(scan.path)
                file.attrs["source_frame_gains_applied"] = bool(apply_source_gains)
                file.attrs["uncertainty"] = workflow.UNCERTAINTY
                for field, values in pattern.items():
                    file.create_dataset(field, data=values)
                file["group_angles_deg"] = subset.angles
                file["group_i0"] = subset.i0
                file["applied_frame_gains"] = gains
            supported = (pattern["accumulators"][1] > 0) & np.isfinite(pattern["intensity"])
            gain_label = "applied" if apply_source_gains else "unapplied"
            xy_destination = Path(xy_directory) if xy_directory is not None else destination
            xy_destination.mkdir(parents=True, exist_ok=True)
            xy_path = xy_destination / f"{scan.path.stem}__{cut.name}__gi_{gain_label}__self_corrected_2theta.xy"
            with xy_path.open("x", encoding="ascii") as file:
                np.savetxt(file, np.column_stack((pattern["axis"][supported], pattern["intensity"][supported])),
                           header="2theta_deg intensity", fmt="%.10g")
            with (destination / "manifest.json").open("x") as file:
                json.dump({"source_scan": str(scan.path), "settings": export_settings,
                           "theta_range": list(theta_range), "theta_npt": npt,
                           "raw_saturation": saturation, "source_frame_gains_applied": bool(apply_source_gains),
                           "xy_filename": xy_path.name,
                           "xy_path": str(xy_path.resolve()),
                           "corrections_refitted": True, "validation": "not independent; self-consistency fit",
                           "original_group_indices": list(selection),
                           "source_frame_indices": [m.tolist() for m in members_by_group]}, file, indent=2)
        return {"pattern": pattern, "fit": fit, "base_mask": base_mask,
            "selection": selection, "calibration_path": calibration_path, "xy_path": xy_path,
                "group_angles": subset.angles, "group_i0": subset.i0}


def save_experiment(directory, scan, calibration_path, cuts, patterns, selections, settings):
    """Exclusive-create all products; selection manifest preserves provenance."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source_scan": str(scan.path), "detector_path": str(scan.detector_path),
        "calibration_path": str(calibration_path), "settings": settings,
        "source_frame_gains_applied": False, "corrections_refitted": False,
        "uncertainty": workflow.UNCERTAINTY,
        "cuts": [asdict(cut) for cut in cuts],
        "selections": {name: {str(i): members.tolist() for i, members in groups.items()}
                       for name, groups in selections.items()},
    }
    with (directory / "manifest.json").open("x") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)
    with h5py.File(directory / "patterns.h5", "x") as file:
        file.attrs["schema"] = "eiger_short_scan_comparison_v1"
        file.attrs["source_scan"] = str(scan.path)
        file.attrs["calibration_path"] = str(calibration_path)
        file.attrs["settings_json"] = json.dumps(settings)
        file.attrs["uncertainty"] = workflow.UNCERTAINTY
        for name, pattern in patterns.items():
            group = file.create_group(name)
            for field, values in pattern.items():
                group.create_dataset(field, data=values)
            supported = (pattern["accumulators"][1] > 0) & np.isfinite(pattern["intensity"])
            with (directory / f"{name}.xy").open("x") as xy:
                np.savetxt(xy, np.column_stack((pattern["axis"][supported], pattern["intensity"][supported])),
                           header="2theta_deg intensity", fmt="%.10g")
    return directory