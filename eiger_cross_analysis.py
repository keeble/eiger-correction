"""Ten-scan, fixed-mask detector-response benchmark used by the cross-analysis notebook."""

from pathlib import Path
import contextlib
import csv
import hashlib
import json
import time

import h5py
import numpy as np

import eiger_workflow as workflow


def local_path(path, workspace):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(workspace).resolve()):
        raise ValueError(f"Output must be inside the workspace: {path}")
    return path


def prepare(data_directory, scan_numbers, run_directory, workspace, settings):
    """Preflight all ten inputs and lock a resumable run to its inputs/settings/code."""
    data_directory = Path(data_directory).resolve()
    run_directory = local_path(run_directory, workspace)
    if run_directory.is_relative_to(data_directory):
        raise ValueError("The raw-data directory must remain read-only")
    for name in ("calibrations", "comparisons", "figures", "tmp", "logs"):
        (run_directory / name).mkdir(parents=True, exist_ok=True)
    scans, configs, input_stats = {}, {}, []
    for number in scan_numbers:
        path = data_directory / f"i15-1-{number}.nxs"
        companions = [path.with_name(path.stem + suffix) for suffix in
                      ("-fastcs_eiger_000001.h5", "-i0.h5", "-fastcs_eiger_meta.h5")]
        for source in [path, *companions]:
            source = source.resolve(strict=True)
            if not source.is_relative_to(data_directory):
                raise ValueError(f"Input resolves outside the raw-data directory: {source}")
            stat = source.stat()
            input_stats.append([str(source), stat.st_size, stat.st_mtime_ns])
        scans[number] = workflow.load_scan(path)
        # HDF5 external links / VDS must not read detector counts from another directory.
        for source, dataset in ((scans[number].detector_path, scans[number].detector_dataset),
                                (companions[1], "entry/data/data")):
            with h5py.File(source, "r") as file:
                node = file[dataset]
                referenced = [Path(node.file.filename)]
                if node.is_virtual:
                    referenced.extend(Path(workflow._text(item.file_name)) for item in node.virtual_sources())
                if node.external:
                    referenced.extend(Path(workflow._text(item[0])) for item in node.external)
                for linked in referenced:
                    linked = (source.parent / linked).resolve() if not linked.is_absolute() else linked.resolve()
                    if not linked.is_relative_to(data_directory):
                        raise ValueError(f"HDF5 data points outside the allowed directory: {linked}")
        configs[number] = workflow.geometry_from_scan(path)
    first = next(iter(scans))
    canonical_geometry = json.dumps(configs[first], sort_keys=True)
    for number, scan in scans.items():
        if json.dumps(configs[number], sort_keys=True) != canonical_geometry:
            raise ValueError(f"Geometry differs for {number}; a common geometry is required for this benchmark")
        reference = scans[first]
        for field in ("energy_kev", "thickness_m", "wavelength_m"):
            if not np.isclose(getattr(scan, field), getattr(reference, field), rtol=1e-6, atol=0):
                raise ValueError(f"{field} mismatch for {number}")
        if (scan.shape, scan.orientation, scan.material) != (reference.shape, reference.orientation, reference.material):
            raise ValueError(f"Detector mismatch for {number}")
    manifest = {"schema": "eiger_cross_analysis_v1", "scan_numbers": list(scan_numbers),
                "settings": settings, "inputs": input_stats, "geometry": configs[first],
                "pyfai_version": workflow.pyFAI.version, "numpy_version": np.__version__,
                "code_sha256": {Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                for module in (workflow, __import__(__name__))}}
    serialized = json.dumps(manifest, sort_keys=True, indent=2)
    fingerprint = hashlib.sha256(serialized.encode()).hexdigest()
    manifest_path = run_directory / "manifest.json"
    if manifest_path.exists():
        if manifest_path.read_text() != serialized:
            raise ValueError("Inputs, settings or code changed. Choose a new RUN_NAME; old results will not be mixed.")
    else:
        with manifest_path.open("x") as stream:
            stream.write(serialized)
    return scans, configs, fingerprint


def make_model(scan, config, settings, run_directory):
    return workflow.PhysicalModel(config, scan, settings["polarization"], settings["density"],
                                  temporary_directory=Path(run_directory) / "tmp")


def _complete(path, fingerprint):
    if not path.exists():
        return False
    with h5py.File(path, "r") as file:
        if not file.attrs.get("complete", False) or file.attrs.get("run_fingerprint") != fingerprint:
            raise ValueError(f"Incomplete or incompatible existing output: {path}")
    return True


def extract_all(scans, configs, settings, run_directory, fingerprint):
    """One full fit per scan; only one raw fitting cache is held at a time."""
    run_directory = Path(run_directory)
    fit_settings = settings["fit"]
    options = {key: value for key, value in fit_settings.items() if key != "q_range"}
    paths = {}
    for number, scan in scans.items():
        destination = run_directory / "calibrations" / f"{number}_detector_corrections.h5"
        paths[number] = destination
        if _complete(destination, fingerprint):
            print(f"{number}: reusing verified calibration", flush=True)
            continue
        partial = destination.with_suffix(".partial.h5")
        partial.unlink(missing_ok=True)
        start = time.monotonic()
        model = make_model(scan, configs[number], settings, run_directory)
        model.vertical_check()
        boundaries, sensor_ids = workflow.detector_layout(scan.shape)
        base_mask = boundaries | scan.native_mask
        sensor_ids[base_mask] = -1
        workflow.constant_pattern_check(model, base_mask, fit_settings["q_range"], fit_settings["npt"])
        cache = workflow.build_cache(scan, model, base_mask, fit_settings["q_range"],
                                     fit_settings["npt"], settings["saturation"])
        indices = np.array(sorted(cache), dtype=int)
        print(f"{number}: fitting {len(indices)} positions; cache "
              f"{sum(a.nbytes for f in cache.values() for a in f.values()) / 2**20:.1f} MiB", flush=True)
        with (run_directory / "logs" / f"fit_{number}.txt").open("w") as log:
            with contextlib.redirect_stdout(log):
                fit = workflow.fit_response(cache, indices, sensor_ids, len(scan.angles), **options)
        diagnostics = [{"method": "All contributing positions fitted; independent-scan validation in the 10x10 benchmark",
                        "self_application_is_training_diagnostic": True}]
        workflow.save_calibration(partial, scan, model, fit, base_mask, sensor_ids, fit_settings, diagnostics)
        with h5py.File(partial, "r+") as file:
            file.attrs["run_fingerprint"] = fingerprint
            file.attrs["complete"] = True
            file.attrs["elapsed_seconds"] = time.monotonic() - start
        partial.rename(destination)
        print(f"{number}: saved; sensor gains={np.round(fit['sensor_gains'], 5)}; "
              f"bad pixels={fit['bad_pixel_mask'].sum()}; {time.monotonic() - start:.1f}s", flush=True)
        del cache, fit, model
    return paths


def overlap_metrics(intensity, sigma, count, min_count=100, min_bins=20):
    """Adjacent contributing frames; each frame-pair has equal weight in the median."""
    indices = np.flatnonzero(np.any(count >= min_count, axis=1))
    centered, absolute, noise_scaled, offsets, bins_used = [], [], [], [], []
    for left, right in zip(indices[:-1], indices[1:]):
        valid = (count[left] >= min_count) & (count[right] >= min_count)
        valid &= np.isfinite(intensity[left]) & np.isfinite(intensity[right])
        valid &= (intensity[left] > 0) & (intensity[right] > 0)
        valid &= np.isfinite(sigma[left]) & np.isfinite(sigma[right])
        if valid.sum() < min_bins:
            continue
        log_ratio = np.log(intensity[left, valid] / intensity[right, valid])
        offset = np.median(log_ratio)
        residual = log_ratio - offset
        variance = (sigma[left, valid] / intensity[left, valid])**2 + (sigma[right, valid] / intensity[right, valid])**2
        good_variance = variance > 0
        centered.append(100 * np.sqrt(np.mean(residual**2)))
        absolute.append(100 * np.sqrt(np.mean(log_ratio**2)))
        offsets.append(100 * abs(offset))
        noise_scaled.append(np.sqrt(np.mean(residual[good_variance]**2 / variance[good_variance])) if good_variance.any() else np.nan)
        bins_used.append(int(valid.sum()))
    if not centered:
        raise ValueError("Insufficient adjacent-frame overlap. Adjust the q range explicitly, not the quality thresholds.")
    return {"centered_overlap_percent": float(np.median(centered)),
            "absolute_overlap_percent": float(np.median(absolute)),
            "frame_offset_percent": float(np.median(offsets)),
            "noise_scaled_rms": float(np.nanmedian(noise_scaled)),
            "overlap_pairs": len(centered), "overlap_bins": sum(bins_used)}


def compare_all(scans, configs, calibration_paths, settings, run_directory, fingerprint):
    """Read each raw frame once; apply ten maps plus a unit-divisor baseline on one fixed mask."""
    run_directory = Path(run_directory)
    numbers = list(scans)
    calibrations = {number: workflow.load_calibration(calibration_paths[number]) for number in numbers}
    boundaries, _ = workflow.detector_layout(next(iter(scans.values())).shape)
    # No response-derived bad-pixel mask: avoids selecting validation pixels using a target's own fit.
    common_mask = boundaries | np.logical_or.reduce([scan.native_mask for scan in scans.values()])
    if np.count_nonzero(~common_mask) == 0:
        raise ValueError("No pixels survive the fixed common native/boundary mask")
    divisors = [np.ones(common_mask.shape)] + [calibrations[number]["combined_divisor"] for number in numbers]
    for target, scan in scans.items():
        destination = run_directory / "comparisons" / f"target_{target}.h5"
        if _complete(destination, fingerprint):
            print(f"{target}: reusing verified 11-way integrations", flush=True)
            continue
        partial = destination.with_suffix(".partial.h5")
        partial.unlink(missing_ok=True)
        start = time.monotonic()
        model = make_model(scan, configs[target], settings, run_directory)
        model.vertical_check()
        npt = settings["fit"]["npt"]
        shape = (11, len(scan.angles), npt)
        intensity, sigma, count = [np.full(shape, np.nan) for _ in range(3)]
        totals = np.zeros((11, 4, npt))
        axis = None
        with h5py.File(scan.detector_path, "r") as raw:
            for index, angle in enumerate(scan.angles):
                counts = raw[scan.detector_dataset][index]
                for correction, divisor in enumerate(divisors):
                    result = workflow.integrate_frame(model, counts, angle, scan.i0[index], common_mask,
                                                      divisor, npt, settings["fit"]["q_range"],
                                                      saturation=settings["saturation"])
                    if axis is None:
                        axis = result.radial.copy()
                    np.testing.assert_allclose(result.radial, axis)
                    valid = result.sum_normalization > 0
                    intensity[correction, index] = np.where(valid, result.intensity, np.nan)
                    sigma[correction, index] = np.where(valid, result.sigma, np.nan)
                    count[correction, index] = result.count
                    for k, field in enumerate(("sum_signal", "sum_normalization", "sum_variance", "count")):
                        totals[correction, k] += getattr(result, field)
                if (index + 1) % 20 == 0:
                    print(f"Target {target}: {index + 1}/{len(scan.angles)} frames x 11 corrections", flush=True)
        metrics = [overlap_metrics(intensity[k], sigma[k], count[k], settings["min_count"], settings["min_bins"])
                   for k in range(11)]
        supported = totals[:, 1] > 0
        merged = np.divide(totals[:, 0], totals[:, 1], out=np.full((11, npt), np.nan), where=supported)
        merged_sigma = np.divide(np.sqrt(totals[:, 2]), totals[:, 1], out=np.full((11, npt), np.nan), where=supported)
        with h5py.File(partial, "x") as file:
            file.attrs.update(schema="eiger_cross_target_v1", run_fingerprint=fingerprint,
                              source_scan=str(scan.path), target_scan=target, metrics_json=json.dumps(metrics),
                              mask_policy="Fixed union of native masks and detector boundaries; no learned masks",
                              frame_gains_applied=False, uncertainty=workflow.UNCERTAINTY,
                              elapsed_seconds=time.monotonic() - start)
            for name, data in (("correction_scan", [-1, *numbers]), ("q_nm", axis),
                               ("frame_angles_deg", scan.angles), ("frame_i0", scan.i0),
                               ("common_mask", common_mask), ("frame_intensity", intensity),
                               ("frame_sigma", sigma), ("frame_count", count),
                               ("merged_intensity", merged), ("merged_sigma", merged_sigma),
                               ("merged_accumulators", totals)):
                file.create_dataset(name, data=data, compression="gzip", shuffle=True)
            file.attrs["complete"] = True
        partial.rename(destination)
        print(f"Target {target}: saved; baseline overlap {metrics[0]['centered_overlap_percent']:.3f}%; "
              f"{time.monotonic() - start:.1f}s", flush=True)
        del model, intensity, sigma, count
    return common_mask


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyse(scans, run_directory, fingerprint, bootstrap_samples=5000, seed=20261012):
    """Rank by paired baseline improvement on independent targets, never on the diagonal."""
    run_directory = Path(run_directory)
    numbers = list(scans)
    n = len(numbers)
    rows, profiles, uncertainties, metric_data = [], [], [], []
    for target in numbers:
        path = run_directory / "comparisons" / f"target_{target}.h5"
        if not _complete(path, fingerprint):
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as file:
            metrics = json.loads(file.attrs["metrics_json"])
            profiles.append(file["merged_intensity"][()])
            uncertainties.append(file["merged_sigma"][()])
            axis = file["q_nm"][()]
            metric_data.append(metrics)
    # Axis order throughout: target x correction x q. Baseline occupies correction index zero.
    profiles = np.asarray(profiles)
    uncertainties = np.asarray(uncertainties)
    score = np.array([[m["centered_overlap_percent"] for m in metrics[1:]] for metrics in metric_data])
    absolute = np.array([[m["absolute_overlap_percent"] for m in metrics[1:]] for metrics in metric_data])
    baseline = np.array([metrics[0]["centered_overlap_percent"] for metrics in metric_data])
    if np.any(baseline <= 0):
        raise ValueError("Zero baseline overlap prevents a meaningful relative improvement score")
    improvement = 100 * (baseline[:, None] - score) / baseline[:, None]
    off_diagonal = ~np.eye(n, dtype=bool)
    independent = np.where(off_diagonal, improvement, np.nan)
    rng = np.random.default_rng(seed)
    # Resample complete target rows, not bins/frames or 100 apparently independent observations.
    draws = rng.integers(0, n, size=(bootstrap_samples, n))
    bootstrap = np.nanmedian(independent[draws], axis=1)
    intervals = np.nanpercentile(bootstrap, [2.5, 97.5], axis=0)
    bootstrap_winner = np.nanargmax(bootstrap, axis=1)
    best_for_target = np.nanargmin(np.where(off_diagonal, score, np.nan), axis=1)
    ranking = []
    for j, correction in enumerate(numbers):
        selected = off_diagonal[:, j]
        ranking.append({"correction_scan": correction, "independent_targets": int(selected.sum()),
                        "median_improvement_percent": float(np.median(improvement[selected, j])),
                        "bootstrap_low_percent": float(intervals[0, j]), "bootstrap_high_percent": float(intervals[1, j]),
                        "median_overlap_percent": float(np.median(score[selected, j])),
                        "worst_overlap_percent": float(np.max(score[selected, j])),
                        "median_absolute_overlap_percent": float(np.median(absolute[selected, j])),
                        "targets_better_than_baseline": int(np.count_nonzero(improvement[selected, j] > 0)),
                        "target_wins": int(np.count_nonzero(best_for_target == j)),
                        "bootstrap_best_fraction": float(np.mean(bootstrap_winner == j)),
                        "self_overlap_percent": float(score[j, j])})
    ranking.sort(key=lambda row: row["median_improvement_percent"], reverse=True)
    best_number = ranking[0]["correction_scan"]
    best_j = numbers.index(best_number)
    pairwise = []
    for j, correction in enumerate(numbers):
        if j == best_j:
            continue
        selected = off_diagonal[:, j] & off_diagonal[:, best_j]
        difference = score[selected, j] - score[selected, best_j]
        paired = np.median(difference[rng.integers(0, len(difference), size=(bootstrap_samples, len(difference)))], axis=1)
        lo, hi = np.percentile(paired, [2.5, 97.5])
        pairwise.append({"best_correction": best_number, "other_correction": correction,
                         "shared_independent_targets": int(selected.sum()),
                         "median_other_minus_best_percentage_points": float(np.median(difference)),
                         "bootstrap_low": float(lo), "bootstrap_high": float(hi),
                         "best_wins_on_shared_targets": int(np.count_nonzero(difference > 0))})
    for i, target in enumerate(numbers):
        for j, correction in enumerate(numbers):
            # Secondary shape agreement uses other independently corrected repeats; excludes both donor and target.
            peers = [k for k in range(n) if k not in (i, j)]
            candidate = profiles[i, j + 1]
            peer_profiles = profiles[peers, j + 1]
            valid = np.isfinite(candidate) & (candidate > 0) & np.all(np.isfinite(peer_profiles) & (peer_profiles > 0), axis=0)
            repeat_shape = np.nan
            if valid.sum() >= 20:
                target_log = np.log(candidate[valid])
                reference_log = np.median(np.log(peer_profiles[:, valid]), axis=0)
                residual = target_log - reference_log
                repeat_shape = float(100 * np.sqrt(np.mean((residual - np.median(residual))**2)))
            rows.append({"target_scan": target, "correction_scan": correction,
                         "self_application": bool(i == j), **metric_data[i][j + 1],
                         "baseline_overlap_percent": float(baseline[i]),
                         "improvement_percent": float(improvement[i, j]),
                         "leave_target_and_donor_out_repeat_shape_percent": repeat_shape})
    write_csv(run_directory / "pair_metrics_100.csv", rows)
    write_csv(run_directory / "correction_ranking.csv", ranking)
    write_csv(run_directory / "paired_best_comparisons.csv", pairwise)
    write_csv(run_directory / "baseline_metrics_10.csv", [{"target_scan": number, **metric_data[i][0]} for i, number in enumerate(numbers)])
    summary = {"run_fingerprint": fingerprint, "ranking": ranking, "paired_comparisons": pairwise,
               "bootstrap_samples": bootstrap_samples, "bootstrap_seed": seed,
               "primary_metric": "Median baseline-relative improvement of centered adjacent-frame overlap on 9 non-source scans",
               "caveat": "Exploratory target-cluster bootstrap: only 10 related repeats, overlapping calibration data, selection bias, bin covariance and calibration/I0/model uncertainty are not covered."}
    (run_directory / "summary.json").write_text(json.dumps(summary, indent=2))
    return {"numbers": numbers, "rows": rows, "ranking": ranking, "pairwise": pairwise,
            "score": score, "absolute": absolute, "baseline": baseline, "improvement": improvement,
            "axis": axis, "profiles": profiles, "uncertainties": uncertainties,
            "best_number": best_number, "best_j": best_j}


def plot_analysis(analysis, calibration_paths, run_directory):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    run_directory = Path(run_directory)
    numbers = analysis["numbers"]
    labels = [str(number) for number in numbers]
    figures = []
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for ax, values, title, cmap in ((axes[0], analysis["score"], "Centered overlap (%; lower is better)", "viridis_r"),
                                   (axes[1], analysis["improvement"], "Improvement over physical-only (%; higher is better)", "coolwarm")):
        kwargs = {}
        if ax is axes[1]:
            extent = max(float(np.nanmax(np.abs(values))), 1)
            kwargs = {"vmin": -extent, "vmax": extent}
        image = ax.imshow(values, cmap=cmap, **kwargs)
        ax.set(xticks=range(10), xticklabels=labels, yticks=range(10), yticklabels=labels,
               xlabel="Correction source scan", ylabel="Target scan", title=title)
        ax.tick_params(axis="x", labelrotation=90)
        for i in range(10):
            ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False, hatch="///", edgecolor="black", linewidth=1))
            for j in range(10):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        bbox={"facecolor": "white", "alpha": .65, "edgecolor": "none"})
        fig.colorbar(image, ax=ax, shrink=.8)
    fig.suptitle("100 applications; hatched self-applications are excluded from rankings")
    figures.append(("comparison_heatmaps", fig))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    ranking = analysis["ranking"]
    for position, row in enumerate(ranking):
        j = numbers.index(row["correction_scan"])
        independent = np.arange(10) != j
        values = analysis["improvement"][independent, j]
        axes[0].scatter(values, position + np.linspace(-.15, .15, len(values)), s=20, alpha=.6)
        axes[0].plot([row["bootstrap_low_percent"], row["bootstrap_high_percent"]], [position, position], color="black")
        axes[0].plot(row["median_improvement_percent"], position, "ks")
    axes[0].set(yticks=range(10), yticklabels=[str(row["correction_scan"]) for row in ranking],
                xlabel="Improvement over baseline (%)", title="Nine independent targets per correction; median and exploratory 95% interval")
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="gray", linestyle="--")
    for j, number in enumerate(numbers):
        independent = np.arange(10) != j
        axes[1].plot(np.array(numbers)[independent], analysis["score"][independent, j], "o-", label=str(number), alpha=.65)
    axes[1].plot(numbers, analysis["baseline"], "k--", linewidth=2, label="Physical-only")
    axes[1].set(xlabel="Target scan", ylabel="Centered overlap (%)", title="Target dependence / scan-order drift")
    axes[1].ticklabel_format(axis="x", style="plain", useOffset=False)
    axes[1].legend(fontsize=8, ncol=2)
    figures.append(("ranking_and_target_dependence", fig))
    calibrations = {n: workflow.load_calibration(p) for n, p in calibration_paths.items()}
    common = ~np.logical_or.reduce([c["base_mask"].astype(bool) for c in calibrations.values()])
    logs = np.array([np.log(calibrations[n]["combined_divisor"][common]) for n in numbers])
    distance = 100 * np.sqrt(np.mean((logs[:, None] - logs[None, :])**2, axis=2))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    image = axes[0].imshow(distance, cmap="magma")
    axes[0].set(xticks=range(10), xticklabels=labels, yticks=range(10), yticklabels=labels,
                title="Correction-map log-RMS differences (%)")
    axes[0].tick_params(axis="x", labelrotation=90)
    fig.colorbar(image, ax=axes[0])
    for n in numbers:
        axes[1].plot(range(1, 9), calibrations[n]["sensor_gains"], "o-", label=str(n))
    axes[1].set(xlabel="Sensor region", ylabel="Divisor", title="Sensor-gain reproducibility")
    axes[1].legend(fontsize=8, ncol=2)
    figures.append(("correction_reproducibility", fig))
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    best = analysis["best_j"]
    q = analysis["axis"]
    for i, target in enumerate(numbers):
        if i == best:
            continue
        profile = analysis["profiles"][i, best + 1]
        baseline = analysis["profiles"][i, 0]
        valid = np.isfinite(profile) & np.isfinite(baseline) & (profile > 0) & (baseline > 0)
        axes[0].plot(q[valid], profile[valid] / np.median(profile[valid]), label=str(target), alpha=.7)
        axes[1].plot(q[valid], 100 * (profile[valid] / baseline[valid] - 1), alpha=.7)
    axes[0].set(ylabel="Median-normalized intensity", title=f"Best ranked correction {analysis['best_number']}: independent-target patterns")
    axes[0].legend(fontsize=8, ncol=3)
    axes[1].set(xlabel="q (nm$^{-1}$)", ylabel="Change from baseline (%)", title="Correction-induced pattern changes (not truth residuals)")
    figures.append(("best_correction_patterns", fig))
    for name, figure in figures:
        figure.savefig(run_directory / "figures" / f"{name}.png", dpi=160)
    return figures