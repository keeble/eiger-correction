# EIGER Correction Workflow

A small, notebook-driven workflow for extracting detector pixel/sensor response corrections from a compatible raw EIGER scan, then applying a saved calibration to another compatible scan.

## Contents

- `extract_eiger_corrections.ipynb`: fit and validate a calibration, then export it to HDF5.
- `process_eiger_scan.ipynb`: apply an existing calibration without refitting, then write processed HDF5 and 2theta `.xy` output.
- `cross_analyse_eiger_repeats.ipynb`: extract ten corrections (100207–100216), apply every map to every scan, and compare 100 applications against ten physical-only controls.
- `eiger_cross_analysis.py`: resumable cross-analysis, independent-target statistics, provenance and plots.
- `eiger_workflow.py`: shared implementation used by both notebooks.
- `data/`: local raw inputs; ignored by Git.
- `outputs/`: local calibration and processed outputs; ignored by Git.

## Environment

Create the Conda environment from the repository root:

```sh
micromamba env create -f environment.yml
micromamba activate eiger-correction
python -m ipykernel install --user --name eiger-correction --display-name "Python (eiger-correction)"
jupyter lab
```

Select the `Python (eiger-correction)` kernel in VS Code or Jupyter. The environment pins pyFAI to the version used to validate this workflow and constrains NumPy to remain compatible with SciPy.

## Run Order

1. Put a dense, suitable calibration scan and its companion detector, monitor, metadata, and geometry files in `data/`, or edit the path settings in the extraction notebook.
2. Run `extract_eiger_corrections.ipynb` from top to bottom. Inspect the support diagnostics and held-out validation before accepting the calibration. The output is written under `outputs/`.
3. Put a compatible target scan and its companion files in `data/`, or edit the processing notebook paths. Set `CALIBRATION_PATH` to the extracted HDF5.
4. Run `process_eiger_scan.ipynb`. It applies the saved detector map without refitting, plots measured I0 by frame, and writes a processed HDF5 plus a plain-text 2theta `.xy` file under `outputs/`.

By default, companion files are inferred from the NeXus stem: `-fastcs_eiger_000001.h5`, `-i0.h5`, and `-fastcs_eiger_meta.h5`. The notebooks allow explicit path overrides.

## Ten-Repeat Cross-Analysis

Open `cross_analyse_eiger_repeats.ipynb`, select `/dls/science/users/fer45166/envs/pyfai_new/bin/python`, and run top-to-bottom. It reads scans 100207–100216 and their companion files from `/dls/i15-1/data/2026/cm44163-4` in read-only mode, using their embedded geometry. All generated files, including temporary geometry files and caches, remain under `outputs/cross_analysis/<RUN_NAME>/` in this repository.

The notebook first extracts all ten reusable calibration files, then evaluates each on all ten scans, using a common native/boundary mask and each target's own I0. Source-frame gains and learned bad-pixel masks are not used in the primary benchmark. Rankings exclude self-applications and compare overlap-shape improvement on nine other targets against paired physical-only controls. Absolute overlap, repeat agreement, map stability, scan-order dependence and exploratory target-bootstrap intervals help identify tradeoffs. These diagnostics measure consistency, not absolute calibration accuracy.

Completed per-scan checkpoints can be reused after an interruption. Inputs, scientific settings and helper-code fingerprints must match; choose a new `RUN_NAME` after changing them. Outputs include all 100 pair metrics, rankings, paired winner comparisons, compact per-frame/merged HDF5 results, fit diagnostics/logs and PNG plots. Fitting and the 110 integrations per scan set are substantial CPU tasks; inspect the documented convergence and support diagnostics before trusting a result.

Fast synthetic regression tests are provided in `test_eiger_cross_analysis.py`; test artifacts are confined to this repository's output directory.

## Scope and Limitations

The implementation supports the documented EIGER2 CdTe 500K detector (512 x 1028) and requires matching scan/calibration energy, wavelength, sensor material/thickness, orientation, and geometry convention. It is not a general-purpose EIGER reader. Read the notebook explanations before using another detector or geometry.

The fitted map is a pixel/sensor response correction; it is not a fitted function of tth or Q. Physical polarization, solid-angle, and modeled CdTe absorption corrections do vary with geometry. The CdTe absorption model is not measured photon-counting QE. Counting uncertainties omit calibration, monitor, model uncertainty, and covariance between neighboring split bins. Treat the calibration as exploratory until independently validated with suitable reference data.

## Sharing and Data

This repository contains no scan data or calibration results. Share raw inputs separately through the appropriate institutional storage. Do not commit facility data, generated `.h5` results, or output `.xy` files unless their sharing status and size are explicitly reviewed. Choose a license only after confirming the rights and institutional policy for this code and its scientific method.
