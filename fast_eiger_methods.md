# Joint detector-response estimation and integration of repeated-angle EIGER scans

## Abstract

This document describes the method implemented in [extract_eiger_corrections_fast.ipynb](extract_eiger_corrections_fast.ipynb) and [eiger_workflow.py](eiger_workflow.py), with emphasis on the origin and interpretation of the final two-column XY pattern. Repeated detector images at approximately equal motor angles are summed, normalized by their summed incident-beam monitor, and interpreted through a fixed geometry and physical detector model. Redundant measurements of the same scattering-vector magnitude at different detector positions are then used to estimate relative pixel and sensor responses, an unknown common scattering profile, and optional source-position intensity gains. These quantities are estimated by a regularized, robust, alternating fit in logarithmic intensity. The detector response is subsequently held fixed while the images are integrated and merged into a uniform two-theta pattern. The method is a self-consistency calibration, not a refinement against a known reference pattern, and its validity depends on the separability of scattering, detector response, and position-dependent intensity effects.

**Scope.** This is a methods description of the implementation as inspected on 20 October 2026, not a claim that its physical assumptions have been independently established. Numerical examples refer to the live calculation for scan **100587**, not to all scans or necessarily to subsequently edited settings. Diagnostic plots are described in Appendix A.

## 1. Scientific premise and assumptions

The experiment measures a sample while a motor changes the detector position. The same scattering-vector magnitude, $Q$, can therefore be sampled by different physical detector pixels and sensor regions. Conversely, a physical pixel samples different $Q$ values as the detector moves. This overlap is the information used to separate a common scattering pattern from detector-dependent multiplicative sensitivity.

The central assumptions are:

1. **A common one-dimensional scattering profile exists.** After the specified physical corrections, the expected signal depends on $Q$, not independently on azimuth, pixel identity, or acquisition time. Texture, changing sample state, angle-dependent sample absorption, anisotropic background, or geometry errors can violate this assumption.
2. **Detector response is multiplicative and stable.** Each usable pixel has a fixed response, with an additional common scale for its sensor region. These factors do not themselves depend on $Q$, motor angle, or intensity during the scan. No additive detector offset is fitted.
3. **Position effects are separable.** When enabled, one extra scalar per fitted analysis position can absorb residual intensity changes not explained by I0. It cannot represent a change in the *shape* of the scattering pattern at that position.
4. **Geometry and physical corrections are externally specified.** Detector geometry, wavelength, polarization, and the attenuation model are not refined in this notebook. Their errors can be misidentified as detector response or position gains.
5. **Photon-counting noise is an adequate starting approximation.** Raw-count variance is taken as the count itself for integration. Robust fitting additionally allows a fractional discrepancy floor; this does not turn the final uncertainty into a complete model-error estimate.
6. **Nearby motor angles can be replaced by one angle.** Summing images is exact as a count operation, but assigning all of them a common geometry is an approximation when their motor readings differ.

The desired output is a **relative, physically normalized scattering intensity**, not an absolute differential cross-section or a structure factor $S(Q)$. No sample-mass normalization, empty-capillary subtraction, Compton subtraction, or PDF normalization is performed here.

## 2. Inputs, notation, and fixed physical model

### 2.1 Input observations

For raw frame $i$ and detector pixel $p$, let $C_{ip}$ be the recorded count, $M_i$ the positive I0 monitor measurement, and $\alpha_i$ the motor two-theta reading. The loader reads the angle array from the NeXus scan and the detector counts, monitor, and detector metadata from the corresponding files. It checks consistency of frame counts, masks, detector shape, CdTe material, and positive energy and thickness.

The detector has $512\times1028=526{,}336$ stored pixels and is partitioned into eight sensor regions. A native detector mask is combined with an explicit boundary mask. Masking is an exclusion, not replacement by a zero-intensity observation.

The geometry is supplied by the configured goniometer calibration JSON. It maps each motor angle and pixel to a scattering angle $\beta_{ip}=2\theta_{ip}$ and hence

$$
Q_{ip}=\frac{4\pi}{\lambda}\sin\left(\frac{\beta_{ip}}{2}\right).
$$

The motor angle $\alpha_i$ is **not** the scattering angle of every pixel: the latter varies across the detector. Internal fitting uses $Q$ in nm$^{-1}$; the sampling diagnostic displays Å$^{-1}$, with $Q_{\mathrm{Å}^{-1}}=Q_{\mathrm{nm}^{-1}}/10$.

### 2.2 Physical normalization

At each position the code constructs a known positive normalization factor

$$
K_{ip}=\Delta\Omega_{ip}\,P_{ip}\,A_{ip}.
$$

Here $\Delta\Omega$ is the pyFAI solid-angle array multiplied by the pixel-area/distance-squared prefactor, $P$ is the pyFAI polarization factor, and $A$ is the planar CdTe absorption/efficiency factor. The configured polarization factor is 0.95. The attenuation coefficient is calculated from xraylib's total mass attenuation coefficient and the specified density, 6.2 g cm$^{-3}$. With thickness $t$ expressed in centimetres,

$$
T_0=\exp(-\mu t),\qquad \eta_0=1-T_0.
$$

The code obtains the angle-dependent transmission correction from pyFAI and multiplies it by $\eta_0$ to form $A$. This is a Beer–Lambert sensor model, **not a measured quantum-efficiency calibration**. Charge sharing, threshold effects, and other departures from this model are not separately parameterized.

Wavelength and detector orientation are checked against the scan metadata. A geometry check verifies the predominantly vertical scattering-plane convention relevant to polarization. A synthetic constant-pattern integration tests consistency between the physical-normalization arrays and the integration implementation. Neither test proves that the external geometry is correct for the actual sample.

## 3. Step 1: compress repeated motor-angle observations

Sorted unique motor angles are grouped using a minimum-anchored tolerance. A new angle joins the current group only if

$$
\alpha-\min(\alpha\text{ in that group})\leq\epsilon,
\qquad \epsilon=0.001^\circ
$$

at the current default. This is not chained nearest-neighbour grouping and not a symmetric tolerance around a mean. Setting the tolerance to zero groups exact repeats only.

For group $g$ with raw-frame members $\mathcal G_g$,

$$
C_{gp}=\sum_{i\in\mathcal G_g}C_{ip},\qquad
M_g=\sum_{i\in\mathcal G_g}M_i,\qquad
\bar\alpha_g=\frac{1}{|\mathcal G_g|}\sum_{i\in\mathcal G_g}\alpha_i.
$$

**I0 is not used to calculate the motor angle.** Every raw-frame motor reading contributes equally. I0 enters only the intensity normalization.

Counts are summed rather than averaged so that Poisson count variance remains naturally associated with the summed count. A group pixel is excluded if *any* constituent count is nonfinite, negative, or at or above the raw saturation threshold, currently $10^7$. The surviving group sum may legitimately exceed that threshold; downstream group processing therefore uses an infinite sum-level saturation threshold.

Summation sacrifices information: individual-repeat anomalies are no longer identifiable, and ten repeats do not become ten independent observations for pixel-response refinement. They become one observation with a larger count and summed monitor. If aggregation is disabled, each raw frame remains an analysis position.

## 4. Step 2: construct the fitting observations

The fit uses only the configured high-$Q$ interval, currently **200–400 nm$^{-1}$ (20–40 Å$^{-1}$)**, divided into **800 uniform $Q$ bins**. These are distinct from the final export bins.

For each analysis position, pixel-centre $Q$ assigns each admitted pixel to exactly one fitting bin. The cache records the pixel identity, bin index, count, and $\log(M_gK_{gp})$. It excludes the base mask, nonfinite counts, saturation failures, and pixels outside the fitting interval. In the normal aggregated path, negative constituent counts have already been excluded. The cache itself does not explicitly reject negative values in the unaggregated fallback; that implementation detail should not be mistaken for a physically valid negative-count noise model.

The fitting observations are **individual pixel measurements**, not already integrated intensity points. The cache does not use pyFAI's bounding-box pixel splitting. The observed logarithmic quantity is

$$
y_{gp}=\log(C_{gp}+\tfrac12)-\log(M_gK_{gp}).
$$

The half-count offset permits zero-count observations to enter the logarithmic fit. It is used only in fitting; it is not added to the counts exported through integration.

## 5. Step 3: jointly estimate scattering and multiplicative responses

### 5.1 Forward model

The conceptual expected-count model is

$$
\mathbb E[C_{gp}]\approx
M_g K_{gp}\, I_{b(g,p)}\,d_p\,s_{r(p)}\,f_g,
$$

where $b(g,p)$ is the fitting bin and $r(p)$ is the sensor containing pixel $p$. In logarithmic form the fitted relationship is

$$
y_{gp}\approx h_{b(g,p)}+u_p+v_{r(p)}+a_g,
$$

with $h_b=\log I_b$, $u_p=\log d_p$, $v_r=\log s_r$, and $a_g=\log f_g$.

The shared profile $I_b$ is **unknown and estimated from this scan**. No theoretical or independently measured standard pattern is supplied. The total transferable detector divisor is

$$
D_p=d_p s_{r(p)}.
$$

A response above unity represents excess sensitivity: counts are subsequently **divided** by $D_p$, not multiplied by it.

### 5.2 How many quantities are being refined?

| Family | Role | Number of stored entries | Number eligible to vary |
|---|---|---:|---:|
| Profile $h_b$ | Common unknown scattering shape in fit interval | 800 | Supported bins; unsupported values are interpolated |
| Pixel terms $u_p$ | Relative fine-scale sensitivity | 526,336 | Pixels with at least 8 positive-weight observations in that iteration |
| Sensor terms $v_r$ | Relative scale of eight sensor regions | 8 | 8, subject to centering |
| Position terms $a_g$ | Residual source-position intensity scale | All analysis positions | Contributing fitted positions if enabled; otherwise none |

Pixels failing the per-iteration observation requirement are reset to $u_p=0$, so their pixel factor is unity, although their sensor factor can still differ from unity. Positions outside the fitting cache retain $a_g=0$ and therefore $f_g=1$. A bad-pixel mask is derived from residuals; its Boolean entries are not additional continuously refined parameters.

Let $P$ denote the number of eligible pixel terms, $B$ the number of directly supported profile bins, and $G$ the fitted position count. The **nominal coefficient count** is

$$
P+B+8+G
$$

when position gains are enabled. It is not merely eight gains, nor is the final 1,000-point XY curve another family of fit coefficients.

#### Current-scan example

A read-only audit of the live scan-100587 calculation gave:

- 310 raw frames, 36 distinct exact motor readings, and 31 tolerance groups, each containing 10 frames;
- 23 analysis positions contributing to the fitting interval;
- 9,182,326 cached pixel measurements;
- 509,013 base-unmasked pixels, all with at least eight cached observations;
- all 800 profile bins supported;
- 30 pixels classified as persistent bad-pixel candidates.

Thus **509,844** is the nominal maximum coefficient count for this full fit: $509{,}013+800+8+23$. The eight-observation audit counts cached measurements, not the accepted robust weights of the last update, so it is an upper bound on actively updated pixel terms. Likewise, the number of non-unity fitted pixels is not an independent parameter-count estimator.

### 5.3 Scale ambiguities and identifiability

Multiplying the common profile by a constant and dividing all sensor or position gains by that constant leaves the multiplicative predictions unchanged. Likewise, a sensor-wide offset can be exchanged between its pixel terms and its sensor term.

The implementation fixes these conventions by:

- transferring the weighted mean pixel-log offset within each sensor into that sensor term;
- centering the eight sensor logs to zero mean and transferring the offset to the profile;
- centering the fitted position logs to zero mean and transferring the offset to the profile.

The geometric means of sensor gains and fitted position gains are therefore unity. Under full support and with pixels eligible in all eight sensors, the centering conventions remove ten nominal coordinates: eight pixel/sensor offsets, one sensor/profile offset, and one position/profile offset. This would give $P+B+G-2$, or **509,834** coordinates in the illustrative upper-bound count.

This is **not a formal effective statistical degrees of freedom**. Observation eligibility changes, robust rejection, shrinkage, clipping, and unsupported bins complicate that concept. Priors also select a preferred decomposition. Most importantly, counting more measurements than coefficients does not establish identifiability: the pixel–position–$Q$ overlap must actually connect the unknowns. A systematic physical error or genuine scattering anisotropy can be absorbed by a response field even when the numerical fit appears stable.

## 6. Step 4: regularized robust alternating refinement

The fit initializes pixel, sensor, and position log gains to zero. The common profile is initialized from weighted logarithmic observations. It then performs **10 configured alternating iterations**, not a simultaneous unconstrained nonlinear least-squares solve and not an automatic convergence-to-tolerance procedure.

At each iteration the prediction is

$$
\widehat C_{gp}=\exp[h_b+u_p+v_r+a_g+\log(M_gK_{gp})],
$$

with the exponential argument numerically clipped to $[-10,30]$. The residual scale and standardized residual are

$$
\sigma_{\log,gp}=\sqrt{\frac{1}{\max(\widehat C_{gp},1)}+\delta^2},\qquad
z_{gp}=\frac{y_{gp}-(h_b+u_p+v_r+a_g)}{\sigma_{\log,gp}},
\qquad\delta=0.003.
$$

The 0.3% floor limits the influence of very high counts. The iteration weights are

$$
w_{gp}=\sigma_{\log,gp}^{-2}
\min\left(1,\frac{2}{\max(|z_{gp}|,10^{-12})}\right).
$$

Starting with the **third iteration**, weights are set to zero where $|z|>5$. Rejection is recomputed each iteration rather than maintained as a permanent observation blacklist.

Using those weights, the iteration proceeds in order:

1. **Profile:** update each supported $h_b$ by a weighted residual mean after removing the current gain terms. Interpolate unsupported bins in bin-index space; this is not a smoothness penalty on supported profile bins.
2. **Pixels:** for pixels with at least eight positive-weight observations, update the log term from its weighted residual sum divided by $\sum w+300$. This shrinks weakly constrained responses towards unity. Clip these preliminary logs to $[-0.15,0.15]$, then transfer sensor-wise mean offsets as described above.
3. **Sensors:** update each sensor log from residual sums, with a quadratic penalty equivalent to a 0.05 log-scale prior width: denominator $\sum w+1/0.05^2$. Clip preliminary logs to $[-0.10,0.10]$, then center them.
4. **Positions:** if enabled, update each fitted position log with denominator $\sum w+1/0.01^2$, clip preliminary values to $[-0.05,0.05]$, and center them.

Because centering follows clipping, these clipping intervals are bounds on the preliminary updates, **not strict final bounds on every returned centered log gain**. Priors and clipping are algorithmic stabilizers, not independently measured calibration uncertainties. The code records gain changes and rejected fractions, but ten iterations alone do not guarantee convergence or physical correctness.

After the last iteration, residuals are recomputed. A pixel becomes a persistent bad-pixel candidate if it has at least eight cached observations, at least three residuals beyond five scale units, and an outlier fraction of at least 20%. This fixed mask is used in final integration.

**Important distinction:** individual observations rejected during fitting are not automatically removed from the final export. The export applies the fixed bad-pixel mask and validity checks, but does not replay profile-dependent dynamic rejection.

## 7. Step 5: held-out checks and calibration persistence

With validation enabled, contributing analysis positions are split into alternating-index training and test sets, then the roles are reversed. Whole groups are held out; repeats within a group cannot leak separately into the opposite split. The illustrative scan uses training sets of 12 and 11 positions. These are two additional complete fits, followed by the full-data fit that supplies the export correction; their parameter estimates are not averaged to construct the final map.

Held-out integrations compare physical corrections alone, physical plus detector response, and an additional position gain estimated using the lower-$Q$ portion of the held-out data. Higher-$Q$ residuals test whether that scalar helps beyond the pixels used to estimate it. Adjacent-position overlap metrics test agreement in common supported bins. This probes reproducibility within this scan, not transferability to a different sample or scan. The notebook does not impose a numerical pass/fail threshold that blocks export after poor validation metrics.

The optional calibration HDF5 output stores detector divisors, masks, physical settings, geometry, fit history, profile, validation results, and source-position gains. It is written with exclusive creation and read back for consistency. The notebook can instead keep an equivalent correction payload in memory. Calibration status is explicitly exploratory.

## 8. Step 6: produce a new integrated pattern with the fitted response fixed

### 8.1 What is applied?

The export uses the original grouped count images, not the fitted $I_b$. For each group, the supplied corrected signal and raw-count variance are

$$
C'_{gp}=\frac{C_{gp}}{D_p},\qquad
V'_{gp}=\frac{C_{gp}}{D_p^2}.
$$

pyFAI then applies solid-angle, polarization, absorption, and monitor normalization consistently with the physical model. For this **same source scan**, the notebook explicitly enables the fitted position gains, so the scalar normalization includes $M_g f_g$ rather than $M_g$ alone. Thus a fitted $f_g>1$ reduces that position's normalized intensity.

This is a best-effort source-scan normalization. The transferable correction is the detector divisor and mask; source-position gains are **not** a detector flat field. The processing code requires source path, angles, and monitors to match before applying them. Positions not covered by the fitting interval have unity gains, even though their images can contribute to the exported pattern.

### 8.2 Output coordinates and range

The processing routine computes two integrations: an auxiliary $Q$ pattern over the fitting interval, and the final two-theta pattern. Only the latter supplies the XY file in this notebook.

The final two-theta integration uses `THETA_BIN_COUNT`, currently **1,000**, with a lower bound of **0°**. A numeric `TWO_THETA_MAX_DEG` sets the upper bound. If it is `None`, the upper bound is the maximum finite, unmasked pyFAI **pixel-centre** scattering angle over all analysis positions, using the final fixed mask. It is not simply the maximum motor angle, and it does not include the full angular extent of pixel corners.

For the audited calculation the resolved range was 0–67.33276184°. Bin centres in two-theta are uniformly spaced. Transforming them into $Q$ would produce nonuniform $Q$ spacing; the XY file itself remains in **two-theta degrees**.

The integrator uses the configured pyFAI bounding-box/CSR/Cython method, including fractional pixel contributions across bins. This differs from the single-bin pixel-centre assignments used for fitting and observation diagnostics.

### 8.3 Merge before dividing

For group $g$ and output bin $k$, pyFAI returns accumulated signal $S_{gk}$, normalization $N_{gk}$, propagated variance $V_{gk}$, and a fractional contribution count. The implementation sums these accumulators across positions and forms

$$
I_k^{\mathrm{merged}}=\frac{\sum_g S_{gk}}{\sum_g N_{gk}},\qquad
\sigma_k=\frac{\sqrt{\sum_g V_{gk}}}{\sum_g N_{gk}}.
$$

It does **not** take an unweighted average of the per-position integrated intensities. At a schematic level, the signal contains split contributions from $C/D$, while normalization contains their physical factors and $M_gf_g$; the code uses pyFAI's returned accumulators rather than reconstructing them independently.

The fitted detector response is applied outside 20–40 Å$^{-1}$ as well. This extrapolates the assumption that the response is independent of $Q$ and incidence conditions; it is not extra calibration evidence. The processing call explicitly permits this extrapolation.

## 9. Step 7: write the XY file and interpret its contents

Only bins with positive merged normalization and finite coordinate and intensity are selected. The notebook writes two columns, headed **2theta_deg intensity**, using approximately ten significant digits and exclusive file creation. Unsupported bins are omitted rather than filled with zeros; the number of exported rows can therefore be smaller than the requested number of bins. The audited run exported 1,000 rows.

The exported intensity includes:

- summed-repeat counting and summed-I0 normalization;
- the fixed geometry, solid angle, polarization, and CdTe absorption model;
- the fitted pixel and sensor divisor and fixed bad-pixel mask;
- fitted source-position gains where available;
- accumulation and merging across positions.

It does **not** include rolling-background subtraction, mean-centering, an uncertainty column, theoretical scattering corrections, or independent absolute-intensity calibration. Although an internal counting-noise $\sigma$ exists, it excludes uncertainty in the fitted calibration, I0, geometry, physical model, and inter-bin covariance from pixel splitting. Those omissions are especially relevant when judging fine residual features.

The temporary aggregated detector file is removed after the save/export stage. Input NeXus and companion data are opened read-only and remain unchanged. Existing calibration and XY files are not overwritten.

## 10. Interpretation and limitations

The final pattern should be regarded as the result of **a regularized self-calibration followed by fixed-correction integration**, not as a direct measurement corrected by a separately established detector calibration. In particular:

- A feature common to measurements at the same $Q$ is naturally represented in the fitted profile. A feature tied to a detector pixel is naturally represented in its response, but incomplete overlap can make this separation ambiguous.
- Unmodelled sample or instrument effects can be incorporated into a detector response estimated from the same sample. Independent-scan validation is needed before transferring it.
- Scalar position gains can improve overlap while concealing a real position-dependent amplitude change. They cannot repair a $Q$-dependent discrepancy.
- More repeats increase counting precision but, once aggregated, do not increase the number of independently located observations used to refine a pixel response.
- A flatter background-subtracted plot is not evidence of improved detector calibration: that baseline is estimated separately for visualization and never enters the XY export.

The strongest test of transferability would be to freeze the detector divisor and physical model, apply them to independent scans without carrying over the source-position gains, and evaluate overlap and residual structure without retuning the response.

---

## Appendix A. Diagnostic figures and what they establish

### A.1 Motor-angle repeats and tolerance

The first figure compares exact-repeat counts with group totals, plots angle offsets from each group minimum against the tolerance window, and enlarges the widest-span group. Its shaded bands describe a grouping rule, not an uncertainty in the motor angle. Representative angles are arithmetic means without I0 weights.

### A.2 Response maps, sensor scales, and source-position gains

The fitted pixel divisor and combined divisor maps reveal spatial structure in the correction. The colour limits are display limits, not bounds on fitted values. The sensor plot shows the eight centered relative gains. The position-gain plot shows source-only residual intensity scales. Structured maps or large gains require interpretation; they do not independently prove a detector defect.

### A.3 Pre-integration sampling versus Q

The two sampling panels use final two-theta bin edges transformed into Å$^{-1}$, with secondary two-theta axes. They use the final fixed mask and count-validity rules, and reread source counts read-only when necessary.

**Panel A** counts retained pixel × analysis-position observations per bin. For grouped data a second curve multiplies each group's retained pixel count by its raw-frame multiplicity. This is a count of constituent measurements represented in the retained sum at the group geometry, not recovered raw-frame geometry or additional independent motor positions.

**Panel B** counts positions supplying at least one valid pixel to each bin. Thousands of pixels at one position still count as one position. The dashed line marks the total position count. This is overlap coverage, not photon statistics, formal parameter identifiability, or an error bar.

Both panels assign a pixel centre once. They do not reproduce the fractional contribution counts returned by the final pyFAI integration. Large counts indicate redundancy; they do not measure the independence of all those measurements.

### A.4 Integrated pattern and baseline-subtracted display

The final three-panel figure shows the exact exported data, a high-angle zoom with an estimated background, and the same zoom after background subtraction. The default display interval is 30–85° and the rolling radius is 5°, corresponding to a roughly 10° full window.

The implemented baseline is a one-dimensional grey morphological opening (minimum then maximum filtering) followed by moving-average smoothing. It is a **rolling-ball-style approximation**, not a literal spherical kernel. The calculation interpolates to a uniform angular grid over the available full pattern before filtering and then returns to the exported coordinates. Missing intervals and edges can affect the estimate. The residual is intensity minus that baseline, without mean-centering or division by uncertainty; it is not required to have zero mean. None of these visualization operations changes the XY data.

## Appendix B. Reproducibility and implementation map

| Stage | Main implementation |
|---|---|
| Input loading and checks | `load_scan()` in [eiger_workflow.py](eiger_workflow.py) |
| Grouping and summed raw counts | `aggregate_equal_angle_frames()` in [eiger_workflow.py](eiger_workflow.py) |
| Fixed physical arrays | `PhysicalModel` in [eiger_workflow.py](eiger_workflow.py) |
| Fit observations and refinement | `build_cache()` and `fit_response()` in [eiger_workflow.py](eiger_workflow.py) |
| Held-out comparisons | `validate_response()` and `overlap_summary()` in [eiger_workflow.py](eiger_workflow.py) |
| Persisted correction | `save_calibration()` and `load_calibration()` in [eiger_workflow.py](eiger_workflow.py) |
| Automatic angular maximum | `theta_range_for_scan()` in [eiger_workflow.py](eiger_workflow.py) |
| Fixed-response integration and merging | `process_scan()`, `integrate_frame()`, and `merge_results()` in [eiger_workflow.py](eiger_workflow.py) |
| Two-column selection, export, figures | [extract_eiger_corrections_fast.ipynb](extract_eiger_corrections_fast.ipynb) |

To reproduce a particular XY file, retain the source scan identifier, geometry file content, software versions, grouping tolerance, physical constants, fitting range and bin count, regularization/rejection settings, final two-theta bin count and resolved range, and whether source-position gains were applied. The calibration HDF5 stores many of these, but the plain two-column XY header alone is not complete provenance. Runtime and memory diagnostics describe computational cost and do not alter the inferred response or exported intensity.
