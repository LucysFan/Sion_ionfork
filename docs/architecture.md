# Shuttle Pipeline – Architecture Mapping to Specification v2

> This document maps every mandatory architectural layer from `planar_trap_shuttling_agent_specification_v2.pdf` to concrete code, artifacts and runnable configs.

## 1. Physical Baseline (spec §2)

- **Minimum model**: 3D local harmonic `field.Local` (position, Hessian, secular frequencies, normal modes) used everywhere. Full RF micromotion / non-Gaussian propagation explicitly **not** modelled – declared in `manifest.json` and `validation.json` as limitation.
- **SI units**: all public APIs accept meters, seconds, volts, kg, C. Internal scaling `L=1 µm, T=1 µs` only inside `dynamics.py`. `finite()` validates inputs.

## 2. Eight Mandatory Layers (spec §3.8)

| Layer | Module | Key Entry |
|---|---|---|
| 1 Field model | `shuttle_sim/field.py` – `SionTrap` + `load_trap()` | `basis()`, `rf()`, `local()`, `invert()` |
| 2 Control synthesis | `shuttle_sim/control.py` | `Runner.commands()` uses time warp + corrective voltage basis; frequency profile comes from task settings; `VoltageKnotParameterization` is a standalone helper |
| 3 Voltage inversion | `shuttle_sim/voltage.py` + `SionTrap.invert()` | hard 3D force balance + axial curvature, bounds, regularization, null-space, feasibility check |
| 4 Hardware execution | `shuttle_sim/hardware.py` – `Hardware` | DAC quantisation, ZOH, exact RC cascade (expm), gain/offset, delay/skew, slew clipping |
| 5 Dynamics validation | `shuttle_sim/dynamics.py` | `propagate()` exact piecewise affine vs DOP853, Gaussian covariance, `validate_full_field_centroid()` non-linear |
| 6 Robust evaluation | `shuttle_sim/pipeline.py` – `Runner.robust()` | ensemble over stray, gain, offset, delay, filter, frequency_scale; mean/worst/CVaR |
| 7 Calibration / identification | `shuttle_sim/calibration.py` | `Calibration.identify_from_residuals()`, `identify_filter()`, `update_model()` |
| 8 Closed-loop | `shuttle_sim/closed_loop.py` | `ClosedLoopRefiner.optimize()` – 8-dim, safety-gated, budgeted, history |

The modules are separate files, but they are not separate services: `Runner` calls the synthesis, delivery, dynamics, robust evaluation, calibration, and closed-loop steps sequentially in the same process. The dashboard HTTP server may be run separately; it is a viewer/control interface, not a worker that distributes the physical pipeline.

### TrapAdapter (spec §4)

`SionTrap` works with any `electrode.System`:
- `sion_five_wire` – analytic five-wire geometry via `sion.five_wire_trap_design()`
- `gds` – reads GDS via `gdspy` primary or pure-Python `python-gdsii` fallback, requires explicit `electrode_map` (no implicit polygon order). Pseudopotential assembled as `RF + Σ V_i * basis_i`.

### Canonical Path (spec §4)

- **user route**: `start + s·distance·direction` (config `task.direction` + `distance_m`)
- **nominal well path**: `voltage_spline(s)` → `trap.local()` → `route_r`, `route_H`, `support_diagnostics`
- **realized path**: delivered voltages → `rom()` exact reconstruction (`r`, `H`) → `propagate()`

Convention documented in `path_conventions.json`.

### Transport Task (spec §4)

`shuttle_sim/transport.py` – `TransportTask` dataclass with `one_way` / `round_trip` / `repeated` (cycles), holds, corridor, durations scan, tolerances. Validated at construction.

### Parameterizations (spec §5)

1. **Trajectory-based**: `control.TrajectoryParameterization` describes a polynomial time-warp on the `16s²(1-s)²` envelope; `Runner.commands()` builds the same form inline and preserves endpoint position and slope.
2. **Frequency / confinement**: `control.FrequencyProfile` – fixed or variable profile, configured separately through `task.frequency_mode` and `task.frequency_coeffs`; the summed modulation in `Runner.prepare()` is clipped to ±18% of the base frequency. Frequency coefficients are not GA genes.
3. **Voltage knot**: `control.VoltageKnotParameterization` – CubicSpline knots at decimated route nodes; available as a helper, but not used by `Runner.commands()`.
4. **Corrective basis**: `control.CorrectiveBasis` – used by `Runner.commands()` for 6/8/10 voltage-correction coefficients (3/4/5 envelopes × 2 SVD modes); with two time-warp coefficients this gives GA genome dimensions 8/10/12. The closed-loop optimizer remains fixed at 8 parameters (spec §12).

`pipeline.Runner.commands()` uses (1)+(4) plus baseline choice `smooth` (`10s³-15s⁴+6s⁵`) or `sta` (`35s⁴-84s⁵+70s⁶-20s⁷`). For a warped time law `σ(s)`, the STA acceleration term differentiates the composition: `[P''(σ)(σ')² + P'(σ)σ'']/(Tω₀)²`.

### Voltage Inversion (spec §5)

`SionTrap.invert(target, axis, omega, prev)`:
- solves `G·ΔV = -F_RF` (3) + `aᵀ H a = mω² - aᵀ H_RF a` (1) with `SLSQP`, bounds, `regularization·||V||²`, scale-normalized `A,b`, checks `||local - target||<5 nm`.

### Hardware Forward Model (spec §6)

`Hardware.deliver(commands, subdivisions)`:
- validates `sample_rate`, `dac_bits ∈ [2,32]`, `lowpass>0`, `gain>0`, `delay≥0`
- ZOH at `k/fs`, LSB quantisation, clipping → `Rejected('hardware_saturation')`, slew → `Rejected('slew_limit')`
- **exact** continuous RC cascade via `expm(M·dt)` on dense ZOH grid, DC steady-state prehistory, per-channel `gain·held+offset`, per-channel `interp(t-delay)`.
- **timing**: `trigger_latency_s` (static common shift) + `jitter_s` as **per-sample global time jitter** `rng.uniform(-jitter, jitter, len(t))` applied as `interp(t-delay-trig-time_jitter, t, y)` (P4/B16); old `jitter_per_channel_s` renamed to `skew_per_channel_s` (static skew), full `time_jitter_s`, `max_time_jitter_s`, `jitter_model` in report.
- distortion report: `max_deviation_V`, `rms`, `channel_max`, `lsb`, `delay`, `trigger_latency_s`, `jitter_s`, `skew`, `time_jitter`, `hold`, `filter`, `hardware_identifier`.

### Waveform Postprocessing (spec §6)

`Runner.commands()` → `Hardware.deliver()` is the explicit step: continuous param → sample-wise `cmd` → resample on `1/fs` grid (zero_order), interpolation policy (`linear` for ROM, `zero_order` for ideal), clipping repair (reject), delivered estimate, CSV export + SHA256 metadata.

### Timing & Sync (spec §7)

- common ideal clock, per-channel `delay_s`, `trigger_latency` in metadata, holds explicit, filter state persists through legs/cycles, `grid()` builds one contiguous `t` array with direction flips.

### Local Reduced-Order Model (spec §7)

`sroute` (41 support nodes, -0.05…1.05), `voltage_spline`, `Hspline`, `Gspline`, `BHspline`, plus full `trap.local()` reconstruction when `exact_nodes` requested. Diagnostics per node: position, `omega/(2π)`, modes, overlap, `mode_rotation_risk` (`overlap<0.95`), `relative_min_gap`.

### Multi-Fidelity & Independent Validation (spec §8)

- **cheap**: linear-response ROM (`H = Hspline + dv·BHspline`) + `propagate(method='exact', hold='linear')`
- **fine**: exact nodes reconstruction (`trap.local` at `comp_nodes`/`valid_nodes`/`refined_nodes`) + `propagate`
- **independent**: `method='dop853'` vs `exact` (different integrator, same physics) – documented in `validation.json`. Limitation “No full RF micromotion” explicit. Tolerance `validation_n_tolerance` saved.

### Objective & Metrics (spec §8)

`Runner.optimize()` evaluates the nominal case and, only when `robust=True`, the configured training scenarios. For standard weights `w_n=w_pos=w_well=1`, `w_lag=0`, each scenario score is `ℓ_j = log1p(n_j/n0) + log1p((e_pos,j/tol)²) + log1p((e_well,j/tol)²)`, where `n0=task.max_n_total` and `tol=task.position_tolerance_m`. The objective is `0.35·mean_j(ℓ_j) + 0.65·max_j(ℓ_j) + λ·mean_i[((θ_i−c_i)/s_i)²]`; `s_i` is half the allowed range of coefficient `i`, `c_i` its center, and `λ=1e-4` by default. The weights come from `loss_weights` (defaults `{n:1,pos:1,well:1,lag:0}`); the older `rerank_weights` key is only a compatibility fallback. A lag term is included only when explicitly weighted and a valid path metric exists. Rejected evaluations use `rejection_penalty()` rather than the smooth objective. Metrics include `n_total`, `n_modes`, `coherent`, `covariance_n`, `position_error_m`, `well_target_error_m`, `max_transport_error_m` (`is_max_transport_error_fallback` marks whether it is a true path maximum rather than an endpoint fallback), `phase_space_displacement`, and `minimum_frequency_hz`.

Candidate reranking in `Runner.run()` evaluates four candidates (zero `sta`, zero `smooth`, and the `unaware`/`aware` optimized `sta` candidates) on the same `exact_nodes` and `subdivisions`, using the same weighted loss plus normalized control penalty. Scores are ordered by a `1e-9` rounded key, with the `sta` baseline preferred for tied keys and position error as the next key. If every candidate receives a rejection penalty, the run fails with `all_candidates_rejected`. `optimization_improved_over_baseline` is reported separately from the selected candidate.

Aggregation policies in `observables.aggregate_losses()`: `weighted_worst`, `worst_case`, `mean`, `cvar_20`.

Hierarchy (spec §8): `safety` → `confinement` → `hardware realizability` → `excitation` → `speed`.

### Robust (spec §10)

`Runner.scenarios()` perturbs `gain (log)`, `offset`, `delay (uniform)`, `lowpass (log)`, `stray (3D)`, and `frequency_scale (log)` from a seeded, declared distribution. `Runner.robust()` reports `mean_n`, `worst_n`, `cvar_20pct_n`, and `pass_fraction` (the fraction with `accepted=true`). `all_scenarios_valid` instead indicates that every scenario returned metrics; `worst_n` and CVaR are withheld if a scenario calculation failed. These are simulated uncertainty scenarios, not a measured apparatus distribution.

### Calibration (spec §9–10)

`Calibration` is **separate** from the closed-loop optimizer:
- `identify_from_residuals(dataset)` fits affine `meas ≈ gain·cmd+offset` per channel; `identify_filter(step)` can estimate a time constant from a supplied step response.
- An empty dataset returns provenance `blocked_by_missing_external_data` and failure code `missing_external_data`; it does not fabricate an estimate.
- In `Runner.run()`, the calibration demonstration generates three command/delivery pairs from a declared synthetic hardware model (`gain=1.02`, `offset=0.015 V`, `delay=5 ns`). The residual fit identifies gain and offset only; filter poles and delay are retained, not estimated. It then separately resynthesizes and safety-checks with the updated model. `calibration_demo.json`, `calibration_resynthesis.json`, and `calibration_blocked.json` record these paths. This is a code-path demonstration, not experimental calibration.

The closed-loop demonstration is a separate emulated run using perturbed hardware as its synthetic measurement truth; it is not a refinement that consumes the calibration result. The repository does not execute the combined sequence on a real apparatus.

### Observables (spec §10)

`shuttle_sim/observables.py`:
- `observables_from_simulation()` maps `n_total, coherent, n_modes` → `sideband.red/blue_proxy`, `asymmetry =1/(2n+1)`, `second_order`, `recooling_proxy=n`, `displacement_alpha=√n_coh`, per-mode breakdown.
- `processing_chain()` = `raw→obs→loss` with provenance.
- Scalar loss mirrors optimizer loss (log1p) – no abstract “experiment score”.

### Closed-Loop (spec §11)

`ClosedLoopRefiner` is constructed with the `Runner`, `SafetyLayer`, and task; `optimize(measure_fn, initial_theta, bounds=None, budget=...)` accepts exactly eight coefficients. `measure_fn(theta)` returns a dictionary containing at least `loss` and optionally `observables`. In `Runner.run()`, the demonstration emulates measurements by evaluating a declared perturbed `Hardware`; no apparatus is called. Powell search uses bounds `(-0.1,0.1) ×2 + (-0.4,0.4)×6` by default. Safety is checked before each authorized measurement; a safety-rejected candidate does not consume the measurement budget, and the full history is logged. High-dimensional input is rejected before measurement (tested).

### Safety (spec §13)

`SafetyLayer` applies a fail-closed gate before closed-loop measurements and the final simulation-only authorization decision:
- delivered-voltage bounds (`trap.lo/hi`), command slew (`hw.max_jump_V`), local-minimum existence, frequency floor, and finite numerics;
- filter-intent check compares `max_deviation_V` with the first configured key among `distortion_threshold_V`, `max_distortion_V`, and `distortion_threshold` (default 0.25 V). Exceeding it returns `delivered_intent_violated`. `Runner.run()` can attempt a limited resynthesis and repeats the safety check; failure remains blocked. This hard threshold is not applied as a separate per-genome filter in the main GA loop; strategy H instead has a soft RMS-distortion term. The repository still sets physical `export_authorized=false`. `fallback_state.json` holds the last modeled safe state when one is available.

### Failure Classification (spec §14)

Observed rejection, quality, and run-status labels include `no_minimum`, `weak_confinement`, `infeasible_voltages`, `hardware_saturation`, `slew_limit`, `route_corridor`, `corridor_violation`, `rom_trust_region`, `delivered_intent_violated`, `excessive_excitation`, `transport_accuracy`, `well_target_accuracy`, `numerical_failure`, `all_candidates_rejected`, `calibration_inconsistency`, `missing_external_data`, `integrity_failure`, and `optimizer_budget_exhaustion_or_stagnation`. These are not all `field.Rejected` exceptions: some are quality flags or report statuses, and the run records them with the relevant metrics.

### Reproducibility (spec §15–16)

Key artifacts from a full `Runner.run()` (`output/*`; conditional branches may add further files):
- `config.json` (machine-readable)
- `trap_snapshot.json`, `route_diagnostics.json`, `transport_task.json`, `path_conventions.json`
- `ideal_waveform.csv`, `dac_waveform.csv`, `delivered_waveform.csv`, `delivered_path.csv`, `delivered_local_diagnostics.json`
- `comparison.json` (smooth/sta/ideal/delivered/unaware/aware)
- `validation.json` (cheap vs fine vs dop853, deltas, tolerance, independence statement)
- `robust.json`, `hold_scan.json`, `duration_scan.json`, `safety_report.json`, `fallback_state.json`
- `observables.json`, `observable_loss.json`, `observable_aggregation.json`
- `calibration_demo.json`, `calibration_blocked.json`, `calibration_params.json`
- `closed_loop_emulated.json`, `closed_loop_param_check.json`
- `waveform_metadata.json` (channel_order, sample_period, units, hw_identifier, SHA256, safety, closed_loop dim)
- `manifest.json` (package, python/numpy/scipy versions, platform, elapsed, input SHA256, source SHA256 per file)

Scenario generation uses `uncertainty.seed`; synthetic raw-count generation uses the separate `synthetic_measurement_seed` and `synthetic_measurement_shots` settings. The synthetic calibration and closed-loop perturbations are declared in code and their provenance is written to artifacts.

### Evidence (spec §17)

A full `Runner.run()` records, for the single configuration selected for that run:
1. zero-parameter `smooth` and `sta` open-loop comparisons;
2. unaware/aware search candidates and a hardware-delivered selected waveform;
3. independent integrator and refined-grid validation;
4. the configured held-out robust ensemble (default count 8; mean, worst, CVaR, and pass fraction);
5. an eight-parameter emulated closed-loop attempt with history and a dimension-rejection check;
6. the selected transport protocol and geometry, recorded in `transport_task.json` and `trap_snapshot.json`.

One-way, round-trip, repeated-cycle, five-wire, and GDS examples are separate configurations/runs; no individual run silently covers every protocol or geometry. The output contains the per-run artifacts listed above and in the manifest.

