# Shuttle Pipeline – Architecture Mapping to Specification v2

> This document maps every mandatory architectural layer from `planar_trap_shuttling_agent_specification_v2.pdf` to concrete code, artifacts and runnable configs.

## 1. Physical Baseline (spec §2)

- **Minimum model**: 3D local harmonic `field.Local` (position, Hessian, secular frequencies, normal modes) used everywhere. Full RF micromotion / non-Gaussian propagation explicitly **not** modelled – declared in `manifest.json` and `validation.json` as limitation.
- **SI units**: all public APIs accept meters, seconds, volts, kg, C. Internal scaling `L=1 µm, T=1 µs` only inside `dynamics.py`. `finite()` validates inputs.

## 2. Eight Mandatory Layers (spec §3.8)

| Layer | Module | Key Entry |
|---|---|---|
| 1 Field model | `shuttle_sim/field.py` – `SionTrap` + `load_trap()` | `basis()`, `rf()`, `local()`, `invert()` |
| 2 Control synthesis | `shuttle_sim/control.py` | `TrajectoryParameterization`, `FrequencyProfile` (fixed/variable), `VoltageKnotParameterization`, `CorrectiveBasis` |
| 3 Voltage inversion | `shuttle_sim/voltage.py` + `SionTrap.invert()` | hard 3D force balance + axial curvature, bounds, regularization, null-space, feasibility check |
| 4 Hardware execution | `shuttle_sim/hardware.py` – `Hardware` | DAC quantisation, ZOH, exact RC cascade (expm), gain/offset, delay/skew, slew clipping |
| 5 Dynamics validation | `shuttle_sim/dynamics.py` | `propagate()` exact piecewise affine vs DOP853, Gaussian covariance, `validate_full_field_centroid()` non-linear |
| 6 Robust evaluation | `shuttle_sim/pipeline.py` – `Runner.robust()` | ensemble over stray, gain, offset, delay, filter, frequency_scale; mean/worst/CVaR |
| 7 Calibration / identification | `shuttle_sim/calibration.py` | `Calibration.identify_from_residuals()`, `identify_filter()`, `update_model()` |
| 8 Closed-loop | `shuttle_sim/closed_loop.py` | `ClosedLoopRefiner.optimize()` – 8-dim, safety-gated, budgeted, history |

All layers are **separate files** with explicit imports – not a single opaque notebook cell.

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

1. **Trajectory-based**: `control.TrajectoryParameterization` – polynomial time-warp on `16s²(1-s)²` envelope, preserves boundary v=a=0.
2. **Frequency / confinement**: `control.FrequencyProfile` – `fixed` (constant `omega0`) vs `variable` (`omega(s)=omega0·(1+ a·env + b·env·(2s-1))`). Pipeline `prepare()` switches on `task.frequency_mode`.
3. **Voltage knot**: `control.VoltageKnotParameterization` – CubicSpline knots at decimated route nodes.
4. **Corrective basis**: `control.CorrectiveBasis` – `theta[0:2]` time-warp + `theta[2:8]` 3 envelopes × 2 SVD geometry modes (top-2 modes of `route_V`). This is the **low-dim (8)** closed-loop space (spec §12).

`pipeline.Runner.commands()` uses (1)+(4) plus baseline choice `smooth` (10s³-15s⁴+6s⁵) vs `sta` (35s⁴-84s⁵+70s⁶-20s⁷ + s''/(Tω)²).

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

`optimize()` loss: `log1p(n/n0)+log1p((pos/tol)²)+log1p((well/tol)²)` aggregated `0.35·mean+0.65·max` + `1e-4·||theta||²` (P1/B1 unified, weights `rerank_weights` default `{n:1,pos:1,well:1}`). Metrics saved: `n_total`, `n_modes`, `coherent`, `covariance_n`, `position_error_m`, `well_target_error_m`, `max_transport_error_m` (`is_max_transport_error_fallback` flag, P2 — max over trajectory vs well/nominal, not just final error), `phase_space_displacement`, `minimum_frequency_hz`.

Candidate reranking (B2/P3): all baselines (`sta`+`smooth` zeros + `unaware`+`aware` optimized) evaluated on **identical** `exact_nodes`/`subdivisions` with same normalized `log1p` score; tie `eps 1e-9` prefers `sta` (lexicographic). Flag `optimization_improved_over_baseline` computed honestly from `rerank_full`.

Aggregation policies in `observables.aggregate_losses()`: `weighted_worst`, `worst_case`, `mean`, `cvar_20`.

Hierarchy (spec §8): `safety` → `confinement` → `hardware realizability` → `excitation` → `speed`.

### Robust (spec §10)

`Runner.scenarios()` perturbs `gain (log)`, `offset`, `delay (uniform)`, `lowpass (log)`, `stray (3D)`, `frequency_scale (log)` seeded RNG. Reports `mean_n`, `worst_n`, `cvar_20pct_n`, `pass_fraction` (all must be `accepted`).

### Calibration (spec §9–10)

`Calibration` is **separate** from closed-loop optimizer:
- `identify_from_residuals(dataset)` – affine fit `meas ≈ gain·cmd+offset` per channel; `identify_filter(step)` – tau from step response.
- Dataset must be provided; empty → `blocked_by_missing_external_data` (no fake solution).
- `update_model(result)` returns new `Hardware`; pipeline demo after identification re-evaluates with updated hardware and saves `calibration_demo.json` + `calibration_blocked.json` showing both paths.

Sequence (spec §10): `identification → update_model → resynthesis → optional closed-loop fine-tuning` – implemented in pipeline (synthetic demo: identification on 3 thetas, then closed-loop uses perturbed truth).

### Observables (spec §10)

`shuttle_sim/observables.py`:
- `observables_from_simulation()` maps `n_total, coherent, n_modes` → `sideband.red/blue_proxy`, `asymmetry =1/(2n+1)`, `second_order`, `recooling_proxy=n`, `displacement_alpha=√n_coh`, per-mode breakdown.
- `processing_chain()` = `raw→obs→loss` with provenance.
- Scalar loss mirrors optimizer loss (log1p) – no abstract “experiment score”.

### Closed-Loop (spec §11)

`ClosedLoopRefiner`:
- input: `theta` (8-dim) + `Hardware` + `measure_fn`
- `measure_fn(theta)->{loss, observables}` – in production would call apparatus; in repo an **emulated** version uses perturbed `Hardware` (declared provenance, not fake).
- `optimize()` – Powell, `budget_evaluations`, bounds `(-0.1,0.1) ×2 + (-0.4,0.4)×6`, safety gating via `SafetyLayer`, full history logged, high-dim insertion rejected (tested).

### Safety (spec §13)

`SafetyLayer` checks before export:
- voltage bounds (`trap.lo/hi`), slew (`hw.max_jump_V`), clipping, filter deviation, `trap.local` existence, `weak_confinement` (frequency floor), numerics (`finite`). Returns `SafetyReport(passed, checks, failure_code)`. `fallback_state.json` holds last known safe `theta=0`.

### Failure Classification (spec §14)

`field.Rejected` codes: `no_minimum`, `weak_confinement`, `infeasible_voltages`, `hardware_saturation`, `slew_limit`, `route_corridor`, `rom_trust_region`, `excessive_excitation`, `transport_accuracy`, `well_target_accuracy`, `numerical_failure`, `optimizer_budget_exhaustion`, `calibration_inconsistency`, `missing_external_data`, `integrity_failure`, etc. – all logged, not just traceback.

### Reproducibility (spec §15–16)

Artifacts per run (`output/*`):
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

Seeds fixed (`uncertainty.seed`).

### Evidence (spec §17)

Every run shows:
1. nominal open-loop (zero theta smooth/sta)
2. hardware-aware waveform (delivered)
3. independent validation (exact vs dop853, refined)
4. robust ensemble (6–8 scenarios, mean/worst/CVaR)
5. closed-loop-ready low-dim (8) + history
6. complete round-trip (reference) + one-way/repeated/gds runs
7. saved artifacts reproducible.

