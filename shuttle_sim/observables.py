"""Experiment-facing observables pipeline.

Spec refs: §10 observables, §8 loss hierarchy, §11 raw→observable→loss.

Even when no hardware is attached, the observables are defined as *physical*
proxies that would be measured in experiment (sideband spectroscopy, recooling,
displacement). The simulation therefore provides a consistent mapping:

raw measurement (simulated: n_total, coherent displacement, etc.)
  → processed observable (sideband amplitude ratio, asymmetry, recooling proxy)
  → scalar loss / robust aggregate

All conversions are documented and use SI inputs.

Implemented observables (literature-consistent):
- red_sideband_proxy / blue_sideband_proxy : ∝ thermal occupation
- sideband_asymmetry : (n_blue - n_red)/(n_blue + n_red) ≈ n/(n+1) for thermal
- first/second order sideband proxies
- recooling_proxy : motional energy in units of quanta
- displacement_sensitive : coherent amplitude proxy

The scalar loss is log-scaled and robust-aggregated (mean + worst-case) exactly
as in pipeline.optimize().
"""
import numpy as np
from .field import finite

def observables_from_simulation(sim_metrics: dict) -> dict:
    """Convert validated dynamics output to experiment-style observables.

    Inputs are the propagate() summary (n_modes, coherent_n, position errors, etc.)
    Returns a dict of physically meaningful quantities that could be inferred
    from sideband spectroscopy or recooling.
    """
    n_total = float(sim_metrics["n_total"])
    n_modes = np.asarray(sim_metrics["n_modes"])
    coherent = float(sim_metrics.get("coherent_n_total", n_total))
    # Thermal-like proxies: for low n, sideband ratio ≈ n/(n+1)
    # We simulate "measured" sideband amplitudes as:
    #   A_red ∝ n, A_blue ∝ n+1  (thermal, Lamb-Dicke)
    # Asymmetry η = (A_blue - A_red)/(A_blue + A_red)
    # For coherent states, sideband ratio differs; we use coherent fraction.
    # These are proxies, not fitted spectra, but mathematically tied to n.
    # Clip to avoid division by zero
    eps = 1e-12
    # Simulated first-order sideband "amplitudes" (arb. units)
    # Normalize so asymmetry maps to n
    # We choose A_red = n, A_blue = n+1  → asymmetry = 1/(2n+1)
    A_red = n_total
    A_blue = n_total + 1
    asymmetry = (A_blue - A_red) / (A_blue + A_red + eps)  # =1/(2n+1)
    # Second-order proxy (scales as n^2 for thermal)
    A_red_2 = 0.5 * n_total**2
    A_blue_2 = 0.5 * (n_total + 1)**2 + 0.5 * n_total**2

    # Recooling proxy: energy removed during recooling ∝ total excitation
    recooling_energy_quanta = n_total  # in quanta, IS the motional energy
    recooling_proxy = recooling_energy_quanta  # directly

    # Displacement-sensitive proxy: for coherent displacement d, sideband oscillates as J0/J1
    # Coherent fraction = coherent / (coherent + incoherent + eps)
    # If coherent dominates, displacement sensor ~ sqrt(coherent)
    cov_n = float(sim_metrics.get("covariance_n_total", 0))
    coherent_frac = coherent / (n_total + eps)
    displacement_alpha = np.sqrt(max(coherent, 0))  # |alpha| ≈ sqrt(n_coh)

    # Per-mode breakdown
    per_mode = []
    for i, n in enumerate(n_modes):
        per_mode.append({
            "mode": int(i),
            "n": float(n),
            "sideband_asymmetry_proxy": float(1/(2*n+1)) if n+0.5>eps else 1.0,
        })

    return {
        "n_total": n_total,
        "n_modes": n_modes.tolist(),
        "coherent_n_total": coherent,
        "sideband": {
            "red_proxy": float(A_red),
            "blue_proxy": float(A_blue),
            "asymmetry": float(asymmetry),  # experimental observable
            "second_order_red_proxy": float(A_red_2),
            "second_order_blue_proxy": float(A_blue_2),
            "per_mode": per_mode,
        },
        "recooling_proxy_quanta": float(recooling_proxy),
        "displacement_proxy_alpha": float(displacement_alpha),
        "coherent_fraction": float(coherent_frac),
        "position_error_m": float(sim_metrics.get("position_error_m", 0)),
        "well_target_error_m": float(sim_metrics.get("well_target_error_m", 0)),
        "mapping": "raw simulation n_total → sideband asymmetry =1/(2n+1) (thermal proxy), recooling ∝n, displacement ∝sqrt(n_coh)",
        "provenance": "simulation proxies; experiment would fit actual sideband Rabi flops",
    }

def scalar_loss_from_observables(obs: dict, task: dict) -> float:
    """Scalar loss mirroring pipeline.optimize() but expressed via observables.

    Hierarchy:
      1. infeasible -> large penalty (handled upstream via Rejected)
      2. excitation (log1p n/n0) + position errors
      3. robust aggregation outside
    """
    n = float(obs["n_total"])
    pos = float(obs["position_error_m"])
    well = float(obs["well_target_error_m"])
    n0 = float(task["max_n_total"])
    tol = float(task["position_tolerance_m"])
    val = np.log1p(n / n0)
    val += np.log1p((pos / tol) ** 2)
    val += np.log1p((well / tol) ** 2)
    return float(val)

def processing_chain(sim_metrics: dict, task: dict) -> dict:
    """Full chain raw→observable→loss for one evaluation."""
    obs = observables_from_simulation(sim_metrics)
    loss = scalar_loss_from_observables(obs, task)
    return {"observables": obs, "loss": loss, "provenance": obs["mapping"]}

def aggregate_losses(losses: list, mode="weighted_worst"):
    """Explicit aggregation policy.

    Supported:
      - weighted_worst: 0.35*mean +0.65*max  (pipeline default)
      - worst_case: max
      - mean: mean
      - cvar_20: tail mean of worst 20%
    """
    arr = np.asarray(losses, float)
    if mode == "weighted_worst":
        return float(0.35*arr.mean() + 0.65*arr.max())
    if mode == "worst_case":
        return float(arr.max())
    if mode == "mean":
        return float(arr.mean())
    if mode == "cvar_20":
        k = max(1, int(np.ceil(0.2*len(arr))))
        return float(np.sort(arr)[-k:].mean())
    raise ValueError(f"unknown aggregation {mode}")
