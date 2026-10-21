"""Voltage inversion layer – trajectory → electrode voltages.

Spec refs: §5 voltage inversion, §8 hierarchy.

- Hard constraints: 3D force balance + axial curvature
- Handles bounds, regularization, null-space freedom, temporal smoothness
- Feasibility check: verifies that local minimum actually lands at target
- Laplacian freedom: radial curvatures not prescribed (DC Hessians obey ∇²Φ=0)

Public entry: SionTrap.invert() already implements this.
This module documents the layer and provides a wrapper with extra diagnostics.
"""
import numpy as np
from .field import finite, Rejected

def invert_with_diagnostics(trap, target_m, axial_direction, axial_omega_rad_s, previous_V, reg=1e-4):
    """Thin wrapper around trap.invert with provenance."""
    v, local = trap.invert(target_m, axial_direction, axial_omega_rad_s, previous_V, regularization=reg)
    return {
        "voltage_V": v,
        "local": local,
        "residual_force_N": float(np.linalg.norm(trap.energy_derivative(local.position_m, v, 1))),
        "target_error_m": float(np.linalg.norm(local.position_m - target_m)),
        "axial_frequency_hz": float(local.omega_rad_s[0]/(2*np.pi)),
        "regularization": reg,
    }
