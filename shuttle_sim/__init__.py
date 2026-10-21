"""Simulation-only, geometry-aware single-ion transport. No apparatus I/O.

Layers (spec §3.8):
- field (TrapAdapter, SionTrap)
- control (parameterizations)
- voltage (inversion layer)
- hardware (forward model)
- dynamics (validation)
- safety (gates)
- observables (experiment-facing loss)
- calibration (identification)
- closed_loop (safe low-dim refinement)
- transport (task entity)
"""
__version__ = '0.3.0'
from . import field, hardware, dynamics, control, voltage, safety, observables, calibration, closed_loop, transport
__all__ = ['field','hardware','dynamics','control','voltage','safety','observables','calibration','closed_loop','transport']
