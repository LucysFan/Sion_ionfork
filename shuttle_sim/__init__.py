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
# Numba JIT for sion/electrode/expressions hangs with numpy 2.4+ (compiler_lock deadlock) — disable JIT globally for stability.
# This makes sion run in object mode (slower but correct) and prevents load_trap from hanging on first import.
import os as _os
_os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
from . import field, hardware, dynamics, control, voltage, safety, observables, calibration, closed_loop, transport
__all__ = ['field','hardware','dynamics','control','voltage','safety','observables','calibration','closed_loop','transport']
