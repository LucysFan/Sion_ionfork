"""Island GA package for shuttling.

Exposes Genome, Island, GAEngine, and CUDA helpers.
Simulation-tunable: theta 8-10; experiment-tunable: same theta via closed-loop.
"""
from .genome import Genome, BOUNDS_8, BOUNDS_10
from .island import Island
from .engine import GAEngine
from .cuda import cuda_info

__all__ = ["Genome","BOUNDS_8","BOUNDS_10","Island","GAEngine","cuda_info"]
