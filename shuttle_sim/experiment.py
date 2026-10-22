"""Experiment-facing interface (spec: experiment-integrable level).

A real apparatus is attached by supplying `transport(theta, waveform_meta) ->
{'loss': float, 'observables': {...}}`. Without a transport the interface is
honestly blocked: it raises Rejected('experiment_communication_failure'),
never a fabricated score. Measured observables are validated before use;
malformed/non-finite observables raise
Rejected('invalid_observable_extraction').
"""
import numpy as np
from .field import Rejected, finite

REQUIRED_OBSERVABLE_KEYS = ()

def validate_measured_observables(observables):
    if observables is None:
        return {}
    if not isinstance(observables, dict):
        raise Rejected('invalid_observable_extraction','observables must be a mapping')
    for k,v in observables.items():
        if isinstance(v,(int,float,np.floating)) and not np.isfinite(v):
            raise Rejected('invalid_observable_extraction',f'observable {k} is non-finite')
    return observables

class ExperimentInterface:
    """Callable transport adapter for ClosedLoopRefiner.measure_fn."""
    def __init__(self, transport=None):
        self.transport = transport
    @property
    def connected(self):
        return self.transport is not None
    def measure(self, theta):
        if self.transport is None:
            raise Rejected('experiment_communication_failure',
                           'blocked_by_missing_external_data: no apparatus transport attached')
        try:
            out = self.transport(np.asarray(theta,float))
        except Rejected:
            raise
        except Exception as ex:
            raise Rejected('experiment_communication_failure', f'apparatus transport failed: {ex}')
        if not isinstance(out, dict) or 'loss' not in out:
            raise Rejected('invalid_observable_extraction','apparatus must return {loss, observables}')
        if not np.isfinite(float(out['loss'])):
            raise Rejected('invalid_observable_extraction','apparatus loss is non-finite')
        validate_measured_observables(out.get('observables',{}))
        return out
    __call__ = measure

class BlockedExperiment(ExperimentInterface):
    """Explicit blocked state used when no apparatus is attached."""
    def __init__(self):
        super().__init__(transport=None)
