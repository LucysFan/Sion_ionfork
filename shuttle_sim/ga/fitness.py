"""Fitness — hierarchical, multi-objective, cheap ROM for GA.

Uses Runner.evaluate with cheap ROM (exact_nodes=None) for speed.
Full exact evaluation only for hall of fame.

Loss components are logged separately for dashboard.
"""
import numpy as np
from ..field import Rejected, finite
from ..observables import processing_chain

def hierarchical_fitness(metrics, task, distortion=None, theta=None):
    """Scalar fitness + components for dashboard.

    Returns dict with scalar 'fitness' and all sub-metrics.
    """
    # metrics is Runner.evaluate result (propagation summary)
    n = float(metrics.get('n_total', 1e6))
    pos = float(metrics.get('position_error_m', 1e-3))
    well = float(metrics.get('well_target_error_m', 1e-3))
    n0 = float(task.get('max_n_total', 0.5))
    tol = float(task.get('position_tolerance_m', 5e-6))
    val = np.log1p(n/n0) + np.log1p((pos/tol)**2) + np.log1p((well/tol)**2)
    # hardware soft penalties
    if distortion is not None:
        rms = float(distortion.get('rms_deviation_V', 0))
        # 0.1V rms is ok, >0.3V penalize
        val += 0.5*np.log1p(max(0, rms-0.1)/0.1)
    if theta is not None:
        val += 1e-4*np.sum(np.asarray(theta)**2)
    # fmin penalty: if provided in metrics
    fmin = float(metrics.get('minimum_frequency_hz', task.get('min_frequency_hz', 400e3)))
    f0 = float(task.get('min_frequency_hz', 400e3))
    if fmin < f0:
        val += 2*np.log1p((f0-fmin)/0.1e6)
    # observable chain for dashboard
    try:
        chain = processing_chain(metrics, task)
        obs_loss = chain['loss']
        # blend observable loss lightly
        val = 0.8*val + 0.2*obs_loss
    except Exception:
        pass
    return float(val)

def evaluate_genome(genome, runner, baseline='sta', exact_nodes=None, hardware=None, stray=None, freq_scale=1.0):
    """Evaluate one genome with cheap ROM (or exact if requested).

    Populates genome.fitness, genome.metrics, genome.status.
    Fix R6: extra genes for 10/12 dims are now extended voltage basis and are NOT ignored.
    They are passed directly to runner.evaluate (which now supports 10/12 fully).
    For true frequency-shaping mode (task variable), frequency_coeffs are handled via
    runner.task, but extra genes now represent voltage shaping, so no prepare needed.
    """
    # Use full theta (8/10/12) directly – pipeline.commands now handles extended basis without re-prepare
    theta = genome.theta if genome.dim in (8, 10, 12) else genome.to_theta8()
    # No need to stash/restore frequency_coeffs – extra genes are voltage shaping, not frequency_mode
    # Keep legacy frequency_mode intact; if task was variable, its coeffs remain as configured
    try:
        # cheap: exact_nodes=None → ROM linear response; expensive: pass exact_nodes
        metrics = runner.evaluate(theta, baseline=baseline, hardware=hardware, stray=stray,
                                  frequency_scale=freq_scale, exact_nodes=exact_nodes, subdivisions=runner.cfg.get('subdivisions',4))
        # hierarchical
        fit = hierarchical_fitness(metrics, runner.task, distortion=metrics.get('distortion'), theta=theta)
        # robust aggregation if island R: evaluate on 2 train scenarios and take worst
        if genome.island in ('R','robust'):
            from ..hardware import Hardware
            train = runner.scenarios(runner.cfg.get('robust_training_samples',2), runner.cfg['uncertainty']['seed']+1)
            vals=[fit]
            for sc in train:
                try:
                    m2=runner.evaluate(theta,'sta', hardware=Hardware(**sc['hardware']), stray=sc['stray_V_m'], frequency_scale=sc['frequency_scale'], exact_nodes=exact_nodes)
                    vals.append(hierarchical_fitness(m2, runner.task, m2.get('distortion'), theta))
                except Exception:
                    vals.append(1e6)
            fit = float(0.35*np.mean(vals)+0.65*np.max(vals))
            metrics['robust_train_vals']=vals
        genome.fitness = float(fit)
        genome.metrics = metrics
        genome.status = 'ok' if metrics.get('accepted', True) else 'quality_fail'
        if metrics.get('quality_failures'):
            genome.status = ','.join(metrics['quality_failures'])
    except Rejected as ex:
        genome.fitness = 1e6
        genome.metrics = {'error':str(ex), 'code':ex.code}
        genome.status = ex.code
    except Exception as ex:
        genome.fitness = 1e6
        genome.metrics = {'error':str(ex), 'code':'numerical_failure'}
        genome.status = 'numerical_failure'
    return genome

def evaluate_exact(genome, runner, baseline='sta', hardware=None):
    """Full exact validation for hall of fame (slow) – uses full theta (8/10/12) so extra genes are validated."""
    theta = genome.theta if genome.dim in (8,10,12) else genome.to_theta8()
    nodes=runner.cfg.get('validation_nodes',321)
    subdiv=runner.cfg.get('validation_subdivisions',4)
    try:
        m=runner.evaluate(theta, baseline, hardware=hardware, exact_nodes=nodes, subdivisions=subdiv)
        m['exact_nodes']=nodes
        return m
    except Rejected as ex:
        return {'error':str(ex), 'code':ex.code, 'exact_nodes':nodes}
