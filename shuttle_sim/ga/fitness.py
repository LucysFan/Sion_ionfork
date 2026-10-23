"""Единая скалярная оценка кандидата для островного поиска.

Быстрая оценка использует линейную модель отклика; точная проверка
оставляется для итогового зала кандидатов. Компоненты сохраняются отдельно.
"""
import numpy as np
from ..field import Rejected, finite
from ..observables import scalar_loss_from_observables, dimensionless_control_penalty

def scalar_fitness(metrics, task, distortion=None, theta=None, control_bounds=None,
                   control_regularization_weight=1e-4, loss_weights=None):
    """Скалярная оценка из общей физической потери и дополнительных аппаратных цен."""
    # The shared default scores final-state quality; path lag is diagnostic
    # unless its weight is explicitly enabled in the task configuration.
    val = scalar_loss_from_observables(metrics, task, weights=loss_weights)
    if distortion is not None:
        rms = float(distortion.get('rms_deviation_V', 0))
        # 0.1 V is the declared soft scale; the ratio is dimensionless.
        val += 0.5*np.log1p(max(0, rms-0.1)/0.1)
    if theta is not None and control_bounds is not None:
        val += dimensionless_control_penalty(theta, control_bounds,
                                             control_regularization_weight)
    # This is a hard confinement gate in Runner; retain a soft term only for
    # callers that supply an already-computed below-floor metric.
    fmin = float(metrics.get('minimum_frequency_hz', task.get('min_frequency_hz', 400e3)))
    f0 = float(task.get('min_frequency_hz', 400e3))
    if fmin < f0:
        val += 2*np.log1p((f0-fmin)/0.1e6)
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
        fit = scalar_fitness(metrics, runner.task, distortion=metrics.get('distortion'),
                             theta=theta, control_bounds=genome.bounds,
                             control_regularization_weight=runner.cfg.get('control_regularization_weight',1e-4),
                             loss_weights=runner.loss_weights)
        # robust aggregation if island R: evaluate on 2 train scenarios and take worst
        if genome.island in ('R','robust'):
            from ..hardware import Hardware
            train = runner.scenarios(runner.cfg.get('robust_training_samples',2), runner.cfg['uncertainty']['seed']+1)
            vals=[fit]
            for sc in train:
                try:
                    m2=runner.evaluate(theta,'sta', hardware=Hardware(**sc['hardware']), stray=sc['stray_V_m'], frequency_scale=sc['frequency_scale'], exact_nodes=exact_nodes)
                    vals.append(scalar_fitness(
                        m2, runner.task, m2.get('distortion'), theta, genome.bounds,
                        runner.cfg.get('control_regularization_weight',1e-4),
                        loss_weights=runner.loss_weights))
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
