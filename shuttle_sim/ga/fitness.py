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
    # max frequency gate (if configured)
    if 'max_frequency_hz' in task and task['max_frequency_hz'] is not None:
        try:
            fmax = float(metrics.get('maximum_frequency_hz', 0))
            fmax_allowed = float(task['max_frequency_hz'])
            if fmax > fmax_allowed:
                val += 2*np.log1p((fmax - fmax_allowed)/0.1e6)
        except: pass
    # heating rate gate (new 2026-10-30: <100 ph/s)
    try:
        hr = metrics.get('heating_rate_hz')
        if hr is not None:
            allowed = task.get('max_heating_rate_hz', None)
            if allowed is None:
                try:
                    allowed = float(task.get('max_n_total', 0.5))/float(task.get('duration_s', 1))
                    # cap to 100 for new requirement
                    if allowed > 100:
                        allowed = 100
                except: allowed = 100
            if allowed is not None and hr > float(allowed)*1.05:
                # scale: 100 ph/s is target, excess 100 → log1p(1)=0.69*2=1.38, excess 1000 → log1p(10)=2.39*2=4.78
                val += 2.0*np.log1p(max(0.0, hr - float(allowed))/max(float(allowed), 50.0))
                # also store for breakdown
                metrics['_heating_penalty'] = 2.0*np.log1p(max(0.0, hr - float(allowed))/max(float(allowed), 50.0))
    except: pass
    # height gates (soft, pipeline now soft)
    try:
        hv_msg = metrics.get('height_violation_msg')
        if hv_msg:
            # height violation present → penalize
            # estimate violation distance from message or use fixed
            val += 2.0*np.log1p(1.0)  # ~1.38
            metrics['_height_penalty'] = 2.0*np.log1p(1.0)
        else:
            # also check if height bounds violated via explicit well heights if pipeline didn't set msg
            if ('height_min_m' in task and task['height_min_m'] is not None) or ('height_max_m' in task and task['height_max_m'] is not None):
                # if metrics has well heights, check
                # pipeline now stores height_violation but not heights; we can check via n_total? fallback no penalty
                pass
        # also penalize weak_confinement already handled via fmin, but ensure
        if metrics.get('weak_confinement'):
            # weak already penalized via fmin< f0, but add extra if needed
            pass
    except: pass
    if 'height_min_m' in task and task['height_min_m'] is not None:
        try:
            # estimate height from metrics if available
            h = metrics.get('well_height_m', metrics.get('height_m', metrics.get('minimum_height_m', None)))
            if h is None:
                # fallback: use pipeline's stored min height if present
                h = metrics.get('height_min_violation', None)
            # if we have explicit z values we penalize
            # pipeline already rejects, but add soft for near-bound
        except: pass
    # initial / final frequency targets (soft penalty if deviates)
    if 'frequency_initial_hz' in task and task['frequency_initial_hz'] is not None:
        try:
            f_start = float(metrics.get('frequency_start_hz', fmin))
            target = float(task['frequency_initial_hz'])
            err = abs(f_start - target)
            if err > 0.02e6:
                val += 1.0*np.log1p(err/0.05e6)
        except: pass
    if 'frequency_final_hz' in task and task['frequency_final_hz'] is not None:
        try:
            f_end = float(metrics.get('frequency_end_hz', fmin))
            target = float(task['frequency_final_hz'])
            err = abs(f_end - target)
            if err > 0.02e6:
                val += 1.0*np.log1p(err/0.05e6)
        except: pass
    return float(val)

_W_prepare_cache = {}

def _prepare_for_W(runner, freq_coeffs):
    """Prepare runner for W island frequency variable mode with caching.
    freq_coeffs: array length 2 in [-0.15,0.15], maps to ω(s)=ω0*(1+a·env+b·env·(2s-1)), clip ±18% in prepare.
    Returns True if prepared, raises Rejected on failure.
    """
    key = tuple(np.round(np.asarray(freq_coeffs, float), 4).tolist())
    if key in _W_prepare_cache:
        # restore from cache by copying cached prepared arrays back onto runner
        cached = _W_prepare_cache[key]
        for k, v in cached.items():
            if isinstance(v, np.ndarray):
                setattr(runner, k, v.copy() if hasattr(v, 'copy') else v)
            else:
                setattr(runner, k, v.copy() if isinstance(v, (list, dict)) else v)
        runner.frequency_mode = 'variable'
        runner.frequency_coeffs = list(key)
        runner.task['frequency_mode'] = 'variable'
        runner.task['frequency_coeffs'] = list(key)
        return True
    # not cached: do fresh prepare and cache result
    runner.task['frequency_mode'] = 'variable'
    runner.task['frequency_coeffs'] = list(map(float, freq_coeffs))
    runner.prepare()
    # cache prepared state (only the arrays that prepare touches)
    to_cache = {}
    for attr in ['sroute','frequency_mode','frequency_coeffs','route_V','voltage_spline','route_r','route_H','route_G','route_BH','Hspline','Gspline','BHspline','correction_modes','support_diagnostics']:
        if hasattr(runner, attr):
            val = getattr(runner, attr)
            # CubicSpline not easily copyable via copy, store reference and rebuild? For now store the arrays, splines will be rebuilt from arrays on restore via prepare cache hit path above we just copy spline objects
            # Spline objects are fine to share (they are read-only)
            to_cache[attr] = val
    _W_prepare_cache[key] = to_cache
    # limit cache size
    if len(_W_prepare_cache) > 64:
        # drop oldest
        oldest = next(iter(_W_prepare_cache))
        del _W_prepare_cache[oldest]
    return True

def evaluate_genome(genome, runner, baseline='sta', exact_nodes=None, hardware=None, stray=None, freq_scale=1.0):
    """Evaluate one genome with cheap ROM (or exact if requested).

    Populates genome.fitness, genome.metrics, genome.status.
    Fix R6: extra genes for 10/12 dims are now extended voltage basis and are NOT ignored.
    They are passed directly to runner.evaluate (which now supports 10/12 fully).
    Split islands: V = pure voltage 10 генов (4 базиса×2 ΔV), W = pure frequency
    10 генов (3 базиса×2 ΔV + 2 freq a·env+b·env·(2s-1) via FrequencyProfile variable).
    This satisfies request: один остров крутит напряжение, другой частоту.
    """
    # Decide theta handling per island
    island = genome.island
    is_W = island in ('W', 'frequency', 'freq')
    is_V = island in ('V', 'variable', 'voltage')
    # For W, split theta: first 8 are voltage/warp, last 2 are frequency
    if is_W and genome.dim >= 10:
        theta_voltage = genome.theta[:8].copy()
        freq_coeffs = genome.theta[8:10].copy()
        # voltage part stays 8 dims (2 warp +6 volt), frequency via variable profile
        theta = theta_voltage
        use_W_prepare = True
    else:
        # V and others: full theta as voltage (including extended harmonic for V)
        theta = genome.theta if genome.dim in (8, 10, 12) else genome.to_theta8()
        freq_coeffs = None
        use_W_prepare = False
    # If W and caller already supplied freq_scale !=1 (robust scenario), we respect caller's scale and not do variable prepare
    effective_freq_scale = freq_scale
    # For W with variable frequency we will use prepare, not frequency_scale uniform
    do_prepare = use_W_prepare and freq_scale == 1.0
    # Stash runner state for W prepare so other islands not polluted
    stashed = None
    if do_prepare:
        # stash global runner task and prepared arrays
        stashed = dict(
            task_mode=runner.task.get('frequency_mode'),
            task_coeffs=runner.task.get('frequency_coeffs'),
            freq_mode=getattr(runner, 'frequency_mode', 'fixed'),
            freq_coeffs=getattr(runner, 'frequency_coeffs', [0.0,0.0]),
            sroute=getattr(runner, 'sroute', None),
            route_V=getattr(runner, 'route_V', None),
            route_r=getattr(runner, 'route_r', None),
            route_H=getattr(runner, 'route_H', None),
            route_G=getattr(runner, 'route_G', None),
            route_BH=getattr(runner, 'route_BH', None),
            correction_modes=getattr(runner, 'correction_modes', None),
            support_diagnostics=getattr(runner, 'support_diagnostics', None),
            # splines are derived from above, will be restored via cache or prepare
            voltage_spline=getattr(runner, 'voltage_spline', None),
            Hspline=getattr(runner, 'Hspline', None),
            Gspline=getattr(runner, 'Gspline', None),
            BHspline=getattr(runner, 'BHspline', None),
        )
        try:
            _prepare_for_W(runner, freq_coeffs)
        except Exception as ex:
            # restore before re-raising as Rejected
            for k, v in stashed.items():
                if k.startswith('task_'):
                    continue
                if v is not None:
                    setattr(runner, k if not k.startswith('task_') else k, v)
            # also restore task
            if stashed['task_mode'] is not None:
                runner.task['frequency_mode'] = stashed['task_mode']
            if stashed['task_coeffs'] is not None:
                runner.task['frequency_coeffs'] = stashed['task_coeffs']
            raise
    # Detect segmented K/M/C islands: they evaluate stitched Kx(D/K) path
    is_stitched = False
    stitched_segments = None
    is_periodic = False
    periodic_repeats = None
    if isinstance(island, str) and (island.startswith('K') or island.startswith('M') or island.startswith('C') or island in ('STITCH','KRAT')):
        is_stitched = True
        import re
        m = re.match(r'^[KMC](\d+)$', island)
        if m:
            try:
                stitched_segments = int(m.group(1))
            except:
                stitched_segments = 'auto'
        elif island in ('K','M','C','STITCH','KRAT'):
            stitched_segments = 'auto'
        else:
            stitched_segments = 'auto'
        if isinstance(stitched_segments, int) and stitched_segments > 10:
            stitched_segments = 10
        if isinstance(stitched_segments, int) and stitched_segments < 2:
            stitched_segments = 2
    elif isinstance(island, str) and (island.startswith('P') or island in ('PERIODIC',)):
        is_periodic = True
        import re
        m = re.match(r'^P(\d+)$', island)
        if m:
            try:
                periodic_repeats = int(m.group(1))
            except:
                periodic_repeats = 'auto'
        elif island in ('P','PERIODIC'):
            periodic_repeats = 'auto'
        else:
            periodic_repeats = 'auto'
        if isinstance(periodic_repeats, int) and periodic_repeats > 10:
            periodic_repeats = 10
        if isinstance(periodic_repeats, int) and periodic_repeats < 2:
            periodic_repeats = 2

    try:
        if is_stitched:
            if do_prepare:
                metrics = runner.evaluate(theta, baseline=baseline, hardware=hardware, stray=stray,
                                          frequency_scale=effective_freq_scale, exact_nodes=exact_nodes, subdivisions=min(1, runner.cfg.get('subdivisions',4)), path_samples_per_interval=4)
            else:
                metrics = runner.evaluate_segmented(theta, baseline=baseline, segments=stitched_segments, hardware=hardware, stray=stray,
                                                   frequency_scale=effective_freq_scale, exact_nodes=exact_nodes, subdivisions=min(1, runner.cfg.get('subdivisions',4)), path_samples_per_interval=4)
        elif is_periodic:
            if do_prepare:
                metrics = runner.evaluate(theta, baseline=baseline, hardware=hardware, stray=stray,
                                          frequency_scale=effective_freq_scale, exact_nodes=exact_nodes, subdivisions=min(1, runner.cfg.get('subdivisions',4)), path_samples_per_interval=4)
            else:
                metrics = runner.evaluate_krat_periodic(theta, baseline=baseline, repeats=periodic_repeats, hardware=hardware, stray=stray,
                                                       frequency_scale=effective_freq_scale, exact_nodes=exact_nodes, subdivisions=min(1, runner.cfg.get('subdivisions',4)), path_samples_per_interval=4)
        else:
                metrics = runner.evaluate(theta, baseline=baseline, hardware=hardware, stray=stray,
                                          frequency_scale=effective_freq_scale, exact_nodes=exact_nodes, subdivisions=min(1, runner.cfg.get('subdivisions',4)), path_samples_per_interval=4)
        # For W, theta is voltage subset (8) but penalty should consider full genome (10) including freq coeffs
        theta_for_penalty = genome.theta
        bounds_for_penalty = genome.bounds
        fit = scalar_fitness(metrics, runner.task, distortion=metrics.get('distortion'),
                             theta=theta_for_penalty, control_bounds=bounds_for_penalty,
                             control_regularization_weight=runner.cfg.get('control_regularization_weight',1e-4),
                             loss_weights=runner.loss_weights)
        # robust aggregation if island R: evaluate on 2 train scenarios and take worst
        if genome.island in ('R','robust'):
            from ..hardware import Hardware
            train = runner.scenarios(runner.cfg.get('robust_training_samples',2), runner.cfg['uncertainty']['seed']+1)
            vals=[fit]
            for sc in train:
                try:
                    # For R island keep scenario's frequency_scale; for V-derived frequency we already applied above only to primary fit
                    m2=runner.evaluate(theta,'sta', hardware=Hardware(**sc['hardware']), stray=sc['stray_V_m'], frequency_scale=sc['frequency_scale'], exact_nodes=exact_nodes)
                    vals.append(scalar_fitness(
                        m2, runner.task, m2.get('distortion'), theta, genome.bounds,
                        runner.cfg.get('control_regularization_weight',1e-4),
                        loss_weights=runner.loss_weights))
                except Exception as ex_rob:
                    # Было 1e6 → давало 8.83e5 (0.35*mean+0.65*max) и полностью убивало R-остров:
                    # даже если основной fit ~0.5, два фейла 1e6 давали (0.5+2e6)/3≈666k среднее, max 1e6 → 883k.
                    # Делаем умеренный штраф, чтобы робастность учитывалась но не душила остров и позволяла миграцию.
                    try:
                        code = getattr(ex_rob, 'code', str(ex_rob)) if hasattr(ex_rob, 'code') else str(ex_rob)
                    except: code='robust_fail'
                    # мягкий штраф: primary + 6..8 (эквивалент ~5-10 n) вместо 1e6
                    vals.append(float(fit) + 7.0)
            # cap robust vals to avoid одного выброса 1e6 который снова даст 8e5 — клипуем к fit+10
            try:
                vals = [min(float(v), float(fit)+10.0) if float(v) > float(fit)+10 else float(v) for v in vals]
            except: pass
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
    finally:
        if stashed is not None and do_prepare:
            # restore runner to original fixed mode regardless of success/failure
            # Use prepare to rebuild fixed splines reliably (avoid stale spline objects)
            try:
                runner.task['frequency_mode'] = stashed['task_mode'] if stashed['task_mode'] is not None else 'fixed'
                runner.task['frequency_coeffs'] = stashed['task_coeffs'] if stashed['task_coeffs'] is not None else [0.0,0.0]
                # If original was fixed, do a fresh prepare to rebuild splines from fixed route
                # If we have cached fixed state, use it; otherwise prepare
                if stashed['task_mode'] in (None, 'fixed') or stashed['freq_mode'] == 'fixed':
                    # Check if we have a cached fixed prepare (key (0,0))
                    fixed_key = (0.0, 0.0)
                    if fixed_key in _W_prepare_cache:
                        cached = _W_prepare_cache[fixed_key]
                        for k, v in cached.items():
                            setattr(runner, k, v)
                        runner.frequency_mode = 'fixed'
                        runner.frequency_coeffs = [0.0, 0.0]
                        runner.task['frequency_mode'] = 'fixed'
                        runner.task['frequency_coeffs'] = [0.0, 0.0]
                    else:
                        runner.prepare()
                        # cache fixed for future
                        to_cache = {}
                        for attr in ['sroute','frequency_mode','frequency_coeffs','route_V','voltage_spline','route_r','route_H','route_G','route_BH','Hspline','Gspline','BHspline','correction_modes','support_diagnostics']:
                            if hasattr(runner, attr):
                                to_cache[attr] = getattr(runner, attr)
                        _W_prepare_cache[fixed_key] = to_cache
                else:
                    for k in ['frequency_mode','frequency_coeffs','sroute','route_V','route_r','route_H','route_G','route_BH','correction_modes','support_diagnostics','voltage_spline','Hspline','Gspline','BHspline']:
                        if stashed.get(k) is not None:
                            setattr(runner, k, stashed[k])
            except Exception:
                try:
                    runner.task['frequency_mode'] = 'fixed'
                    runner.task['frequency_coeffs'] = [0.0, 0.0]
                    runner.prepare()
                except Exception:
                    pass
    return genome

def evaluate_exact(genome, runner, baseline='sta', hardware=None):
    """Full exact validation for hall of fame (slow) – uses full theta (8/10/12) so extra genes are validated.
    For W island, frequency coeffs are handled via variable prepare as in evaluate_genome.
    """
    island = genome.island
    is_W = island in ('W', 'frequency', 'freq')
    if is_W and genome.dim >= 10:
        theta = genome.theta[:8].copy()
        freq_coeffs = genome.theta[8:10].copy()
        stashed = dict(
            task_mode=runner.task.get('frequency_mode'),
            task_coeffs=runner.task.get('frequency_coeffs'),
            freq_mode=getattr(runner, 'frequency_mode', 'fixed'),
            freq_coeffs=getattr(runner, 'frequency_coeffs', [0.0,0.0]),
            sroute=getattr(runner, 'sroute', None),
            route_V=getattr(runner, 'route_V', None),
            route_r=getattr(runner, 'route_r', None),
            route_H=getattr(runner, 'route_H', None),
            route_G=getattr(runner, 'route_G', None),
            route_BH=getattr(runner, 'route_BH', None),
            correction_modes=getattr(runner, 'correction_modes', None),
            support_diagnostics=getattr(runner, 'support_diagnostics', None),
            voltage_spline=getattr(runner, 'voltage_spline', None),
            Hspline=getattr(runner, 'Hspline', None),
            Gspline=getattr(runner, 'Gspline', None),
            BHspline=getattr(runner, 'BHspline', None),
        )
        try:
            _prepare_for_W(runner, freq_coeffs)
            nodes=runner.cfg.get('validation_nodes',321)
            subdiv=runner.cfg.get('validation_subdivisions',4)
            m=runner.evaluate(theta, baseline, hardware=hardware, exact_nodes=nodes, subdivisions=subdiv)
            m['exact_nodes']=nodes
            return m
        except Rejected as ex:
            nodes=runner.cfg.get('validation_nodes',321)
            return {'error':str(ex), 'code':ex.code, 'exact_nodes':nodes}
        finally:
            try:
                runner.task['frequency_mode'] = stashed['task_mode'] if stashed['task_mode'] is not None else 'fixed'
                runner.task['frequency_coeffs'] = stashed['task_coeffs'] if stashed['task_coeffs'] is not None else [0.0,0.0]
                for k in ['frequency_mode','frequency_coeffs','sroute','route_V','route_r','route_H','route_G','route_BH','correction_modes','support_diagnostics','voltage_spline','Hspline','Gspline','BHspline']:
                    if stashed.get(k) is not None:
                        setattr(runner, k, stashed[k])
            except Exception:
                pass
    theta = genome.theta if genome.dim in (8,10,12) else genome.to_theta8()
    nodes=runner.cfg.get('validation_nodes',321)
    subdiv=runner.cfg.get('validation_subdivisions',4)
    try:
        m=runner.evaluate(theta, baseline, hardware=hardware, exact_nodes=nodes, subdivisions=subdiv)
        m['exact_nodes']=nodes
        return m
    except Rejected as ex:
        return {'error':str(ex), 'code':ex.code, 'exact_nodes':nodes}
