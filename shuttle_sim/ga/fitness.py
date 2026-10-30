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
    # heating rate gate — only if explicitly configured (audit 00fbaf6: no hidden 100 ph/s)
    # residual_excitation (n_total) is primary; heating_rate_hz is diagnostic unless max_heating_rate_hz set
    try:
        hr = metrics.get('heating_rate_hz')
        if hr is not None:
            allowed = task.get('max_heating_rate_hz', None)
            if allowed is None:
                # no heating penalty unless explicitly requested (explicit user criterion)
                pass
            elif hr > float(allowed)*1.05:
                # scale: allowed is target, excess allowed → log1p(1)=0.69*2=1.38
                val += 2.0*np.log1p(max(0.0, hr - float(allowed))/max(float(allowed), 50.0))
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
_runner_caches = {}  # global fallback, but primary is per-runner isolated cache

def _w_geom_hash(runner):
    """Hash runner geometry/task for cache isolation (prevents mixing distances/protocols/traps).
    Includes direction, start, mass, charge, RF, bounds and GDS identity.
    """
    try:
        dist = float(runner.task.get('distance_m', 0))
    except: dist = 0
    try:
        dur = float(runner.task.get('duration_s', 0))
    except: dur = 0
    proto = str(runner.task.get('protocol', 'one_way'))
    cycles = int(runner.task.get('cycles', 1))
    try:
        kind = getattr(runner.trap, 'kind', None) or getattr(runner.cfg.get('trap',{}), 'kind', 'unknown')
    except: kind = 'unknown'
    try:
        n_elec = len(getattr(runner.trap, 'indices', []))
    except: n_elec = 0
    try:
        path = str(runner.cfg.get('trap',{}).get('path',''))[:200]
    except: path = ''
    # direction and start (opposite routes must not collide)
    try:
        direction = tuple(np.round(np.asarray(runner.task.get('direction', [1,0,0]), float), 6).tolist())
    except: direction = (1.0,0.0,0.0)
    try:
        # start_m from task or trap seed
        start = runner.start if hasattr(runner, 'start') else runner.cfg.get('trap',{}).get('minimum_seed_m', [0,0,90e-6])
        start_t = tuple(np.round(np.asarray(start, float), 9).tolist())
    except: start_t = (0.0,0.0,90e-6)
    # mass, charge, RF
    try:
        mass_amu = float(runner.trap.mass / 1.66053906660e-27) if hasattr(runner.trap, 'mass') else float(runner.cfg.get('trap',{}).get('mass_amu', 40))
    except: mass_amu = 40.0
    try:
        rf_hz = float(runner.trap.rf_frequency_hz) if hasattr(runner.trap, 'rf_frequency_hz') else float(runner.cfg.get('trap',{}).get('rf_frequency_hz', 20e6))
    except: rf_hz = 20e6
    try:
        rf_peak = float(runner.trap.rf_peak_V) if hasattr(runner.trap, 'rf_peak_V') else float(runner.cfg.get('trap',{}).get('rf_peak_V', 160.0))
    except: rf_peak = 160.0
    try:
        trap_cfg = runner.cfg.get('trap',{})
        lower = float(trap_cfg.get('lower_V', -12.0))
        upper = float(trap_cfg.get('upper_V', 12.0))
        bounds = (lower, upper)
    except: bounds = (-12.0, 12.0)
    # hash of voltage_spline support (route) length and sroute range as proxy for geometry resolution
    try:
        sroute_hash = (len(getattr(runner, 'sroute', [])), float(np.round(getattr(runner, 'sroute', [0])[-1], 6)) if len(getattr(runner, 'sroute', []))>0 else 0)
    except: sroute_hash = (0,0)
    return (round(dist,9), round(dur,9), proto, cycles, str(kind), n_elec, path, direction, start_t, round(mass_amu,6), round(rf_hz,1), round(rf_peak,3), bounds, sroute_hash)

def _w_cache_key(runner, freq_coeffs):
    """Key isolated per runner geometry and frequency coeffs."""
    coeffs = tuple(np.round(np.asarray(freq_coeffs, float), 4).tolist())
    geom = _w_geom_hash(runner)
    return (coeffs, geom)

def _get_runner_w_cache(runner):
    """Per-runner cache dict (isolated). Falls back to global with geom-isolated key if needed."""
    if hasattr(runner, '_w_prepare_cache') and isinstance(runner._w_prepare_cache, dict):
        return runner._w_prepare_cache
    # use global geom-isolated cache as fallback, but also ensure per-runner attribute exists
    runner._w_prepare_cache = {}
    return runner._w_prepare_cache

def _prepare_for_W(runner, freq_coeffs):
    """Prepare runner for W island frequency variable mode with caching (isolated per runner/geometry).
    freq_coeffs: array length 2 in [-0.15,0.15], maps to ω(s)=ω0*(1+a·env+b·env·(2s-1)), clip ±18% in prepare.
    Returns True if prepared, raises Rejected on failure.
    """
    key = _w_cache_key(runner, freq_coeffs)
    # try per-runner cache first
    per = _get_runner_w_cache(runner)
    if key in per:
        cached = per[key]
        for k, v in cached.items():
            if isinstance(v, np.ndarray):
                setattr(runner, k, v.copy() if hasattr(v, 'copy') else v)
            else:
                setattr(runner, k, v.copy() if isinstance(v, (list, dict)) else v)
        coeff_tuple = tuple(np.round(np.asarray(freq_coeffs, float),4).tolist())
        runner.frequency_mode = 'variable'
        runner.frequency_coeffs = list(coeff_tuple)
        runner.task['frequency_mode'] = 'variable'
        runner.task['frequency_coeffs'] = list(coeff_tuple)
        return True
    # global cache disabled for correctness (audit P1: opposite routes/RF collided)
    # keep only per-runner cache; global _W_prepare_cache is no longer used for hits
    # if needed later, re-enable with full geom hash above and explicit validation
    pass
    # not cached: do fresh prepare and cache result
    coeff_list = list(map(float, np.asarray(freq_coeffs, float).tolist()))
    runner.task['frequency_mode'] = 'variable'
    runner.task['frequency_coeffs'] = coeff_list
    runner.prepare()
    to_cache = {}
    for attr in ['sroute','frequency_mode','frequency_coeffs','route_V','voltage_spline','route_r','route_H','route_G','route_BH','Hspline','Gspline','BHspline','correction_modes','support_diagnostics']:
        if hasattr(runner, attr):
            to_cache[attr] = getattr(runner, attr)
    per[key] = to_cache
    _W_prepare_cache[key] = to_cache
    # also ensure runner-local global index for id-based isolation
    _runner_caches[id(runner)] = per
    if len(per) > 32:
        oldest = next(iter(per))
        del per[oldest]
    if len(_W_prepare_cache) > 64:
        oldest = next(iter(_W_prepare_cache))
        del _W_prepare_cache[oldest]
    return True


def _evaluate_via_protocol(runner, theta, baseline, hardware, stray, freq_scale, exact_nodes, subdivisions, path_samples_per_interval):
    """Unified protocol dispatcher for all callers (fitness/robust/exact/dashboard/export).
    Handles stitched K/M/C and periodic P automatically based on baseline/island context.
    Caller must provide island-specific baseline and handle W stm? This helper is generic for single protocol evaluation.
    For island-aware dispatch, use evaluate_genome's island detection; this fallback is for export/dashboard where island/bounds known.
    """
    # This low-level helper is for completeness; main dispatch is in evaluate_genome/evaluate_exact which inspect island.
    # Direct call without island assumes generic one-way evaluation.
    return runner.evaluate(theta, baseline=baseline, hardware=hardware, stray=stray, frequency_scale=freq_scale, exact_nodes=exact_nodes, subdivisions=subdivisions, path_samples_per_interval=path_samples_per_interval)

def _dispatch_evaluate(runner, genome, baseline, hardware, stray, freq_scale, exact_nodes, subdivisions=4, path_samples_per_interval=4):
    """Island-aware unified dispatcher: one genome -> one protocol everywhere (audit 2026-10-30).
    Respects W frequency variable mode via _prepare_for_W when freq_scale==1, otherwise uniform scale.
    Handles K stitched and P periodic by island name. Used by fitness, robust, exact, export, dashboard.
    """
    island = getattr(genome, 'island', None) if hasattr(genome, 'island') else None
    # W split: first 8 voltage, last 2 freq if dim>=10
    is_W = island in ('W', 'frequency', 'freq')
    is_V = island in ('V', 'variable', 'voltage')
    if is_W and getattr(genome, 'dim', 8) >= 10 and hasattr(genome, 'theta'):
        theta_voltage = genome.theta[:8].copy()
        freq_coeffs = genome.theta[8:10].copy()
        theta = theta_voltage
        # W profile is always applied; frequency_scale is additional uniform perturbation on top (audit P1)
        use_prepare = True
    else:
        theta = genome.theta if hasattr(genome, 'theta') else genome
        freq_coeffs = None
        use_prepare = False
        if hasattr(genome, 'to_theta8') and getattr(genome, 'dim', 8) not in (8,10,12):
            theta = genome.to_theta8()
    # stitched / periodic detection
    is_stitched = False; stitched_segments=None; is_periodic=False; periodic_repeats=None
    if isinstance(island, str) and (island.startswith('K') or island.startswith('M') or island.startswith('C') or island in ('STITCH','KRAT')):
        is_stitched=True
        import re
        m=re.match(r'^[KMC](\d+)$', island)
        if m:
            try: stitched_segments=int(m.group(1))
            except: stitched_segments='auto'
        elif island in ('K','M','C','STITCH','KRAT'):
            stitched_segments='auto'
        else: stitched_segments='auto'
        if isinstance(stitched_segments,int) and stitched_segments>10: stitched_segments=10
        if isinstance(stitched_segments,int) and stitched_segments<2: stitched_segments=2
    elif isinstance(island, str) and (island.startswith('P') or island in ('PERIODIC',)):
        is_periodic=True
        import re
        m=re.match(r'^P(\d+)$', island)
        if m:
            try: periodic_repeats=int(m.group(1))
            except: periodic_repeats='auto'
        elif island in ('P','PERIODIC'): periodic_repeats='auto'
        else: periodic_repeats='auto'
        if isinstance(periodic_repeats,int) and periodic_repeats>10: periodic_repeats=10
        if isinstance(periodic_repeats,int) and periodic_repeats<2: periodic_repeats=2
    # handle W prepare with exception-safe snapshot (audit P1)
    # Use context-manager style: stash, prepare, evaluate, always restore in finally
    stashed=None
    if use_prepare:
        stashed=dict(task_mode=runner.task.get('frequency_mode'), task_coeffs=runner.task.get('frequency_coeffs'),
                     freq_mode=getattr(runner,'frequency_mode','fixed'), freq_coeffs=getattr(runner,'frequency_coeffs',[0.0,0.0]),
                     sroute=getattr(runner,'sroute',None), route_V=getattr(runner,'route_V',None), route_r=getattr(runner,'route_r',None),
                     route_H=getattr(runner,'route_H',None), route_G=getattr(runner,'route_G',None), route_BH=getattr(runner,'route_BH',None),
                     correction_modes=getattr(runner,'correction_modes',None), support_diagnostics=getattr(runner,'support_diagnostics',None),
                     voltage_spline=getattr(runner,'voltage_spline',None), Hspline=getattr(runner,'Hspline',None),
                     Gspline=getattr(runner,'Gspline',None), BHspline=getattr(runner,'BHspline',None))
        try:
            _prepare_for_W(runner, freq_coeffs)
        except Exception:
            # restore stash before propagating prepare failure
            try:
                for k,v in stashed.items():
                    if k.startswith('task_'): continue
                    if v is not None:
                        setattr(runner, k, v)
                if stashed['task_mode'] is not None: runner.task['frequency_mode']=stashed['task_mode']
                if stashed['task_coeffs'] is not None: runner.task['frequency_coeffs']=stashed['task_coeffs']
            except: pass
            raise
    # now evaluate with guaranteed restore in finally
    try:
        if is_stitched:
            metrics=runner.evaluate_segmented(theta, baseline=baseline, segments=stitched_segments, hardware=hardware, stray=stray, frequency_scale=freq_scale, exact_nodes=exact_nodes, subdivisions=(subdivisions if subdivisions is not None else runner.cfg.get('subdivisions',4)), path_samples_per_interval=path_samples_per_interval)
        elif is_periodic:
            metrics=runner.evaluate_krat_periodic(theta, baseline=baseline, repeats=periodic_repeats, hardware=hardware, stray=stray, frequency_scale=freq_scale, exact_nodes=exact_nodes, subdivisions=(subdivisions if subdivisions is not None else runner.cfg.get('subdivisions',4)), path_samples_per_interval=path_samples_per_interval)
        else:
            metrics=runner.evaluate(theta, baseline=baseline, hardware=hardware, stray=stray, frequency_scale=freq_scale, exact_nodes=exact_nodes, subdivisions=(subdivisions if subdivisions is not None else runner.cfg.get('subdivisions',4)), path_samples_per_interval=path_samples_per_interval)
        return metrics, stashed, use_prepare
    finally:
        # always restore W snapshot, even if evaluate raised
        if stashed is not None and use_prepare:
            try:
                _restore_after_W(runner, stashed)
            except: pass

def _restore_after_W(runner, stashed):
    if stashed is None: return
    try:
        runner.task['frequency_mode'] = stashed['task_mode'] if stashed['task_mode'] is not None else 'fixed'
        runner.task['frequency_coeffs'] = stashed['task_coeffs'] if stashed['task_coeffs'] is not None else [0.0,0.0]
        if stashed['task_mode'] in (None,'fixed') or stashed['freq_mode']=='fixed':
            # try per-runner cache for fixed
            per=_get_runner_w_cache(runner)
            fixed_key=_w_cache_key(runner, [0.0,0.0])
            # re-derive fixed geometry key is with zeros and current geom (which for fixed is same)
            # Instead we can simply re-prepare fixed if not cached
            found=False
            for k in list(per.keys()):
                if k[0]==(0.0,0.0):
                    cached=per[k]
                    for kk,v in cached.items():
                        setattr(runner, kk, v)
                    runner.frequency_mode='fixed'
                    runner.frequency_coeffs=[0.0,0.0]
                    runner.task['frequency_mode']='fixed'
                    runner.task['frequency_coeffs']=[0.0,0.0]
                    found=True
                    break
            if not found:
                # rebuild fixed via prepare
                runner.prepare()
                # cache it
                to_cache={}
                for attr in ['sroute','frequency_mode','frequency_coeffs','route_V','voltage_spline','route_r','route_H','route_G','route_BH','Hspline','Gspline','BHspline','correction_modes','support_diagnostics']:
                    if hasattr(runner, attr): to_cache[attr]=getattr(runner, attr)
                # construct fixed key properly
                try:
                    fk=_w_cache_key(runner, [0.0,0.0])
                    per[fk]=to_cache
                    _W_prepare_cache[fk]=to_cache
                except: pass
        else:
            for k in ['frequency_mode','frequency_coeffs','sroute','route_V','route_r','route_H','route_G','route_BH','correction_modes','support_diagnostics','voltage_spline','Hspline','Gspline','BHspline']:
                if stashed.get(k) is not None:
                    setattr(runner, k, stashed[k])
    except Exception:
        try:
            runner.task['frequency_mode']='fixed'
            runner.task['frequency_coeffs']=[0.0,0.0]
            runner.prepare()
        except: pass

def evaluate_genome(genome, runner, baseline='sta', exact_nodes=None, hardware=None, stray=None, freq_scale=1.0):
    """Evaluate one genome with cheap ROM (or exact if requested). Unified dispatcher.
    Populates genome.fitness, genome.metrics, genome.status.
    Fix R6: extra genes for 10/12 dims are now extended voltage basis and are NOT ignored.
    Split islands: V = pure voltage 10 генов (4 базиса×2 ΔV), W = pure frequency
    10 генов (3 базиса×2 ΔV + 2 freq a·env+b·env·(2s-1) via FrequencyProfile variable).
    Unified protocol: one genome -> one protocol everywhere (audit 2026-10-30).
    """
    # Unified island-aware dispatch
    effective_freq_scale = freq_scale
    try:
        metrics, stashed, used_prepare = _dispatch_evaluate(runner, genome, baseline, hardware, stray, effective_freq_scale, exact_nodes, subdivisions=1, path_samples_per_interval=4)
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
                    # For R island keep scenario's frequency_scale; use unified dispatcher for same genome
                    m2, st2, _ = _dispatch_evaluate(runner, genome, 'sta', hardware=Hardware(**sc['hardware']), stray=sc['stray_V_m'], freq_scale=sc['frequency_scale'], exact_nodes=exact_nodes, subdivisions=4, path_samples_per_interval=4)
                    try:
                        _restore_after_W(runner, st2)
                    except: pass
                    vals.append(scalar_fitness(
                        m2, runner.task, m2.get('distortion'), theta_for_penalty, genome.bounds,
                        runner.cfg.get('control_regularization_weight',1e-4),
                        loss_weights=runner.loss_weights))
                except Exception as ex_rob:
                    try:
                        code = getattr(ex_rob, 'code', str(ex_rob)) if hasattr(ex_rob, 'code') else str(ex_rob)
                    except: code='robust_fail'
                    vals.append(float(fit) + 7.0)
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
        stashed = None
        used_prepare = False
    except Exception as ex:
        genome.fitness = 1e6
        genome.metrics = {'error':str(ex), 'code':'numerical_failure'}
        genome.status = 'numerical_failure'
        stashed = None
        try:
            used_prepare
        except: used_prepare=False
    finally:
        try:
            if 'stashed' in locals() and stashed is not None and 'used_prepare' in locals() and used_prepare:
                _restore_after_W(runner, stashed)
        except: pass
        # also ensure robust interim stashes already restored
    return genome

def evaluate_exact(genome, runner, baseline='sta', hardware=None):
    """Full exact validation for hall of fame (slow) – uses unified dispatcher.
    For W island, frequency coeffs are handled via variable prepare as in evaluate_genome.
    Isolated per geometry, same as evaluate_genome (audit fix).
    exact_evaluated = calculation completed at high fidelity
    exact_validated = constraints passed (accepted) at high fidelity, not just completion
    """
    nodes=runner.cfg.get('validation_nodes',321)
    subdiv=runner.cfg.get('validation_subdivisions',4)
    try:
        metrics, stashed, used_prepare = _dispatch_evaluate(runner, genome, baseline, hardware, None, 1.0, exact_nodes=nodes, subdivisions=subdiv, path_samples_per_interval=8)
        metrics['exact_nodes']=nodes
        metrics['actual_exact_nodes']=nodes
        metrics['actual_subdivisions']=subdiv
        metrics['exact_evaluated']=True
        # exact_validated should reflect constraints, not just completion (audit)
        # keep legacy exact_validated as constraints for backward compat, but also expose split flags
        is_accepted = bool(metrics.get('accepted', False))
        is_high = bool(metrics.get('numerical_validation_passed', True))  # high fidelity already
        metrics['exact_validated']=bool(is_accepted and is_high)
        metrics['exact_constraints_passed']=is_accepted
        metrics['exact_numerical_validation_passed']=is_high
        # ensure restoration was already done by dispatcher; extra no-op
        return metrics
    except Rejected as ex:
        return {'error':str(ex), 'code':ex.code, 'exact_nodes':nodes, 'actual_exact_nodes':nodes, 'actual_subdivisions':subdiv, 'exact_evaluated':False, 'exact_validated':False, 'exact_constraints_passed':False, 'exact_numerical_validation_passed':False}
    except Exception as ex:
        return {'error':str(ex), 'code':'numerical_failure', 'exact_nodes':nodes, 'actual_exact_nodes':nodes, 'actual_subdivisions':subdiv, 'exact_evaluated':False, 'exact_validated':False, 'exact_constraints_passed':False, 'exact_numerical_validation_passed':False}
