"""GA Engine — orchestrates islands, migration, hall of fame, checkpoints, dashboard state.

Real-time dashboard writes are async via atomic file replace, never blocks evolution.
Trap tab: separate tab draws trap itself and animates shuttling for each island's top 3% best candidates.
Design is non-blocking: file polling, atomic replace, cheap decimated ROM, interval + theta-hash cache.
"""
import os as _os
_os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import time, json, pathlib, threading, queue, math
import numpy as np
from pathlib import Path

from .genome import Genome
from .island import Island
from .migration import RingMigration
from .fitness import evaluate_exact
from .cuda import cuda_info, get_xp
from ..observables import scalar_loss_from_observables, dimensionless_control_penalty

class GAEngine:
    def __init__(self, runner, config, out_dir="output/ga_run"):
        """
        runner: prepared Runner (already prepare() called)
        config: dict with keys:
          - islands: [{"strategy":"S","pop":20,"baseline":"sta"}, ...]
          - generations: int
          - migration_interval: int
          - n_migrants: int
          - hall_size: int
          - seed: int
        out_dir: Path for checkpoints and state.json
        """
        self.runner=runner
        self.cfg=config
        self.out=Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out/"history").mkdir(exist_ok=True)
        self.rng=np.random.default_rng(config.get('seed', 42))
        # CUDA: использовать CuPy если есть и не отключено
        try:
            _cuda_tmp=cuda_info()
        except Exception:
            _cuda_tmp={"has_cupy":False,"has_numba":False}
        # сохраняем cuda_info для дашборда, но позже перезапишем self.cuda
        self._cuda_tmp = _cuda_tmp
        # correctly read use_cuda from ga section or top-level, and only enable if real CUDA device exists
        _cfg_use = config.get('use_cuda', None)
        if _cfg_use is None:
            _cfg_use = config.get('ga', {}).get('use_cuda', True) if isinstance(config.get('ga'), dict) else True
        _has_real_cuda = bool(_cuda_tmp.get('has_cupy') or _cuda_tmp.get('numba_cuda_available'))
        self.use_cuda = bool(_cfg_use) and _has_real_cuda
        try:
            if self.use_cuda and _cuda_tmp.get('has_cupy'):
                self.xp = get_xp()
                # sanity: get_xp should return cupy if has_cupy
                if getattr(self.xp, '__name__', 'numpy') != 'cupy':
                    # cupy import failed despite has_cupy flag (race) — fallback to numpy but keep use_cuda if numba cuda available
                    if not _cuda_tmp.get('numba_cuda_available'):
                        self.use_cuda = False
                    self.xp = np if getattr(self.xp, '__name__', 'numpy') != 'cupy' and not _cuda_tmp.get('numba_cuda_available') else self.xp
            elif self.use_cuda and _cuda_tmp.get('numba_cuda_available'):
                # numba CUDA available but cupy not — use numpy for xp, but flag CUDA true (kernel-level)
                self.xp = np
            else:
                self.xp = np
                if not _has_real_cuda:
                    self.use_cuda = False
        except Exception:
            self.xp = np
            self.use_cuda = False
        # глобальный pop_size переопределяет per-island pop если задан
        _global_pop = config.get('pop_size', config.get('pop', None))
        if _global_pop is not None:
            try:
                _global_pop = int(_global_pop)
                if _global_pop > 0:
                    for _ic in config.get('islands', []):
                        _ic['pop'] = _global_pop
                self._global_pop = _global_pop
            except Exception:
                self._global_pop = None
        else:
            self._global_pop = None
        self.islands=[]
        # процентный карантин: isolated_until можно задать как 12 (абсолют), "40%" или 0.4 / isolated_until_ratio
        def _resolve_isolated_until(raw, total_gens):
            if raw is None or raw == "":
                return 0
            if isinstance(raw, str):
                s = raw.strip()
                if s.endswith('%'):
                    try:
                        pct = float(s[:-1])
                        return int(round(pct/100.0 * total_gens))
                    except: 
                        return 0
                # строка-число
                try:
                    # "0.4" как дробь → процент, иначе абсолют
                    fv = float(s)
                    if 0 < fv < 1 and '.' in s:
                        return int(round(fv * total_gens))
                    return int(round(fv))
                except:
                    return 0
            if isinstance(raw, float):
                if 0 < raw < 1:
                    return int(round(raw * total_gens))
                return int(round(raw))
            # int или иное
            try:
                return int(raw)
            except:
                return 0
        total_gens = int(config.get('generations', 20))
        # карантин из дашборда: глобальный и per-island
        _global_quarantine_raw = config.get('quarantine_until', None)
        if _global_quarantine_raw is None:
            _global_quarantine_raw = config.get('ga', {}).get('quarantine_until', None) if isinstance(config.get('ga'), dict) else None
        if _global_quarantine_raw is None:
            # legacy alias isolated_until at top level
            _global_quarantine_raw = config.get('isolated_until', None)
            if _global_quarantine_raw is None and isinstance(config.get('ga'), dict):
                _global_quarantine_raw = config['ga'].get('isolated_until', None)
        _per_island_quarantine = {}
        # несколько возможных мест хранения per-island
        for src in [config.get('_quarantine_per_island'), config.get('quarantine_per_island'), (config.get('ga', {}) or {}).get('_quarantine_per_island'), (config.get('ga', {}) or {}).get('quarantine_per_island')]:
            if isinstance(src, dict):
                for k,v in src.items():
                    _per_island_quarantine[str(k)] = v
            elif isinstance(src, list):
                for item in src:
                    if isinstance(item, dict) and 'island' in item:
                        _per_island_quarantine[str(item['island'])] = item.get('isolated_until', item.get('isolated_until_raw', item.get('value')))
        for i, icfg in enumerate(config.get('islands', [])):
            strat=icfg.get('strategy','S')
            pop=icfg.get('pop',20)
            base=icfg.get('baseline','sta')
            segs = icfg.get('segments', None)
            if segs is None and isinstance(strat, str):
                import re
                if len(strat)>1 and strat[0] in ('K','M','C'):
                    m=re.match(r'^[KMC](\d+)$', strat)
                    if m:
                        try: segs=int(m.group(1))
                        except: segs='auto'
                elif strat[0]=='P':
                    m=re.match(r'^P(\d+)$', strat)
                    if m:
                        try: segs=int(m.group(1))
                        except: segs='auto'
                    elif strat=='P':
                        segs='auto'
            raw_iso = icfg.get('isolated_until', icfg.get('isolated_until_ratio', 0))
            if raw_iso in (0, '', None) and isinstance(strat, str) and strat[0] in ('K','M','C','P'):
                raw_iso = total_gens  # K/P stitched/periodic isolated 100%
            # поддержка обоих ключей: isolated_until и isolated_until_ratio (0..1 или "40%")
            if 'isolated_until_ratio' in icfg and 'isolated_until' not in icfg:
                raw_iso = icfg['isolated_until_ratio']
            isolated_until = _resolve_isolated_until(raw_iso, total_gens)
            # dashboard override: per-island и глобальный карантин для K/P
            if strat in _per_island_quarantine:
                raw_iso = _per_island_quarantine[strat]
                isolated_until = _resolve_isolated_until(raw_iso, total_gens)
            elif _global_quarantine_raw is not None and isinstance(strat, str) and strat[0] in ('K','M','C','P') and raw_iso == total_gens:
                raw_iso = _global_quarantine_raw
                isolated_until = _resolve_isolated_until(raw_iso, total_gens)
            isl=Island(i, strat, pop, rng=self.rng.integers(0,1<<31), baseline=base, isolated_until=isolated_until, segments=segs)
            # сохраним исходное значение для дашборда/дебага
            isl.isolated_until_raw = raw_iso
            isl.isolated_until_percent = (isolated_until/total_gens*100) if total_gens else 0
            isl.initialize()
            self.islands.append(isl)
        if not self.islands:
            # default 6 islands — разные базы, не только STA
            # For large D (≥80um) reduce pop to keep GA tractable (100um 10 islands *20 pop *1.3s = 260s per gen -> too long)
            try:
                _dist_m = float(getattr(getattr(self, 'runner', None), 'task', {}).get('distance_m', 0) or 0)
            except:
                _dist_m = 0
            _large = _dist_m >= 80e-6
            _def_pop = int(_global_pop) if (_global_pop is not None and int(_global_pop)>0) else None
            if _def_pop is None:
                if _large:
                    defaults=[("S", 6, "smooth"),("A", 6,"sta"),("V", 6,"sta"),("W", 6,"sta"),("R", 6,"sta"),("H", 6,"smooth")]
                else:
                    defaults=[("S", 20, "smooth"),("A", 20,"sta"),("V", 20,"sta"),("W", 20,"sta"),("R", 25,"sta"),("H", 20,"smooth")]
            else:
                defaults=[("S", _def_pop, "smooth"),("A", _def_pop,"sta"),("V", _def_pop,"sta"),("W", _def_pop,"sta"),("R", _def_pop,"sta"),("H", _def_pop,"smooth")]
            # S=smooth адиабатика, A=STA короткий, V=voltage, W=frequency, R=robust, H=hardware/lowpass
            for i,(s,p,b) in enumerate(defaults):
                isl=Island(i,s,p, rng=self.rng.integers(0,1<<31), baseline=b)
                isl.initialize()
                self.islands.append(isl)
        # --- Auto-add KRAT (stitched) + PERIODIC (no-stop) isolated islands for arbitrary D ---
        # K: stitched with holds; P: periodic no-stop (single trajectory, voltage K-periodic).
        # Both generalize for any D: repeats='auto' -> round(D/20um). Also add explicit K5/P5 for comparison.
        # For 100um/100us: P2(2x50), P5(5x20), P10(10x10) all 100um, hall picks minimal n (expect P5 best).
        try:
            has_krat = any(str(isl.strategy).startswith('K') or str(isl.strategy).startswith('M') or str(isl.strategy).startswith('C') for isl in self.islands)
            has_periodic = any(str(isl.strategy).startswith('P') for isl in self.islands)
            auto_add = config.get('auto_add_krat', True)
            # periodic default True independent of krat (so 100um always gets P even if krat config has auto_add_krat:false)
            if 'auto_add_periodic' in config:
                auto_add_p = bool(config.get('auto_add_periodic'))
            else:
                auto_add_p = True
            dist_m = float(getattr(getattr(self, 'runner', None), 'task', {}).get('distance_m', 0) if hasattr(self, 'runner') else config.get('task',{}).get('distance_m',0) or 0)
            if dist_m >= 25e-6:
                import math
                k_auto = max(2, min(10, int(round(dist_m/20e-6)))) if dist_m else 5
                to_add = []
                if auto_add and not has_krat:
                    # smaller pops for large D to keep GA tractable (100um 10 islands *12 pop =120 evals *4s=480s per gen -> too long)
                    _pop_k = 6 if dist_m >= 80e-6 else 8
                    to_add.append(('K', _pop_k, 'sta', 'auto'))
                    if k_auto != 5:
                        to_add.append(('K5', 8, 'sta', 5))
                if auto_add_p and not has_periodic:
                    _pop_p = 6 if dist_m >= 80e-6 else 8
                    to_add.append(('P', _pop_p, 'sta', 'auto'))
                    # also add explicit alternatives for optimal repeat search
                    # 5x20 is often best for 100um, but 10x10 and 2x50 must also be tried — hall picks min n
                    alt = []
                    if k_auto != 5:
                        alt.append(5)
                    if k_auto != 2:
                        alt.append(2)
                    if dist_m >= 40e-6 and k_auto != 10:
                        alt.append(10)
                    if dist_m >= 60e-6 and k_auto != 3 and 3 not in alt:
                        alt.append(3)
                    # add up to 2 alts beyond auto (keep population reasonable)
                    for rep in alt[:2]:
                        strat = f'P{rep}'
                        if not any(s==strat for s,_,_,_ in to_add):
                            _pop_alt = 4 if dist_m >= 80e-6 else 6
                            to_add.append((strat, _pop_alt, 'sta', rep))
                for strat, pop, base, segs in to_add:
                    if any(isl.strategy==strat for isl in self.islands):
                        continue
                    raw_iso_auto = total_gens
                    if strat in _per_island_quarantine:
                        raw_iso_auto = _per_island_quarantine[strat]
                    elif _global_quarantine_raw is not None:
                        raw_iso_auto = _global_quarantine_raw
                    iso = _resolve_isolated_until(raw_iso_auto, total_gens)
                    isl = Island(len(self.islands), strat, pop, rng=self.rng.integers(0,1<<31), baseline=base, isolated_until=iso, segments=segs)
                    isl.isolated_until_raw = raw_iso_auto
                    isl.isolated_until_percent = (iso/total_gens*100) if total_gens else 0
                    isl.initialize()
                    self.islands.append(isl)
        except Exception as e:
            pass
        # also handle bare 'auto_add_periodic' flag for periodic-only disabling
        
        self.migration=RingMigration(self.islands, interval=config.get('migration_interval',5), n_migrants=config.get('n_migrants',2))
        self.generations=config.get('generations', 20)
        self.hall_size=config.get('hall_size', 10)
        self.hall_of_fame=[]  # list of Genome sorted
        self.global_gen=0
        self.start_time=time.time()
        # reuse earlier cuda info (already computed) or fresh if not yet
        try:
            self.cuda = getattr(self, '_cuda_tmp', cuda_info())
            # enrich with xp info
            self.cuda['use_cuda'] = bool(self.use_cuda)
            self.cuda['xp'] = 'cupy' if getattr(self, 'xp', np).__name__ == 'cupy' else 'numpy'
            self.cuda['xp_name'] = getattr(self.xp, '__name__', 'numpy')
        except Exception:
            self.cuda=cuda_info()
        # dashboard state queue (non-blocking)
        self._state_queue=queue.Queue(maxsize=10)
        self._dashboard_thread=None
        # trap geometry — built once, never blocks per-gen
        try:
            self._trap_geometry = self._build_trap_geometry()
        except Exception as e:
            self._trap_geometry = {"error": str(e), "kind": self.cfg.get('trap',{}).get('kind','unknown')}
        self._traj_cache = {}
        self._cached_island_trajs = None
        self._cached_island_trajs_gen = -1
        self._loss_history = []  # for loss-tab timeline: list of {gen, breakdown}
        self._robustness_cache = {}  # gen -> robustness snapshot
        self._sion_export_cache = None

    def _build_trap_geometry(self):
        """Export trap electrode polygons for dashboard Trap tab.

        Runs once at init (expensive field lookup) and then cached.
        Returns JSON-serializable dict with electrodes in µm for frontend to draw.
        No per-generation cost → never slows training.
        """
        runner = self.runner
        system = runner.trap.system
        electrodes=[]
        # helper to decide scale: if coordinates abs mean < 1 -> meters, need *1e6
        sample_scale = None
        try:
            # peek first path
            first = system[0].paths[0]
            arr = np.asarray(first)
            mx = np.abs(arr).max()
            sample_scale = 1e6 if mx < 1 else 1.0
        except:
            sample_scale = 1.0
        for idx, ele in enumerate(system):
            try:
                rf_val = float(system.rfs[idx]) if idx < len(system.rfs) else 0.0
            except:
                rf_val = 0.0
            typ = "rf" if abs(rf_val) > 1e-12 else "dc"
            # name-based heuristic: rf contains 'rf'
            if 'rf' in ele.name.lower():
                typ = "rf"
            elif 'dc' in ele.name.lower():
                typ = "dc"
            paths=[]
            for path in getattr(ele, 'paths', []):
                arr = np.asarray(path, dtype=float)
                if sample_scale == 1e6:
                    arr = arr * 1e6
                # for five_wire the coords are already in µm (~1e4), keep as µm
                # round to 2 decimals to shrink json
                lst = np.round(arr, 2).tolist()
                paths.append(lst)
            electrodes.append(dict(name=ele.name, type=typ, paths=paths, rf=float(rf_val) if np.isfinite(rf_val) else 0.0))
        # also include config geometry for schematic fallback (if GDS not using polygons)
        try:
            start = runner.start.tolist()
            target = runner.target.tolist()
            direction = runner.direction.tolist()
        except:
            start = target = direction = None
        # bbox
        try:
            all_pts = np.concatenate([np.concatenate([np.asarray(p) for p in e["paths"]] ) for e in electrodes if e["paths"]], axis=0) if electrodes else np.array([[0,0]])
            xmin, ymin = all_pts.min(axis=0).tolist()
            xmax, ymax = all_pts.max(axis=0).tolist()
        except:
            xmin=ymin=xmax=ymax=0
        # real fitted height (more accurate than seed) + voltage bounds for dashboard
        try:
            _real_h = float(runner.initial.position_m[2])
        except Exception:
            _real_h = float(runner.cfg['trap'].get('minimum_seed_m', [0,0,90e-6])[2] if isinstance(runner.cfg['trap'].get('minimum_seed_m'), list) else 90e-6)
        try:
            _lo = runner.trap.lo.tolist() if hasattr(runner.trap.lo, 'tolist') else list(runner.trap.lo)
            _hi = runner.trap.hi.tolist() if hasattr(runner.trap.hi, 'tolist') else list(runner.trap.hi)
        except Exception:
            _lo = runner.cfg['trap'].get('lower_V', -12)
            _hi = runner.cfg['trap'].get('upper_V', 12)
        geom=dict(
            kind=runner.cfg['trap']['kind'],
            electrodes=electrodes,
            bbox=dict(xmin=float(xmin), xmax=float(xmax), ymin=float(ymin), ymax=float(ymax)),
            start_m=start, target_m=target, direction=direction,
            distance_m=float(runner.task.get('distance_m', 0)),
            duration_s=float(runner.task.get('duration_s', 0)),
            hold_s=float(runner.task.get('hold_s', 0)),
            height_m=float(_real_h),
            geometry_raw=runner.cfg['trap'].get('geometry', {}),
            length_scale_m=float(runner.cfg['trap'].get('length_scale_m', 1e-6)),
            dc_names=runner.trap.names,
            dc_indices=runner.trap.indices.tolist() if hasattr(runner.trap.indices,'tolist') else list(runner.trap.indices),
            initial_dc_V=runner.initial_V.tolist() if hasattr(runner.initial_V,'tolist') else list(runner.initial_V),
            # for schematic five_wire, also provide parsed wire widths in µm
            shuttle_axis=[1,0,0] if direction is None else direction,
            lower_V=_lo,
            upper_V=_hi,
            dac_min_V=float(runner.hw.dac_min_V),
            dac_max_V=float(runner.hw.dac_max_V),
            rf_peak_V=float(runner.cfg['trap'].get('rf_peak_V', 0)),
            rf_frequency_hz=float(runner.cfg['trap'].get('rf_frequency_hz', 0)),
            seed_height_m=float(runner.cfg['trap'].get('minimum_seed_m', [0,0,90e-6])[2] if isinstance(runner.cfg['trap'].get('minimum_seed_m'), list) and len(runner.cfg['trap'].get('minimum_seed_m'))>=3 else 90e-6),
        )
        return geom

    def _get_island_top_trajectories(self, heavy=True):
        """Per-island top 3% best trajectories + voltages + frequency + classical for Trap tab.

        Non-blocking: cheap ROM decimated to ~100 pts, interval cache, theta-hash.
        Voltage/frequency are decimated similarly (≈120 pts, not full 400) to keep JSON <200kB.
        Classical only for rank0 (full-field centroid, expensive) every 5 gens or on demand.
        heavy=False skips classical and uses cached trajectories.
        """
        percent = float(self.cfg.get('island_top_percent', self.cfg.get('ga',{}).get('island_top_percent', 0.03)) if isinstance(self.cfg.get('ga'), dict) else self.cfg.get('island_top_percent', 0.03))
        if 'ga' in self.cfg and isinstance(self.cfg['ga'], dict) and 'island_top_percent' in self.cfg['ga']:
            percent = float(self.cfg['ga']['island_top_percent'])
        decimate = int(self.cfg.get('trap_traj_decimate', 80))
        if 'trap_traj_decimate' in self.cfg.get('ga', {}):
            decimate = int(self.cfg['ga']['trap_traj_decimate'])
        interval = int(self.cfg.get('trap_traj_interval', 2))
        if 'trap_traj_interval' in self.cfg.get('ga', {}):
            interval = int(self.cfg['ga']['trap_traj_interval'])
        if self._cached_island_trajs is not None and self.global_gen !=0 and self.global_gen % interval !=0 and self.global_gen != self.generations:
            return self._cached_island_trajs
        result=[]
        max_per_island = int(self.cfg.get('trap_traj_max_per_island', 3))
        # voltage decimation for frontend (not full DAC 300+ pts)
        voltage_decimate = max(1, decimate // 2)
        for isl in self.islands:
            pop_valid = [g for g in isl.population if g.fitness is not None]
            if not pop_valid:
                pop_sorted = sorted(isl.population, key=lambda g: 1e9 if g.fitness is None else g.fitness)
            else:
                pop_sorted = sorted(pop_valid, key=lambda g: g.fitness)
            k = max(1, int(math.ceil(len(isl.population)*percent)))
            k = min(k, max_per_island, len(pop_sorted))
            top = pop_sorted[:k]
            cands=[]
            for rank, g in enumerate(top):
                try:
                    theta_key = tuple(np.round(g.theta,5).tolist())
                except:
                    theta_key = tuple()
                cache_key = (isl.strategy, isl.baseline, theta_key, decimate, voltage_decimate)
                cached = self._traj_cache.get(cache_key)
                use_cached = cached is not None and (self.global_gen % 5 != 0)
                if use_cached:
                    traj = cached.get('trajectory')
                    voltages = cached.get('voltages')
                    frequency = cached.get('frequency')
                    classical = cached.get('classical')
                else:
                    traj = None; voltages=None; frequency=None; classical=None
                    # handle W island frequency variable prepare (same as evaluate_genome)
                    is_W = isl.strategy in ('W','frequency','freq') and getattr(g,'dim', len(g.theta))>=10
                    stashed = None
                    freq_coeffs = None
                    theta_for_eval = g.theta.copy()
                    baseline = isl.baseline
                    if is_W:
                        # split theta: first 8 voltage, last 2 frequency
                        try:
                            theta_for_eval = g.theta[:8].copy()
                            freq_coeffs = g.theta[8:10].copy()
                            from .fitness import _prepare_for_W
                            # stash runner state
                            stashed = dict(
                                task_mode=self.runner.task.get('frequency_mode'),
                                task_coeffs=self.runner.task.get('frequency_coeffs'),
                                freq_mode=getattr(self.runner, 'frequency_mode', 'fixed'),
                                freq_coeffs=getattr(self.runner, 'frequency_coeffs', [0.0,0.0]),
                            )
                            # also stash prepared arrays used by pipeline
                            for attr in ['sroute','route_V','voltage_spline','route_r','route_H','route_G','route_BH','Hspline','Gspline','BHspline','correction_modes','support_diagnostics']:
                                if hasattr(self.runner, attr):
                                    stashed[attr]=getattr(self.runner, attr)
                            _prepare_for_W(self.runner, freq_coeffs)
                        except Exception:
                            stashed=None
                    # handle K/M/C stitched islands: use segmented evaluate
                    is_stitched_island = isinstance(isl.strategy, str) and (isl.strategy.startswith('K') or isl.strategy.startswith('M') or isl.strategy.startswith('C') or isl.strategy in ('STITCH','KRAT'))
                    is_periodic_island = isinstance(isl.strategy, str) and (isl.strategy.startswith('P') or isl.strategy in ('PERIODIC',))
                    stitched_segs = getattr(isl, 'segments', None)
                    if is_stitched_island:
                        # parse segments from strategy if not set
                        if stitched_segs is None:
                            import re
                            m=re.match(r'^[KMC](\d+)$', isl.strategy)
                            if m:
                                try: stitched_segs=int(m.group(1))
                                except: stitched_segs='auto'
                            else:
                                stitched_segs='auto'
                        # clamp
                        if isinstance(stitched_segs, int) and stitched_segs<2: stitched_segs=2
                        if isinstance(stitched_segs, int) and stitched_segs>10: stitched_segs=10
                    try:
                        if is_stitched_island:
                            res, trace = self.runner.evaluate_segmented(theta_for_eval, baseline=baseline, segments=stitched_segs, subdivisions=self.runner.cfg.get('subdivisions',4), keep=True, return_trajectory=True, trajectory_decimate=decimate)
                        elif is_periodic_island:
                            res, trace = self.runner.evaluate_krat_periodic(theta_for_eval, baseline=baseline, repeats=stitched_segs, subdivisions=self.runner.cfg.get('subdivisions',4), keep=True, return_trajectory=True, trajectory_decimate=decimate)
                        else:
                            res, trace = self.runner.evaluate(theta_for_eval, baseline=baseline, exact_nodes=None, subdivisions=self.runner.cfg.get('subdivisions',4), keep=True, return_trajectory=True, trajectory_decimate=decimate)
                        traj = res.get('trajectory')
                        if isinstance(traj, dict):
                            traj = dict(traj)
                            traj['_n_total'] = res.get('n_total')
                            traj['_pos_err'] = res.get('position_error_m')
                            traj['_well_err'] = res.get('well_target_error_m')
                        # voltages: from same trace (already have t,v)
                        try:
                            # trace already from combined call
                            t = trace.get('t'); v = trace.get('v'); cmd = trace.get('cmd'); ct = trace.get('ct')
                            # decimate for frontend
                            if t is not None and v is not None:
                                step = max(1, len(t)//120)
                                idx = np.arange(0, len(t), step)
                                # also ensure last point
                                if idx[-1] != len(t)-1:
                                    idx = np.r_[idx, len(t)-1]
                                v_dec = v[idx]
                                t_dec = t[idx]
                                # also need electrode names
                                voltages = dict(
                                    times_s = t_dec.tolist(),
                                    voltages = v_dec.tolist(),  # [time][electrode]
                                    electrode_names = list(self.runner.trap.names),
                                    electrode_indices = self.runner.trap.indices.tolist() if hasattr(self.runner.trap.indices,'tolist') else list(self.runner.trap.indices),
                                    ideal_times_s = ct[::max(1,len(ct)//120)].tolist() if ct is not None else None,
                                    ideal_voltages = cmd[::max(1,len(cmd)//120)].tolist() if cmd is not None else None,
                                    decimate = int(step),
                                )
                            # frequency vs time: from H (if available) or from profile scaling
                            try:
                                H = trace.get('H')
                                if H is not None:
                                    # H shape (N,3,3) -> eig per time
                                    # use decimated H to match t_dec
                                    H_dec = H[idx] if 'idx' in locals() else H[::step]
                                    freqs = []
                                    for h in H_dec:
                                        try:
                                            eig = np.linalg.eigvalsh(h/self.runner.trap.mass)
                                            f = float(np.sqrt(max(eig.min(), 0))/(2*np.pi))
                                            freqs.append(f)
                                        except:
                                            freqs.append(float(self.runner.task.get('min_frequency_hz',400e3)))
                                    frequency = dict(times_s = t_dec.tolist(), frequency_hz = freqs, baseline_freq_hz = float(self.runner.initial.omega_rad_s[0]/(2*np.pi)))
                                else:
                                    # fallback: from sroute scaling
                                    s_vals = np.interp(t_dec, trace.get('ct', t_dec), trace.get('route', np.linspace(0,1,len(t_dec))))
                                    from ..control import FrequencyProfile
                                    fp = FrequencyProfile(mode=self.runner.task.get('frequency_mode','fixed'), coeff=np.asarray(self.runner.task.get('frequency_coeffs',[0,0]),float))
                                    scales = fp.omega_scaling(np.clip(s_vals,0,1))
                                    base = float(self.runner.initial.omega_rad_s[0]/(2*np.pi))
                                    frequency = dict(times_s = t_dec.tolist(), frequency_hz = (base*scales).tolist(), baseline_freq_hz=base)
                            except Exception:
                                frequency = None
                        except Exception:
                            voltages=None; frequency=None
                        # честный classical ODE (full Sion pseudopotential centroid) для rank0
                        if rank==0:
                            # heavy+не кэш → честный ODE, иначе лёгкий r для скорости
                            try:
                                # honest but decimated: 2041->80 honest field solves, ~25x faster
                                do_honest = heavy and (not use_cached) and ((self.global_gen % 4 == 0 and self.global_gen != 0) or self.global_gen == self.generations)
                                if do_honest:
                                    from ..dynamics import validate_full_field_centroid
                                    t_for_c = trace.get('t'); v_for_c = trace.get('v')
                                    if t_for_c is not None and v_for_c is not None:
                                        try:
                                            step_cc = max(1, len(t_for_c)//20)
                                            idx_cc = np.arange(0, len(t_for_c), step_cc)
                                            if idx_cc[-1] != len(t_for_c)-1:
                                                idx_cc = np.r_[idx_cc, len(t_for_c)-1]
                                            t_cc = t_for_c[idx_cc]; v_cc = v_for_c[idx_cc]
                                        except Exception:
                                            t_cc = t_for_c; v_cc = v_for_c
                                        classical_raw = validate_full_field_centroid(
                                            self.runner.trap, t_cc, v_cc,
                                            trace.get('initial', self.runner.initial),
                                            self.runner.target if self.runner.task.get('protocol')=='one_way' else self.runner.start,
                                            return_trajectory=True, trajectory_decimate=decimate)
                                        classical = classical_raw.get('trajectory')
                                        if classical is not None:
                                            classical['note'] = 'full_field_centroid'
                                            classical['coherent_n_total'] = classical_raw.get('coherent_n_total')
                                        else:
                                            classical = None
                                    else:
                                        classical = None
                                else:
                                    # лёгкий fallback для кэша/light обновлений
                                    r = trace.get('r') if 'trace' in locals() else None
                                    t_for_c = trace.get('t') if 'trace' in locals() else None
                                    if r is not None and t_for_c is not None:
                                        step_c = max(1, len(t_for_c)//decimate)
                                        idx_c = np.arange(0, len(t_for_c), step_c)
                                        if idx_c[-1] != len(t_for_c)-1:
                                            idx_c = np.r_[idx_c, len(t_for_c)-1]
                                        classical = dict(times_s=t_for_c[idx_c].tolist(), positions_m=r[idx_c].tolist(), decimate=int(step_c), note="well_centers_fast")
                                    else:
                                        classical=None
                            except Exception as _e_cls:
                                # fallback к лёгкому если ODE упал
                                try:
                                    r = trace.get('r') if 'trace' in locals() else None
                                    t_for_c = trace.get('t') if 'trace' in locals() else None
                                    if r is not None and t_for_c is not None:
                                        step_c = max(1, len(t_for_c)//decimate)
                                        idx_c = np.arange(0, len(t_for_c), step_c)
                                        if idx_c[-1] != len(t_for_c)-1:
                                            idx_c = np.r_[idx_c, len(t_for_c)-1]
                                        classical = dict(times_s=t_for_c[idx_c].tolist(), positions_m=r[idx_c].tolist(), decimate=int(step_c), note="well_centers_fast_fallback", error=str(_e_cls)[:200])
                                    else:
                                        classical=None
                                except:
                                    classical=None
                        else:
                            classical=None
                        # cache
                        self._traj_cache[cache_key] = dict(trajectory=traj, voltages=voltages, frequency=frequency, classical=classical)
                        if len(self._traj_cache) > 300:
                            for kk in list(self._traj_cache.keys())[:50]:
                                del self._traj_cache[kk]
                    except Exception as e:
                        traj = {'error': str(e)[:300]}
                        voltages=None; frequency=None; classical=None
                    finally:
                        if stashed is not None:
                            try:
                                self.runner.task['frequency_mode'] = stashed.get('task_mode') if stashed.get('task_mode') is not None else 'fixed'
                                self.runner.task['frequency_coeffs'] = stashed.get('task_coeffs') if stashed.get('task_coeffs') is not None else [0.0,0.0]
                                for k in ['sroute','route_V','voltage_spline','route_r','route_H','route_G','route_BH','Hspline','Gspline','BHspline','correction_modes','support_diagnostics','frequency_mode','frequency_coeffs']:
                                    if k in stashed and stashed[k] is not None:
                                        setattr(self.runner, k, stashed[k])
                            except Exception:
                                pass
                # for cached case, traj already; need to extract others from cache dict
                if use_cached:
                    # cached is dict with keys
                    traj = cached.get('trajectory') if isinstance(cached, dict) else cached
                    voltages = cached.get('voltages') if isinstance(cached, dict) else None
                    frequency = cached.get('frequency') if isinstance(cached, dict) else None
                    classical = cached.get('classical') if isinstance(cached, dict) else None
                    # if old cache was just trajectory dict (before upgrade), it will be just traj
                    if isinstance(cached, dict) and 'times_s' in cached and 'mean_positions_m' in cached:
                        # old format: cached is traj itself
                        traj = cached; voltages=None; frequency=None; classical=None
                cands.append(dict(rank=rank, theta=g.theta.tolist() if hasattr(g.theta,'tolist') else list(g.theta), fitness=float(g.fitness) if g.fitness is not None else None, status=g.status, baseline=isl.baseline, island=g.island, dim=int(g.dim) if hasattr(g,'dim') else len(g.theta), trajectory=traj, voltages=voltages, frequency=frequency, classical_trajectory=classical))
            result.append(dict(island_id=isl.id, strategy=isl.strategy, pop=len(isl.population), baseline=isl.baseline, isolated_until=isl.isolated_until, is_isolated=isl.generation < isl.isolated_until, k=k, percent=percent, candidates=cands))
        self._cached_island_trajs = result
        self._cached_island_trajs_gen = self.global_gen
        return result

    def _update_hall(self):
        # collect all finite — include not-ok but finite fitness (strict tol makes ok rare)
        all_genomes=[]
        for isl in self.islands:
            for g in isl.population:
                if g.fitness is not None and g.fitness < 1e6:
                    all_genomes.append(g)
        if not all_genomes:
            for isl in self.islands:
                for g in isl.population:
                    if g.fitness is not None:
                        all_genomes.append(g)
        all_genomes.sort(key=lambda g: g.fitness)
        # --- diversity-aware hall: not just top-N by fitness, but distinct trajectories ---
        # STA — лишь одна из инициализаций, GA ищет во всём пространстве 8-10 генов.
        # Чтобы зал не схлопнулся в один STA-локальный минимум, фильтруем по θ-расстоянию
        # (прокси траектории: theta→ΔV→смещение минимума).
        min_dist = float(self.cfg.get('hall_diversity_min_distance', 0.12))
        diverse_hall=[]
        for g in all_genomes:
            if not diverse_hall:
                diverse_hall.append(g)
                continue
            dmin = min(g.distance(h) for h in diverse_hall)
            if dmin >= min_dist:
                diverse_hall.append(g)
            # пропускаем близкого конкурента STA-кластера, ищем следующую ветвь
            if len(diverse_hall) >= self.hall_size:
                break
        # fallback: если разнообразных не хватило (все в одном кластере), добиваем лучшими по fitness
        if len(diverse_hall) < self.hall_size:
            for g in all_genomes:
                if g not in diverse_hall:
                    diverse_hall.append(g)
                    if len(diverse_hall) >= self.hall_size:
                        break
        self.hall_of_fame=diverse_hall[:self.hall_size]
        # store diagnostic: mean pairwise distance inside hall for dashboard
        self._hall_diversity = 0.0
        if len(self.hall_of_fame) > 1:
            ds=[]
            for i in range(len(self.hall_of_fame)):
                for j in range(i+1, len(self.hall_of_fame)):
                    try:
                        ds.append(self.hall_of_fame[i].distance(self.hall_of_fame[j]))
                    except: pass
            self._hall_diversity = float(np.mean(ds)) if ds else 0.0

    def _compute_loss_breakdown(self, genome):
        """Вернуть разложение fitness по компонентам для дашборда loss-tab."""
        try:
            m = genome.metrics or {}
            task = self.runner.task
            # веса из runner или дефолт
            w = dict(self.runner.loss_weights or {"n":1.0,"pos":1.0,"well":1.0,"lag":0.0})
            # скалярные входы
            n = float(m.get('n_total', 0))
            pos = float(m.get('position_error_m', 0))
            well = float(m.get('well_target_error_m', 0))
            lag = m.get('max_ion_well_lag_m', m.get('max_transport_error_m'))
            n0 = float(task.get('max_n_total', 0.5))
            tol = float(task.get('position_tolerance_m', 5e-6))
            dist = float(task.get('distance_m', 30e-6))
            # базовые термы
            import math as _math
            n_term = w.get('n',1.0) * _math.log1p(max(0,n)/max(n0,1e-12)) if n0>0 else 0
            pos_term = w.get('pos',1.0) * _math.log1p((max(0,pos)/max(tol,1e-12))**2) if tol>0 else 0
            well_term = w.get('well',1.0) * _math.log1p((max(0,well)/max(tol,1e-12))**2) if tol>0 else 0
            lag_term = 0
            if w.get('lag',0):
                try:
                    if lag is not None and not bool(m.get('is_max_transport_error_fallback', True)):
                        lag_term = w['lag'] * _math.log1p((float(lag)/max(dist,1e-12))**2)
                except: lag_term=0
            # distortion
            distortion_term = 0
            try:
                d = m.get('distortion')
                if isinstance(d, dict):
                    rms = float(d.get('rms_deviation_V', d.get('max_deviation_V',0)) or 0)
                else:
                    rms = float(d or 0)
                if rms>0.1:
                    distortion_term = 0.5*_math.log1p((rms-0.1)/0.1)
            except: pass
            # control penalty
            control_term = 0
            try:
                theta = genome.theta
                bounds = genome.bounds
                control_term = dimensionless_control_penalty(theta, bounds, self.runner.cfg.get('control_regularization_weight',1e-4))
            except: control_term=0
            # frequency gate (min + max + initial/final мягкие штрафы — должны совпадать с fitness.py)
            freq_term = 0
            try:
                fmin = float(m.get('minimum_frequency_hz', task.get('min_frequency_hz',400e3)))
                f0 = float(task.get('min_frequency_hz',400e3))
                if fmin < f0:
                    freq_term += 2*_math.log1p((f0-fmin)/0.1e6)
                # max frequency (soft) — если задан max_frequency_hz и превышен
                if 'max_frequency_hz' in task and task['max_frequency_hz'] is not None and 'maximum_frequency_hz' in m:
                    try:
                        fmax = float(m.get('maximum_frequency_hz', 0))
                        fmax_allowed = float(task['max_frequency_hz'])
                        if fmax > fmax_allowed:
                            freq_term += 2*_math.log1p((fmax - fmax_allowed)/0.1e6)
                    except: pass
                # initial/final frequency targets
                if 'frequency_initial_hz' in task and task['frequency_initial_hz'] is not None and 'frequency_start_hz' in m:
                    try:
                        f_start = float(m.get('frequency_start_hz', fmin))
                        target = float(task['frequency_initial_hz'])
                        err = abs(f_start - target)
                        if err > 0.02e6:
                            freq_term += 1.0*_math.log1p(err/0.05e6)
                    except: pass
                if 'frequency_final_hz' in task and task['frequency_final_hz'] is not None and 'frequency_end_hz' in m:
                    try:
                        f_end = float(m.get('frequency_end_hz', fmin))
                        target = float(task['frequency_final_hz'])
                        err = abs(f_end - target)
                        if err > 0.02e6:
                            freq_term += 1.0*_math.log1p(err/0.05e6)
                    except: pass
            except: pass
            # height soft (pipeline now soft)
            height_term = 0
            try:
                if m.get('height_violation_msg'):
                    height_term = 2*_math.log1p(1.0)  # ~1.38
                elif 'height_violation' in m.get('quality_failures', []) if isinstance(m.get('quality_failures'), list) else False:
                    height_term = 2*_math.log1p(1.0)
                # also check explicit well heights if available in metrics (not yet stored, keep 0)
            except: pass
            # heating rate term (new 2026-10-30)
            heating_term = 0
            try:
                hr = m.get('heating_rate_hz')
                if hr is not None:
                    allowed = task.get('max_heating_rate_hz', None)
                    if allowed is None:
                        try:
                            allowed = float(task.get('max_n_total', 0.5))/float(task.get('duration_s', 1))
                            if allowed > 100: allowed = 100
                        except: allowed = 100
                    if allowed is not None and hr > float(allowed)*1.05:
                        heating_term = 2*_math.log1p(max(0.0, hr - float(allowed))/max(float(allowed), 50.0))
            except: heating_term=0
            # robust для R — уже в fitness, выделяем отдельно?
            robust_extra = 0
            try:
                if genome.island=='R' and 'robust_train_vals' in m:
                    vals=m['robust_train_vals']
                    mean_v=sum(vals)/len(vals); max_v=max(vals)
                    # часть сверх обычного loss: 0.35*mean+0.65*max vs primary => approx max-mean
                    robust_extra = float(0.65*(max_v-mean_v))
            except: pass
            total = n_term+pos_term+well_term+lag_term+distortion_term+control_term+freq_term+height_term+heating_term+robust_extra
            # невязка vs реальный fitness (из-за округлений/biased) — height_term пока 0, для совместимости с суммой
            return dict(n=n_term, pos=pos_term, well=well_term, lag=lag_term, distortion=distortion_term, control=control_term, freq=freq_term, height=height_term, heating=heating_term, robust=robust_extra, sum=total, fitness=float(genome.fitness) if genome.fitness else total)
        except Exception as e:
            return dict(n=0,pos=0,well=0,lag=0,distortion=0,control=0,freq=0,height=0,heating=0,robust=0,sum=float(genome.fitness or 0), error=str(e))

    def _write_state(self, heavy=True):
        # build dashboard state; heavy=False skips expensive trajectory/robust/sion for per-island quick updates
        islands_state=[isl.snapshot() for isl in self.islands]
        hall=[]
        loss_breakdown=[]
        for g in self.hall_of_fame:
            entry=dict(theta=g.theta.tolist(), fitness=g.fitness, island=g.island, status=g.status,
                   n_total=g.metrics.get('n_total'), pos_err=g.metrics.get('position_error_m'),
                   fmin=g.metrics.get('minimum_frequency_hz'), distortion=g.metrics.get('distortion',{}).get('max_deviation_V') if isinstance(g.metrics.get('distortion'),dict) else None)
            # attach trajectory if already computed (for hall best we compute below)
            if 'trajectory' in g.metrics:
                entry['trajectory'] = g.metrics['trajectory']
            # loss breakdown for loss-tab
            try:
                bd = self._compute_loss_breakdown(g)
                entry['loss_breakdown'] = bd
                loss_breakdown.append(dict(island=g.island, fitness=g.fitness, breakdown=bd, theta=g.theta.tolist()))
            except Exception:
                pass
            hall.append(entry)
        # compute trajectory for best hall leader for dashboard (well + quantum + classical) — only if heavy
        trajectory = None
        if self.hall_of_fame:
            try:
                best = self.hall_of_fame[0]
                # heavy recompute only when needed and heavy=True; light keeps cached
                need_traj = ('trajectory' not in best.metrics or self.global_gen % 5 == 0 or self.global_gen == self.generations)
                if need_traj and heavy:
                    # use unified dispatcher for hall best (preserves W/K/P protocol, not to_theta8 truncation)
                    best_baseline = 'sta'
                    for isl in self.islands:
                        if isl.strategy == best.island:
                            best_baseline = isl.baseline
                            break
                    use_exact_traj = (self.global_gen == self.generations)
                    from .fitness import _dispatch_evaluate
                    exact_nodes_traj = self.runner.cfg.get('validation_nodes',321) if use_exact_traj else None
                    subdiv_traj = self.runner.cfg.get('validation_subdivisions',4) if use_exact_traj else self.runner.cfg.get('subdivisions',4)
                    # dispatch with trajectory
                    try:
                        # _dispatch_evaluate doesn't support return_trajectory, so we handle manually via runner.* but with island-aware dispatch
                        # For trajectory we need keep + return_trajectory, so we manually dispatch but unified logic:
                        island = best.island
                        is_W_traj = island in ('W','frequency','freq') and len(best.theta)>=10
                        is_stitched_traj = isinstance(island, str) and (island.startswith('K') or island.startswith('M') or island.startswith('C') or island in ('STITCH','KRAT'))
                        is_periodic_traj = isinstance(island, str) and (island.startswith('P') or island in ('PERIODIC',))
                        theta_traj = best.theta[:8].copy() if is_W_traj else best.theta
                        freq_traj = best.theta[8:10].copy() if is_W_traj else None
                        stashed_traj = None
                        if is_W_traj:
                            from .fitness import _prepare_for_W, _restore_after_W
                            try:
                                stashed_traj = dict(task_mode=self.runner.task.get('frequency_mode'), task_coeffs=self.runner.task.get('frequency_coeffs'),
                                                    freq_mode=getattr(self.runner, 'frequency_mode','fixed'), freq_coeffs=getattr(self.runner,'frequency_coeffs',[0.0,0.0]),
                                                    sroute=getattr(self.runner,'sroute',None), route_V=getattr(self.runner,'route_V',None), route_r=getattr(self.runner,'route_r',None),
                                                    route_H=getattr(self.runner,'route_H',None), route_G=getattr(self.runner,'route_G',None), route_BH=getattr(self.runner,'route_BH',None),
                                                    correction_modes=getattr(self.runner,'correction_modes',None), support_diagnostics=getattr(self.runner,'support_diagnostics',None),
                                                    voltage_spline=getattr(self.runner,'voltage_spline',None), Hspline=getattr(self.runner,'Hspline',None),
                                                    Gspline=getattr(self.runner,'Gspline',None), BHspline=getattr(self.runner,'BHspline',None))
                                _prepare_for_W(self.runner, freq_traj)
                            except: stashed_traj=None
                        try:
                            if is_stitched_traj:
                                import re
                                m=re.match(r'^[KMC](\d+)$', island)
                                segs = int(m.group(1)) if m else 'auto'
                                res_traj = self.runner.evaluate_segmented(theta_traj, baseline=best_baseline, segments=segs, exact_nodes=exact_nodes_traj, subdivisions=subdiv_traj, return_trajectory=True, trajectory_decimate=150)
                            elif is_periodic_traj:
                                import re
                                m=re.match(r'^P(\d+)$', island)
                                reps = int(m.group(1)) if m else 'auto'
                                res_traj = self.runner.evaluate_krat_periodic(theta_traj, baseline=best_baseline, repeats=reps, exact_nodes=exact_nodes_traj, subdivisions=subdiv_traj, return_trajectory=True, trajectory_decimate=150)
                            else:
                                res_traj = self.runner.evaluate(theta_traj, baseline=best_baseline, exact_nodes=exact_nodes_traj, subdivisions=subdiv_traj, return_trajectory=True, trajectory_decimate=150)
                        finally:
                            if stashed_traj is not None:
                                try:
                                    _restore_after_W(self.runner, stashed_traj)
                                except: pass
                    except Exception as _e_traj:
                        res_traj = {'error': str(_e_traj)[:300]}
                    best.metrics['trajectory'] = res_traj.get('trajectory') if isinstance(res_traj, dict) else None
                    # честный classical по умолчанию (пользователь попросил honest), флаг отключает только если явно False
                    _classical_cfg = self.cfg.get('trajectory_classical', None)
                    if _classical_cfg is None:
                        _classical_cfg = self.runner.cfg.get('trajectory_classical', None)
                    classical_enabled = True if _classical_cfg is None else bool(_classical_cfg)
                    if classical_enabled and ((self.global_gen % 4 == 0 and self.global_gen != 0) or self.global_gen == self.generations):
                        try:
                            from ..dynamics import validate_full_field_centroid
                            # for classical we need keep trace with same protocol (unified)
                            is_W_c = island in ('W','frequency','freq') and len(best.theta)>=10
                            theta_c = best.theta[:8].copy() if is_W_c else best.theta
                            freq_c = best.theta[8:10].copy() if is_W_c else None
                            stashed_c = None
                            if is_W_c:
                                try:
                                    stashed_c = dict(task_mode=self.runner.task.get('frequency_mode'), task_coeffs=self.runner.task.get('frequency_coeffs'),
                                                     freq_mode=getattr(self.runner, 'frequency_mode','fixed'), freq_coeffs=getattr(self.runner,'frequency_coeffs',[0.0,0.0]),
                                                     sroute=getattr(self.runner,'sroute',None), route_V=getattr(self.runner,'route_V',None), route_r=getattr(self.runner,'route_r',None),
                                                     route_H=getattr(self.runner,'route_H',None), route_G=getattr(self.runner,'route_G',None), route_BH=getattr(self.runner,'route_BH',None),
                                                     correction_modes=getattr(self.runner,'correction_modes',None), support_diagnostics=getattr(self.runner,'support_diagnostics',None),
                                                     voltage_spline=getattr(self.runner,'voltage_spline',None), Hspline=getattr(self.runner,'Hspline',None),
                                                     Gspline=getattr(self.runner,'Gspline',None), BHspline=getattr(self.runner,'BHspline',None))
                                    _prepare_for_W(self.runner, freq_c)
                                except: stashed_c=None
                            try:
                                if is_stitched_traj:
                                    _, trace = self.runner.evaluate_segmented(theta_c, baseline=best_baseline, segments=segs if 'segs' in locals() else 'auto', exact_nodes=120, subdivisions=4, keep=True)
                                elif is_periodic_traj:
                                    _, trace = self.runner.evaluate_krat_periodic(theta_c, baseline=best_baseline, repeats=reps if 'reps' in locals() else 'auto', exact_nodes=120, subdivisions=4, keep=True)
                                else:
                                    _, trace = self.runner.evaluate(theta_c, baseline=best_baseline, exact_nodes=120, subdivisions=4, keep=True)
                            finally:
                                if stashed_c is not None:
                                    try: _restore_after_W(self.runner, stashed_c)
                                    except: pass
                            # decimate for honest classical speed
                            try:
                                tt = trace['t']; vv = trace['v']
                                step_c = max(1, len(tt)//20)
                                idx_c = np.arange(0, len(tt), step_c)
                                if idx_c[-1] != len(tt)-1:
                                    idx_c = np.r_[idx_c, len(tt)-1]
                                tt = tt[idx_c]; vv = vv[idx_c]
                            except Exception:
                                tt = trace['t']; vv = trace['v']
                            classical = validate_full_field_centroid(self.runner.trap, tt, vv, trace['initial'], self.runner.target if self.runner.task['protocol']=='one_way' else self.runner.start, return_trajectory=True, trajectory_decimate=150)
                            best.metrics['classical_trajectory'] = classical.get('trajectory')
                        except Exception as e_cls:
                            best.metrics['classical_trajectory_error'] = str(e_cls)
                    if hall:
                        hall[0]['trajectory'] = best.metrics.get('trajectory')
                        if 'classical_trajectory' in best.metrics:
                            hall[0]['classical_trajectory'] = best.metrics['classical_trajectory']
                trajectory = best.metrics.get('trajectory')
                classical_traj = best.metrics.get('classical_trajectory')
                if trajectory is not None:
                    pass
            except Exception as e:
                trajectory = {'error': str(e)}
        gen_hist=[]
        for isl in self.islands:
            for h in isl.history:
                entry=dict(island=isl.id)
                entry.update(h)
                gen_hist.append(entry)
        traj_bundle = None
        if hall and hall[0].get('trajectory') is not None:
            traj = hall[0]['trajectory']
            classical = hall[0].get('classical_trajectory')
            try:
                start = self.runner.start.tolist()
                target = self.runner.target.tolist()
                direction = self.runner.direction.tolist()
            except Exception:
                start = target = direction = None
            traj_bundle = dict(
                hall_index=0,
                island=hall[0].get('island'),
                fitness=hall[0].get('fitness'),
                trajectory=traj,
                classical_trajectory=classical,
                start_m=start,
                target_m=target,
                direction=direction,
                distance_m=float(self.runner.task.get('distance_m', 0)),
                duration_s=float(self.runner.task.get('duration_s', 0)),
            )
        elif trajectory is not None and isinstance(trajectory, dict) and 'times_s' in trajectory:
            traj_bundle = dict(trajectory=trajectory, hall_index=0)

        # --- trap tab data: per-island top 3% trajectories + trap geometry ---
        try:
            if not heavy and self._cached_island_trajs is not None:
                island_trajectories = self._cached_island_trajs
            else:
                island_trajectories = self._get_island_top_trajectories(heavy=heavy)
        except Exception as e:
            island_trajectories = [{"error": str(e)[:500]}]
        # trap geometry cached at init; include minimal refresh if runner changed
        trap_geom = getattr(self, '_trap_geometry', None)
        # update loss history for loss-tab timeline
        try:
            if loss_breakdown:
                # loss_breakdown[0] is {island, fitness, breakdown, theta}
                _first = loss_breakdown[0]
                _bd = _first.get('breakdown', _first) if isinstance(_first, dict) else _first
                self._loss_history.append(dict(gen=self.global_gen, breakdown=dict(_bd), fitness=float(hall[0]['fitness']) if hall and hall[0].get('fitness') is not None else None))
                if len(self._loss_history) > 250:
                    self._loss_history = self._loss_history[-250:]
        except Exception:
            pass

        robustness = None
        try:
            if not heavy:
                should_recompute = False
            else:
                # skip gen0 for speed — robustness starts from gen 1, still honest but not blocking initial state
                if self.global_gen == 0:
                    should_recompute = False
                else:
                    should_recompute = (self.global_gen % 5 == 0 or self.global_gen == self.generations)
            if should_recompute:
                t0_rob = time.time()
                hall_rob = []
                for rank, g in enumerate(self.hall_of_fame):
                    if time.time()-t0_rob > 12: break
                    try:
                        baseline = 'sta'
                        for isl in self.islands:
                            if isl.strategy == g.island:
                                baseline = isl.baseline
                                break
                        from ..hardware import Hardware
                        from .fitness import _dispatch_evaluate, _restore_after_W
                        count = int(self.runner.cfg.get('robust_samples', 4))
                        count = min(count, 6)
                        scenarios = self.runner.scenarios(count, self.runner.cfg['uncertainty']['seed'] + 100 + rank)
                        vals = []
                        accepted_vals = []
                        use_exact = (self.global_gen == self.generations)
                        # use high fidelity for final gen, else cheap
                        exact_nodes = self.runner.cfg.get('validation_nodes', 321) if use_exact else None
                        subdivisions = self.runner.cfg.get('validation_subdivisions',4) if use_exact else 1
                        for sc in scenarios:
                            try:
                                # unified dispatcher handles W/K/P correctly and frequency_scale perturbation on top of W profile
                                m, stashed, used = _dispatch_evaluate(self.runner, g, baseline, Hardware(**sc['hardware']), sc['stray_V_m'], sc['frequency_scale'], exact_nodes=exact_nodes, subdivisions=subdivisions, path_samples_per_interval=4)
                                # dispatcher already restores, no extra needed
                                vals.append(float(m.get('n_total', 1e6)))
                                accepted_vals.append(bool(m.get('accepted', False)))
                            except Exception:
                                vals.append(1e6)
                                accepted_vals.append(False)
                        max_n = float(self.runner.task.get('max_n_total', 0.5))
                        # pass_fraction is now fraction of scenarios where accepted (all constraints) true, not just n_total*1.5
                        # keep legacy n_total threshold as additional diagnostic but primary is accepted
                        pass_frac_accepted = sum(1 for a in accepted_vals if a) / len(accepted_vals) if accepted_vals else 0
                        pass_frac_n = sum(1 for v in vals if v <= max_n*1.5) / len(vals) if vals else 0
                        hall_rob.append(dict(rank=rank, island=g.island, baseline=baseline, theta=g.theta.tolist(), mean_n=float(np.mean(vals)) if vals else None, worst_n=float(np.max(vals)) if vals else None, cvar_20=float(np.sort(vals)[-max(1,int(np.ceil(0.2*len(vals))) ):].mean()) if vals else None, pass_fraction=float(pass_frac_accepted), pass_fraction_n=float(pass_frac_n), vals=vals[:8], accepted_vals=accepted_vals[:8], threshold_n=float(max_n), threshold_accepted=True))
                    except Exception as e:
                        hall_rob.append(dict(rank=rank, error=str(e)[:500]))
                island_rob = []
                for isl in self.islands:
                    if time.time()-t0_rob > 8: break
                    try:
                        pop_valid = [g for g in isl.population if g.fitness is not None]
                        if not pop_valid: continue
                        best = min(pop_valid, key=lambda g: g.fitness)
                        baseline = isl.baseline
                        count = 2
                        from ..hardware import Hardware
                        from .fitness import _dispatch_evaluate
                        scenarios = self.runner.scenarios(count, self.runner.cfg['uncertainty']['seed'] + 200 + isl.id)
                        vals=[]
                        accepted_vals=[]
                        use_exact_island = (self.global_gen == self.generations)
                        exact_nodes_island = self.runner.cfg.get('validation_nodes',321) if use_exact_island else None
                        subdiv_island = self.runner.cfg.get('validation_subdivisions',4) if use_exact_island else 1
                        for sc in scenarios:
                            try:
                                m, stashed, used = _dispatch_evaluate(self.runner, best, baseline, Hardware(**sc['hardware']), sc['stray_V_m'], sc['frequency_scale'], exact_nodes=exact_nodes_island, subdivisions=subdiv_island, path_samples_per_interval=4)
                                vals.append(float(m.get('n_total',1e6)))
                                accepted_vals.append(bool(m.get('accepted', False)))
                            except: 
                                vals.append(1e6)
                                accepted_vals.append(False)
                        max_n = float(self.runner.task.get('max_n_total',0.5))
                        pass_frac = sum(1 for a in accepted_vals if a)/len(accepted_vals) if accepted_vals else 0
                        pass_frac_n = sum(1 for v in vals if v <= max_n*1.5)/len(vals) if vals else 0
                        island_rob.append(dict(island=isl.strategy, baseline=baseline, best_fitness=float(best.fitness), mean_n=float(np.mean(vals)), worst_n=float(np.max(vals)), pass_fraction=float(pass_frac), pass_fraction_n=float(pass_frac_n), vals=vals, accepted_vals=accepted_vals))
                    except: pass
                robustness = dict(hall=hall_rob, islands=island_rob, threshold_n=float(self.runner.task.get('max_n_total',0.5)), gen=self.global_gen, timestamp=time.time())
                self._robustness_cache[self.global_gen] = robustness
                self._robustness_cache['latest'] = robustness
            else:
                robustness = self._robustness_cache.get('latest')
        except Exception as e:
            robustness = dict(error=str(e)[:500], gen=self.global_gen)

        state=dict(
            global_generation=self.global_gen,
            total_generations=self.generations,
            elapsed_s=time.time()-self.start_time,
            islands=islands_state,
            hall_of_fame=hall,
            hall_diversity=float(getattr(self,'_hall_diversity',0.0)),
            trajectory=traj_bundle,
            island_trajectories=island_trajectories,
            trap=trap_geom,
            migration_history=self.migration.history,
            migration_stats=getattr(self.migration, 'stats', {}),
            cuda=self.cuda,
            config=self.cfg,
            loss_breakdown=loss_breakdown,
            loss_weights=dict(self.runner.loss_weights or {"n":1.0,"pos":1.0,"well":1.0,"lag":0.0}),
            loss_formula=r"F = w_n·ln(1+n/n0) + w_pos·ln(1+(pos/tol)²) + w_well·ln(1+(well/tol)²) + w_lag·ln(1+(lag/dist)²) + 0.5·ln(1+max(0,rms-0.1)/0.1) + λ·mean(((θ-θc)/scale)²) + 2·ln(1+max(0,f0-fmin)/1e5) + height + heating(2·ln(1+(hr-100)/100)) + robust 0.35·mean+0.65·max",
            loss_history=getattr(self, '_loss_history', []),
            pop_size=int(self.islands[0].pop_size if self.islands else (getattr(self,'_global_pop',None) or 20)),
            total_pop=sum(isl.pop_size for isl in self.islands),
            use_cuda=bool(getattr(self, 'use_cuda', False)),
            robustness=robustness,
            timestamp=time.time(),
            # dashboard contract note for frontend
            dashboard=dict(note="Trap tab polls state.json every 500ms; GA writes tmp->state.json atomically, no blocking. Per-island top 3% trajectories are decimated cheap ROM (exact_nodes=None) cached by theta hash, recomputed every trap_traj_interval gens.", top_percent=float(self.cfg.get('island_top_percent', 0.03) if not isinstance(self.cfg.get('ga'), dict) else self.cfg.get('ga', {}).get('island_top_percent', 0.03)), decimate=int(self.cfg.get('trap_traj_decimate', 80) if not isinstance(self.cfg.get('ga'), dict) else self.cfg.get('ga', {}).get('trap_traj_decimate', 80))),
        )
        try:
            if not heavy:
                should_export = False
            else:
                if self.global_gen == 0:
                    should_export = False
                else:
                    should_export = (self.global_gen % 5 == 0) or (self.global_gen == self.generations) or (self._sion_export_cache is None and self.global_gen >= 1)
            if should_export:
                algorithms = []
                for isl_data in island_trajectories:
                    if not isl_data.get('candidates'): continue
                    best = isl_data['candidates'][0]
                    volt = best.get('voltages')
                    freq = best.get('frequency')
                    algorithms.append(dict(
                        island=isl_data['strategy'],
                        baseline=isl_data['baseline'],
                        rank=int(best.get('rank',0)),
                        fitness=float(best.get('fitness',0)) if best.get('fitness') is not None else None,
                        theta=best.get('theta'),
                        times_s=volt.get('times_s') if volt else None,
                        voltages=volt.get('voltages') if volt else None,
                        electrode_names=volt.get('electrode_names') if volt else list(self.runner.trap.names),
                        frequency_hz=freq.get('frequency_hz') if freq else None,
                        frequency_times_s=freq.get('times_s') if freq else None,
                        trajectory=best.get('trajectory'),
                        classical_trajectory=best.get('classical_trajectory'),
                    ))
                baselines = {}
                for base_name in ['sta','smooth']:
                    try:
                        theta0 = np.zeros(8)
                        _, trace0 = self.runner.evaluate(theta0, baseline=base_name, exact_nodes=None, keep=True, return_trajectory=False)
                        t0 = trace0.get('t'); v0 = trace0.get('v')
                        if t0 is not None and v0 is not None:
                            step = max(1, len(t0)//120)
                            idx = np.arange(0, len(t0), step)
                            if idx[-1] != len(t0)-1:
                                idx = np.r_[idx, len(t0)-1]
                            baselines[base_name] = dict(
                                theta=theta0.tolist(),
                                times_s=t0[idx].tolist(),
                                voltages=v0[idx].tolist(),
                                electrode_names=list(self.runner.trap.names),
                                baseline=base_name,
                            )
                        else:
                            baselines[base_name] = dict(theta=theta0.tolist(), error="no trace")
                    except Exception as e:
                        baselines[base_name] = dict(error=str(e)[:300])
                sion_export = dict(
                    version="sion-export-v1",
                    trap=dict(kind=self.runner.cfg['trap']['kind'], names=list(self.runner.trap.names), indices=self.runner.trap.indices.tolist() if hasattr(self.runner.trap.indices,'tolist') else list(self.runner.trap.indices), height_m=float(self.runner.cfg['trap'].get('minimum_seed_m',[0,0,90e-6])[2] if isinstance(self.runner.cfg['trap'].get('minimum_seed_m'), list) else 90e-6)),
                    hardware=dict(sample_rate_hz=float(self.runner.hw.sample_rate_hz), dac_bits=int(self.runner.hw.dac_bits), dac_min_V=float(self.runner.hw.dac_min_V), dac_max_V=float(self.runner.hw.dac_max_V)),
                    task=dict(distance_m=float(self.runner.task['distance_m']), duration_s=float(self.runner.task['duration_s']), protocol=self.runner.task['protocol']),
                    algorithms=algorithms,
                    baselines=baselines,
                    hall_of_fame=hall[:5],
                    generation=self.global_gen,
                    timestamp=time.time(),
                )
                # attach to state for frontend download
                state['sion_export'] = dict(version=sion_export['version'], algorithms=len(algorithms), baselines=list(baselines.keys()), generation=self.global_gen)
                # write file
                tmp_sion = self.out / "sion_export.json.tmp"
                final_sion = self.out / "sion_export.json"
                tmp_sion.write_text(json.dumps(sion_export, indent=2, default=str))
                tmp_sion.replace(final_sion)
                # per-algorithm CSVs
                for algo in algorithms:
                    try:
                        name = f"algo_{algo['island']}_{algo['baseline']}.csv"
                        path = self.out / name
                        import csv
                        with open(path,'w',newline='') as f:
                            w=csv.writer(f)
                            w.writerow(['time_s']+algo['electrode_names'])
                            for ti, row in zip(algo['times_s'] or [], algo['voltages'] or []):
                                w.writerow([ti]+list(row))
                    except: pass
                # also baselines CSVs
                for bname, bdata in baselines.items():
                    if 'voltages' not in bdata: continue
                    try:
                        path = self.out / f"baseline_{bname}.csv"
                        import csv
                        with open(path,'w',newline='') as f:
                            w=csv.writer(f)
                            w.writerow(['time_s']+bdata['electrode_names'])
                            for ti, row in zip(bdata['times_s'], bdata['voltages']):
                                w.writerow([ti]+list(row))
                    except: pass
                self._sion_export_cache = sion_export
            else:
                if self._sion_export_cache:
                    state['sion_export'] = dict(version=self._sion_export_cache['version'], algorithms=len(self._sion_export_cache.get('algorithms',[])), baselines=list(self._sion_export_cache.get('baselines',{}).keys()), generation=self.global_gen)
        except Exception as e:
            state['sion_export'] = dict(error=str(e)[:400])
        tmp=self.out/"state.json.tmp"
        final=self.out/"state.json"
        try:
            tmp.write_text(json.dumps(state, indent=2, default=str))
            tmp.replace(final)
            (self.out/"history"/f"gen_{self.global_gen:03d}.json").write_text(json.dumps(state, indent=2, default=str))
        except Exception as e:
            print(f"state write fail {e}")

    def _evaluate_exact_hall(self):
        # audit 2026-10-30: validate every hall candidate, mark not_validated if skipped, use full nodes (not 81)
        # Previously: skip if gens<=2, only 2 candidates, 81 nodes. Now: full validation for correctness (speed secondary).
        if self.global_gen % 5 !=0 and self.global_gen != self.generations:
            # intermediate generations: mark not yet validated (dashboard/consumers can see)
            for g in self.hall_of_fame:
                if 'exact' not in g.metrics and 'exact_validated' not in g.metrics:
                    g.metrics['exact_validated'] = False
                    g.metrics['exact_status'] = 'not_validated'
            return
        # validate all hall members with full fidelity (321 nodes, configured subdivisions)
        for g in self.hall_of_fame:
            try:
                baseline='sta'
                try:
                    for isl in self.islands:
                        if isl.strategy == g.island:
                            baseline = isl.baseline
                            break
                except: pass
                exact=evaluate_exact(g, self.runner, baseline=baseline)
                g.metrics['exact'] = exact
                # evaluate_exact now sets exact_validated flag; always overwrite (was not_validated at intermediate gens)
                g.metrics['exact_validated'] = bool(exact.get('exact_validated', True))
                g.metrics['exact_status'] = 'validated' if g.metrics.get('exact_validated') else 'failed'
            except Exception as e:
                try:
                    g.metrics['exact_error'] = str(e)[:300]
                    g.metrics['exact_validated'] = False
                    g.metrics['exact_status'] = 'error'
                except: pass

    def _get_migration_live(self):
        """Snapshot for live ring — no heavy compute, just current isolation/next migration."""
        try:
            cur = int(self.global_gen)
            interval = int(getattr(self.migration, 'interval', 5))
            # find next generation where should_migrate is True
            nxt = None
            for g in range(cur+1, cur+interval*4+6):
                try:
                    if self.migration.should_migrate(g):
                        nxt = g
                        break
                except Exception:
                    continue
            if nxt is None:
                # fallback to next interval
                nxt = ((cur // interval) + 1) * interval if interval else cur+1
            # islands snapshot for ring (light)
            islands = []
            for isl in self.islands:
                try:
                    fits = [gg.fitness for gg in isl.population if gg.fitness is not None]
                    best = float(min(fits)) if fits else None
                    mean = float(sum(fits)/len(fits)) if fits else None
                except Exception:
                    best = mean = None
                islands.append(dict(
                    id=isl.id,
                    strategy=isl.strategy,
                    isolated_until=int(getattr(isl, 'isolated_until', 0)),
                    is_isolated=bool(getattr(isl, 'isolated_until', 0) > cur),
                    isolated_until_raw=getattr(isl, 'isolated_until_raw', getattr(isl, 'isolated_until', 0)),
                    generation=int(getattr(isl, 'generation', cur)),
                    pop_size=int(getattr(isl, 'pop_size', 20)),
                    baseline=getattr(isl, 'baseline', 'sta'),
                    best_fitness=best,
                    mean_fitness=mean,
                ))
            hist = list(getattr(self.migration, 'history', []))[-8:]
            stats = dict(getattr(self.migration, 'stats', {}))
            return dict(
                global_generation=cur,
                total_generations=int(self.generations),
                interval=interval,
                n_migrants=int(getattr(self.migration, 'n_migrants', 2)),
                next_migration_gen=int(nxt) if nxt is not None else None,
                generations_until_next=int(nxt - cur) if nxt is not None else None,
                islands=islands,
                history=hist,
                stats=stats,
                timestamp=time.time(),
            )
        except Exception as e:
            return dict(error=str(e)[:400], global_generation=int(getattr(self, 'global_gen', 0)))

    def _write_live(self, evaluated, total, island_strategy=None, current_fitness=None, force=False):
        # lightweight live update inside generation — не тяжёлый как _write_state (без trap траекторий)
        try:
            frac = evaluated/total if total else 0
            # fractional completed: for gen>0, completed = gen-1 + frac (0..generations)
            # keeps progress monotonic 0->generations and ensures after gen1 done live=1.0 not 2.0
            if self.global_gen > 0:
                live_gen = (self.global_gen - 1 + frac)
                # clamp to [0, generations]
                live_gen = max(0.0, min(float(self.generations), live_gen))
            else:
                live_gen = frac
            # also keep raw for debugging (old behaviour)
            live_gen_raw = self.global_gen + frac
            # keep recent 40
            if not hasattr(self, '_live_recent'):
                self._live_recent=[]
                self._live_best=None
                self._live_last_write=0
            if current_fitness is not None:
                if self._live_best is None or current_fitness < self._live_best:
                    self._live_best=current_fitness
                self._live_recent.append(dict(island=island_strategy, fitness=float(current_fitness), gen=self.global_gen, ts=time.time()))
                if len(self._live_recent)>60:
                    self._live_recent.pop(0)
            # throttle 0.2s — force bypasses for migration events / gen boundaries
            now=time.time()
            if not force and now - getattr(self,'_live_last_write',0) < 0.2 and evaluated % 2 !=0:
                return
            self._live_last_write=now
            live_state=dict(
                global_generation=self.global_gen,
                live_generation=live_gen,
                live_generation_raw=live_gen_raw,
                total_generations=int(self.generations),
                evaluated_in_gen=evaluated,
                total_in_gen=total,
                gen_progress=frac,
                overall_progress=(live_gen/self.generations) if self.generations else 0,
                elapsed_s=time.time()-self.start_time,
                current_island=island_strategy,
                current_fitness=current_fitness,
                live_best=self._live_best,
                recent=self._live_recent[-30:],
                timestamp=time.time(),
                # also include per-island best for live header
                islands_live=[dict(strategy=isl.strategy, best=float(min([g.fitness for g in isl.population if g.fitness is not None], default=1e9)), evaluated=len([g for g in isl.population if g.fitness is not None]), pop=isl.pop_size) for isl in self.islands],
                # live migration ring — updates every 0.2s even intra-gen so ring is live
                migration_live=self._get_migration_live(),
            )
            tmp=self.out/"live.json.tmp"
            final=self.out/"live.json"
            tmp.write_text(json.dumps(live_state, default=str))
            tmp.replace(final)
        except Exception as e:
            pass

    def run(self, verbose=True):
        # live tracking for real-time inside generation + per-generation state
        if not hasattr(self, '_live_recent'):
            self._live_recent=[]
            self._live_best=None
            self._live_last_write=0
        total_in_gen=sum(isl.pop_size for isl in self.islands)
        # initial evaluate with live
        live_evaluated=0
        def _live_cb_init(island, genome):
            nonlocal live_evaluated
            live_evaluated+=1
            self._write_live(live_evaluated, total_in_gen, island_strategy=island.strategy, current_fitness=genome.fitness)
        for isl in self.islands:
            isl.evaluate(self.runner, live_cb=_live_cb_init)
        # reset live for next gen tracking — start next gen at 0, not 100% (was showing 100% immediately at 2/2)
        self._live_recent=[]
        self._write_live(0, total_in_gen, island_strategy="init", current_fitness=self._live_best)
        self._update_hall(); self._write_state()
        for gen in range(1, self.generations+1):
            self.global_gen=gen
            total_in_gen=sum(isl.pop_size for isl in self.islands)
            live_evaluated=0
            self._live_recent=[]
            # live callback per genome inside this generation
            def _live_cb(island, genome):
                nonlocal live_evaluated
                live_evaluated+=1
                self._write_live(live_evaluated, total_in_gen, island_strategy=island.strategy, current_fitness=genome.fitness)
            for isl in self.islands:
                isl.evolve_one_generation(self.runner, live_cb=_live_cb)
                # incremental per-island — light state to keep dashboard responsive without blocking evolution
                try:
                    self._update_hall()
                    self._write_state(heavy=False)
                except Exception as e:
                    print(f"per-island state fail {e}")
                # live уже пишется per-оценка, но добавим и per-остров маркер
                self._write_live(live_evaluated, total_in_gen, island_strategy=isl.strategy, current_fitness=self._live_best)
            # heavy generation-boundary state (cached intervals keep it cheap)
            try:
                self._write_state(heavy=True)
            except Exception as e:
                print(f"gen boundary state fail {e}")
            if self.migration.should_migrate(gen):
                self.migration.migrate(gen)
                # live ring: immediate forced update so UI sees migration instantly
                try:
                    self._write_live(live_evaluated, total_in_gen, island_strategy="migrated", current_fitness=self._live_best, force=True)
                except: pass
                # re-evaluate migrants with live
                live_evaluated=0
                for isl in self.islands:
                    isl.evaluate(self.runner, live_cb=_live_cb)
                try:
                    self._update_hall(); self._write_state(heavy=True)
                except: pass
                try:
                    self._write_live(total_in_gen, total_in_gen, island_strategy="post_migrate", current_fitness=self._live_best, force=True)
                except: pass
            # ensure live shows gen completion — forced for ring sync
            self._write_live(total_in_gen, total_in_gen, island_strategy="gen_done", current_fitness=self._live_best, force=True)
            if gen%5==0:
                self._evaluate_exact_hall()
                self._write_state()
            if verbose:
                best=min(self.hall_of_fame, key=lambda g: g.fitness) if self.hall_of_fame else None
                if best is None:
                    print(f"Gen {gen}/{self.generations} hall empty (all rejected)")
                else:
                    print(f"Gen {gen}/{self.generations} best {best.fitness:.4g} island {best.island} status {best.status} hall {[round(g.fitness,3) for g in self.hall_of_fame[:3]]}")
            self.checkpoint()
        self._evaluate_exact_hall(); self._write_state()
        # final live at 100%
        self._write_live(total_in_gen, total_in_gen, island_strategy="done", current_fitness=self._live_best)
        self.checkpoint(final=True)
        return self.hall_of_fame

    def checkpoint(self, final=False):
        ckpt=self.out/"checkpoint.json" if not final else self.out/"final.json"
        data=dict(
            global_generation=self.global_gen,
            hall_of_fame=[dict(theta=g.theta.tolist(), fitness=g.fitness, island=g.island, metrics=g.metrics) for g in self.hall_of_fame],
            islands=[dict(id=isl.id, strategy=isl.strategy, generation=isl.generation, history=isl.history) for isl in self.islands],
            cuda=self.cuda,
            elapsed_s=time.time()-self.start_time
        )
        tmp=ckpt.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.replace(ckpt)

    def dashboard_info(self):
        return dict(out=str(self.out.resolve()), state=str((self.out/"state.json").resolve()), cuda=self.cuda)