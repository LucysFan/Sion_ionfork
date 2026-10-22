"""GA Engine — orchestrates islands, migration, hall of fame, checkpoints, dashboard state.

Real-time dashboard writes are async via atomic file replace, never blocks evolution.
Trap tab: separate tab draws trap itself and animates shuttling for each island's top 3% best candidates.
Design is non-blocking: file polling, atomic replace, cheap decimated ROM, interval + theta-hash cache.
"""
import time, json, pathlib, threading, queue, math
import numpy as np
from pathlib import Path

from .genome import Genome
from .island import Island
from .migration import RingMigration
from .fitness import evaluate_exact
from .cuda import cuda_info

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
        self.islands=[]
        for i, icfg in enumerate(config.get('islands', [])):
            strat=icfg.get('strategy','S')
            pop=icfg.get('pop',20)
            base=icfg.get('baseline','sta')
            isolated_until=int(icfg.get('isolated_until', 0))
            isl=Island(i, strat, pop, rng=self.rng.integers(0,1<<31), baseline=base, isolated_until=isolated_until)
            isl.initialize()
            self.islands.append(isl)
        if not self.islands:
            # default 5 islands — разные базы, не только STA (STA «не радует» в эксперименте)
            defaults=[("S",20,"smooth"),("A",20,"sta"),("V",20,"sta"),("R",25,"sta"),("H",20,"smooth")]
            # S=smooth адиабатика, A=STA короткий, V=variable частота, R=robust, H=hardware/lowpass
            for i,(s,p,b) in enumerate(defaults):
                isl=Island(i,s,p, rng=self.rng.integers(0,1<<31), baseline=b)
                isl.initialize()
                self.islands.append(isl)
        self.migration=RingMigration(self.islands, interval=config.get('migration_interval',5), n_migrants=config.get('n_migrants',2))
        self.generations=config.get('generations', 20)
        self.hall_size=config.get('hall_size', 10)
        self.hall_of_fame=[]  # list of Genome sorted
        self.global_gen=0
        self.start_time=time.time()
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
        geom=dict(
            kind=runner.cfg['trap']['kind'],
            electrodes=electrodes,
            bbox=dict(xmin=float(xmin), xmax=float(xmax), ymin=float(ymin), ymax=float(ymax)),
            start_m=start, target_m=target, direction=direction,
            distance_m=float(runner.task.get('distance_m', 0)),
            duration_s=float(runner.task.get('duration_s', 0)),
            hold_s=float(runner.task.get('hold_s', 0)),
            height_m=float(runner.cfg['trap'].get('minimum_seed_m', [0,0,90e-6])[2] if isinstance(runner.cfg['trap'].get('minimum_seed_m'), list) else 90e-6),
            geometry_raw=runner.cfg['trap'].get('geometry', {}),
            length_scale_m=float(runner.cfg['trap'].get('length_scale_m', 1e-6)),
            dc_names=runner.trap.names,
            dc_indices=runner.trap.indices.tolist() if hasattr(runner.trap.indices,'tolist') else list(runner.trap.indices),
            initial_dc_V=runner.initial_V.tolist() if hasattr(runner.initial_V,'tolist') else list(runner.initial_V),
            # for schematic five_wire, also provide parsed wire widths in µm
            shuttle_axis=[1,0,0] if direction is None else direction,
        )
        return geom

    def _get_island_top_trajectories(self):
        """Per-island top 3% best trajectories for Trap tab animation.

        Non-blocking design:
        - cheap ROM (exact_nodes=None) with decimated trajectory (80-120 pts) → ~ms per candidate
        - percent = ceil(pop*0.03) at least 1, capped at 3 per island to bound cost
        - interval cache: recompute only every trap_traj_interval gens or when top thetas change
        - theta-hash cache dict avoids re-evaluating same genome multiple times
        Result size: ~8 islands*1 cand* ~100 pts*3 coords ~ small JSON (<100kB), atomic write never blocks evolve loop.
        """
        percent = float(self.cfg.get('island_top_percent', self.cfg.get('ga',{}).get('island_top_percent', 0.03)) if isinstance(self.cfg.get('ga'), dict) else self.cfg.get('island_top_percent', 0.03))
        # support nested ga config or top-level
        if 'ga' in self.cfg and isinstance(self.cfg['ga'], dict) and 'island_top_percent' in self.cfg['ga']:
            percent = float(self.cfg['ga']['island_top_percent'])
        decimate = int(self.cfg.get('trap_traj_decimate', 80))
        if 'trap_traj_decimate' in self.cfg.get('ga', {}):
            decimate = int(self.cfg['ga']['trap_traj_decimate'])
        # allow override via top-level ga key
        interval = int(self.cfg.get('trap_traj_interval', 2))
        if 'trap_traj_interval' in self.cfg.get('ga', {}):
            interval = int(self.cfg['ga']['trap_traj_interval'])
        # quick cache by gen interval: if not interval gen, return stale but valid cache
        # However if cache is None, we must compute
        if self._cached_island_trajs is not None and self.global_gen !=0 and self.global_gen % interval !=0 and self.global_gen != self.generations:
            # check if top genomes have changed dramatically? We can still return cached to avoid per-gen cost
            # But we need to update island snapshots (best_fitness etc) elsewhere, this cache is trajectory only, not snapshot.
            # Keep returning cached to save compute; top fitness may have improved but traj will refresh on next interval
            # To avoid stale display for more than interval, we already gate.
            return self._cached_island_trajs
        result=[]
        # bound per island to avoid huge state.json
        max_per_island = int(self.cfg.get('trap_traj_max_per_island', 3))
        for isl in self.islands:
            # sort population by fitness
            # include only evaluated; if not evaluated yet, skip
            pop_valid = [g for g in isl.population if g.fitness is not None]
            if not pop_valid:
                # fallback: take whatever
                pop_sorted = sorted(isl.population, key=lambda g: 1e9 if g.fitness is None else g.fitness)
            else:
                pop_sorted = sorted(pop_valid, key=lambda g: g.fitness)
                # if pop smaller than island size due to unevaluated, still handle
            k = max(1, int(math.ceil(len(isl.population)*percent)))
            k = min(k, max_per_island, len(pop_sorted))
            top = pop_sorted[:k]
            cands=[]
            for rank, g in enumerate(top):
                # theta hash for cache (round to 5 decimals)
                try:
                    theta_key = tuple(np.round(g.theta,5).tolist())
                except:
                    theta_key = tuple()
                cache_key = (isl.strategy, isl.baseline, theta_key, decimate)
                cached_traj = self._traj_cache.get(cache_key)
                # reuse if exists and not interval gen requiring fresh physics; but we also want to refresh every 5 gens for safety
                use_cached = cached_traj is not None and (self.global_gen % 5 != 0)
                if use_cached:
                    traj = cached_traj
                else:
                    try:
                        theta = g.theta.copy()
                        baseline = isl.baseline
                        # cheap ROM, return_trajectory with decimation
                        # Note: Runner now supports theta dims 8/10/12
                        res = self.runner.evaluate(theta, baseline=baseline, exact_nodes=None, subdivisions=self.runner.cfg.get('subdivisions',4), return_trajectory=True, trajectory_decimate=decimate)
                        traj = res.get('trajectory')
                        # augment with distortion + n_total for tooltip
                        if isinstance(traj, dict):
                            traj = dict(traj)  # shallow
                            traj['_n_total'] = res.get('n_total')
                            traj['_pos_err'] = res.get('position_error_m')
                            traj['_well_err'] = res.get('well_target_error_m')
                        self._traj_cache[cache_key] = traj
                        # cap cache size to 200 entries LRU-ish
                        if len(self._traj_cache) > 300:
                            # drop oldest 50
                            for kk in list(self._traj_cache.keys())[:50]:
                                del self._traj_cache[kk]
                    except Exception as e:
                        traj = {'error': str(e)[:300]}
                cands.append(dict(rank=rank, theta=g.theta.tolist() if hasattr(g.theta,'tolist') else list(g.theta), fitness=float(g.fitness) if g.fitness is not None else None, status=g.status, baseline=isl.baseline, island=g.island, dim=int(g.dim) if hasattr(g,'dim') else len(g.theta), trajectory=traj))
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

    def _write_state(self):
        # build dashboard state
        islands_state=[isl.snapshot() for isl in self.islands]
        hall=[]
        for g in self.hall_of_fame:
            entry=dict(theta=g.theta.tolist(), fitness=g.fitness, island=g.island, status=g.status,
                   n_total=g.metrics.get('n_total'), pos_err=g.metrics.get('position_error_m'),
                   fmin=g.metrics.get('minimum_frequency_hz'), distortion=g.metrics.get('distortion',{}).get('max_deviation_V') if isinstance(g.metrics.get('distortion'),dict) else None)
            # attach trajectory if already computed (for hall best we compute below)
            if 'trajectory' in g.metrics:
                entry['trajectory'] = g.metrics['trajectory']
            hall.append(entry)
        # compute trajectory for best hall leader for dashboard (well + quantum + classical)
        trajectory = None
        if self.hall_of_fame:
            try:
                best = self.hall_of_fame[0]
                # use cached trajectory if exists and gen hasn't changed much, else recompute
                if 'trajectory' not in best.metrics or self.global_gen % 2 == 0:
                    theta8 = best.to_theta8()
                    best_baseline = 'smooth' if best.island in ('S','H') else 'sta'
                    res_traj = self.runner.evaluate(theta8, baseline=best_baseline, exact_nodes=200, subdivisions=self.runner.cfg.get('subdivisions',4), return_trajectory=True, trajectory_decimate=250)
                    best.metrics['trajectory'] = res_traj.get('trajectory')
                    classical_enabled = self.cfg.get('trajectory_classical', False) or self.runner.cfg.get('trajectory_classical', False)
                    if classical_enabled and (self.global_gen % 5 == 0 or self.global_gen == self.generations):
                        try:
                            from ..dynamics import validate_full_field_centroid
                            _, trace = self.runner.evaluate(theta8, baseline=best_baseline, exact_nodes=120, subdivisions=4, keep=True)
                            classical = validate_full_field_centroid(self.runner.trap, trace['t'], trace['v'], trace['initial'], self.runner.target if self.runner.task['protocol']=='one_way' else self.runner.start, return_trajectory=True, trajectory_decimate=150)
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
        # Must be cheap and cached → never slow training (polling file, no GIL block)
        try:
            island_trajectories = self._get_island_top_trajectories()
        except Exception as e:
            island_trajectories = [{"error": str(e)[:500]}]
        # trap geometry cached at init; include minimal refresh if runner changed
        trap_geom = getattr(self, '_trap_geometry', None)

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
            cuda=self.cuda,
            config=self.cfg,
            timestamp=time.time(),
            # dashboard contract note for frontend
            dashboard=dict(note="Trap tab polls state.json every 500ms; GA writes tmp->state.json atomically, no blocking. Per-island top 3% trajectories are decimated cheap ROM (exact_nodes=None) cached by theta hash, recomputed every trap_traj_interval gens.", top_percent=float(self.cfg.get('island_top_percent', 0.03) if not isinstance(self.cfg.get('ga'), dict) else self.cfg.get('ga', {}).get('island_top_percent', 0.03)), decimate=int(self.cfg.get('trap_traj_decimate', 80) if not isinstance(self.cfg.get('ga'), dict) else self.cfg.get('ga', {}).get('trap_traj_decimate', 80))),
        )
        # atomic write
        tmp=self.out/"state.json.tmp"
        final=self.out/"state.json"
        try:
            tmp.write_text(json.dumps(state, indent=2, default=str))
            tmp.replace(final)
            (self.out/"history"/f"gen_{self.global_gen:03d}.json").write_text(json.dumps(state, indent=2, default=str))
        except Exception as e:
            print(f"state write fail {e}")

    def _evaluate_exact_hall(self):
        if self.global_gen % 5 !=0 and self.global_gen != self.generations:
            return
        for g in self.hall_of_fame[:3]:
            exact=evaluate_exact(g, self.runner)
            g.metrics['exact'] = exact

    def _write_live(self, evaluated, total, island_strategy=None, current_fitness=None):
        # lightweight live update inside generation — не тяжёлый как _write_state (без trap траекторий)
        try:
            frac = evaluated/total if total else 0
            live_gen = self.global_gen + frac
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
            # throttle 0.2s
            now=time.time()
            if now - getattr(self,'_live_last_write',0) < 0.2 and evaluated % 2 !=0:
                return
            self._live_last_write=now
            live_state=dict(
                global_generation=self.global_gen,
                live_generation=live_gen,
                evaluated_in_gen=evaluated,
                total_in_gen=total,
                gen_progress=frac,
                overall_progress=(self.global_gen+frac)/self.generations if self.generations else 0,
                elapsed_s=time.time()-self.start_time,
                current_island=island_strategy,
                current_fitness=current_fitness,
                live_best=self._live_best,
                recent=self._live_recent[-30:],
                timestamp=time.time(),
                # also include per-island best for live header
                islands_live=[dict(strategy=isl.strategy, best=float(min([g.fitness for g in isl.population if g.fitness is not None], default=1e9)), evaluated=len([g for g in isl.population if g.fitness is not None]), pop=isl.pop_size) for isl in self.islands]
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
            if self.migration.should_migrate(gen):
                self.migration.migrate(gen)
                # re-evaluate migrants with live
                live_evaluated=0
                for isl in self.islands:
                    isl.evaluate(self.runner, live_cb=_live_cb)
            self._update_hall()
            self._write_state()
            # ensure live shows gen completion
            self._write_live(total_in_gen, total_in_gen, island_strategy="gen_done", current_fitness=self._live_best)
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
