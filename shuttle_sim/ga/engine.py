"""GA Engine — orchestrates islands, migration, hall of fame, checkpoints, dashboard state.

Real-time dashboard writes are async via atomic file replace, never blocks evolution.
"""
import time, json, pathlib, threading, queue
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
            isl=Island(i, strat, pop, rng=self.rng.integers(0,1<<31), baseline=base)
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
                    # используем baseline острова лучшего, а не хардкод sta — иначе всегда STA, даже если выиграл smooth/F
                    best_baseline = 'smooth' if best.island in ('S','H') else 'sta'
                    # если остров F/variable — всё равно sta как база, freq-genes отдельно
                    res_traj = self.runner.evaluate(theta8, baseline=best_baseline, exact_nodes=200, subdivisions=self.runner.cfg.get('subdivisions',4), return_trajectory=True, trajectory_decimate=250)
                    # store in metrics for hall entry
                    best.metrics['trajectory'] = res_traj.get('trajectory')
                    # also try classical centroid trajectory for comparison (more expensive, only if enabled and every 5 gens)
                    classical_enabled = self.cfg.get('trajectory_classical', False) or self.runner.cfg.get('trajectory_classical', False)
                    if classical_enabled and (self.global_gen % 5 == 0 or self.global_gen == self.generations):
                        try:
                            from ..dynamics import validate_full_field_centroid
                            _, trace = self.runner.evaluate(theta8, baseline=best_baseline, exact_nodes=120, subdivisions=4, keep=True)
                            classical = validate_full_field_centroid(self.runner.trap, trace['t'], trace['v'], trace['initial'], self.runner.target if self.runner.task['protocol']=='one_way' else self.runner.start, return_trajectory=True, trajectory_decimate=150)
                            best.metrics['classical_trajectory'] = classical.get('trajectory')
                        except Exception as e_cls:
                            best.metrics['classical_trajectory_error'] = str(e_cls)
                    # update hall entry
                    if hall:
                        hall[0]['trajectory'] = best.metrics.get('trajectory')
                        if 'classical_trajectory' in best.metrics:
                            hall[0]['classical_trajectory'] = best.metrics['classical_trajectory']
                # expose top-level trajectory for dashboard convenience
                trajectory = best.metrics.get('trajectory')
                classical_traj = best.metrics.get('classical_trajectory')
                # also compute nominal vs user route for context
                # we have runner.start, target, direction
                if trajectory is not None:
                    # trajectory already contains well_centers and mean_positions
                    pass
            except Exception as e:
                # never break dashboard
                trajectory = {'error': str(e)}
        # generation hist
        gen_hist=[]
        for isl in self.islands:
            for h in isl.history:
                entry=dict(island=isl.id)
                entry.update(h)
                gen_hist.append(entry)
        # build trajectory bundle for dashboard: includes well, quantum, classical, time, and context
        traj_bundle = None
        if hall and hall[0].get('trajectory') is not None:
            traj = hall[0]['trajectory']
            classical = hall[0].get('classical_trajectory')
            # context: start/target, direction, task
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

        state=dict(
            global_generation=self.global_gen,
            total_generations=self.generations,
            elapsed_s=time.time()-self.start_time,
            islands=islands_state,
            hall_of_fame=hall,
            hall_diversity=float(getattr(self,'_hall_diversity',0.0)),
            trajectory=traj_bundle,
            migration_history=self.migration.history,
            cuda=self.cuda,
            config=self.cfg,
            timestamp=time.time()
        )
        # atomic write
        tmp=self.out/"state.json.tmp"
        final=self.out/"state.json"
        try:
            tmp.write_text(json.dumps(state, indent=2, default=str))
            tmp.replace(final)
            # also per-gen history file
            (self.out/"history"/f"gen_{self.global_gen:03d}.json").write_text(json.dumps(state, indent=2, default=str))
        except Exception as e:
            print(f"state write fail {e}")

    def _evaluate_exact_hall(self):
        # exact validation for hall (slow) — only every 5 gens or final
        if self.global_gen % 5 !=0 and self.global_gen != self.generations:
            return
        for g in self.hall_of_fame[:3]:  # top 3
            exact=evaluate_exact(g, self.runner)
            g.metrics['exact'] = exact

    def run(self, verbose=True):
        # initial eval
        for isl in self.islands:
            isl.evaluate(self.runner)
        self._update_hall(); self._write_state()
        for gen in range(1, self.generations+1):
            self.global_gen=gen
            # evolve each island
            for isl in self.islands:
                isl.evolve_one_generation(self.runner)
            # migration
            if self.migration.should_migrate(gen):
                self.migration.migrate(gen)
                # re-evaluate migrants (they have fitness None)
                for isl in self.islands:
                    isl.evaluate(self.runner)
            self._update_hall()
            self._write_state()
            # exact hall every 5
            if gen%5==0:
                self._evaluate_exact_hall()
                self._write_state()
            if verbose:
                best=min(self.hall_of_fame, key=lambda g: g.fitness) if self.hall_of_fame else None
                if best is None:
                    print(f"Gen {gen}/{self.generations} hall empty (all rejected)")
                else:
                    print(f"Gen {gen}/{self.generations} best {best.fitness:.4g} island {best.island} status {best.status} hall {[round(g.fitness,3) for g in self.hall_of_fame[:3]]}")
            # checkpoint
            self.checkpoint()
        # final exact
        self._evaluate_exact_hall(); self._write_state()
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
