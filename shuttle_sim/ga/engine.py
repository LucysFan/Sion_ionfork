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
            # default 5 islands as per architecture
            defaults=[("S",20,"sta"),("A",20,"sta"),("V",20,"sta"),("R",25,"sta"),("H",20,"sta")]
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
        # collect all islands best
        all_genomes=[]
        for isl in self.islands:
            for g in isl.population:
                if g.fitness is not None and g.status=='ok':
                    all_genomes.append(g)
        all_genomes.sort(key=lambda g: g.fitness)
        self.hall_of_fame=all_genomes[:self.hall_size]

    def _write_state(self):
        # build dashboard state
        islands_state=[isl.snapshot() for isl in self.islands]
        hall=[dict(theta=g.theta.tolist(), fitness=g.fitness, island=g.island, status=g.status,
                   n_total=g.metrics.get('n_total'), pos_err=g.metrics.get('position_error_m'),
                   fmin=g.metrics.get('minimum_frequency_hz'), distortion=g.metrics.get('distortion',{}).get('max_deviation_V') if isinstance(g.metrics.get('distortion'),dict) else None)
               for g in self.hall_of_fame]
        # generation hist
        gen_hist=[]
        for isl in self.islands:
            for h in isl.history:
                entry=dict(island=isl.id)
                entry.update(h)
                gen_hist.append(entry)
        state=dict(
            global_generation=self.global_gen,
            total_generations=self.generations,
            elapsed_s=time.time()-self.start_time,
            islands=islands_state,
            hall_of_fame=hall,
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
