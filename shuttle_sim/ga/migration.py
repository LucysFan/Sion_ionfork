"""Migration topologies for island GA."""
import numpy as np

class RingMigration:
    """Directed ring S→A→V→R→H→S, every `interval` gens, `n_migrants` per island.
    Поддерживает изолированные острова: если island.isolated_until > generation,
    остров не участвует в миграции (эволюционирует сам), а потом
    в момент де-изоляции делает burst в общую систему.
    """
    def __init__(self, islands, interval=5, n_migrants=2):
        self.islands=islands
        self.interval=interval
        self.n_migrants=n_migrants
        self.history=[]

    def _is_isolated(self, isl, generation):
        return getattr(isl, 'isolated_until', 0) > generation

    def should_migrate(self, generation):
        if generation==0 or generation % self.interval !=0:
            return False
        active = [isl for isl in self.islands if not self._is_isolated(isl, generation)]
        return len(active) >= 2

    def migrate(self, generation):
        deisolated = [isl for isl in self.islands if getattr(isl,'isolated_until',0)==generation]
        if deisolated:
            for src in deisolated:
                migrants = src.get_migrants(self.n_migrants, diverse=True)
                for dst in self.islands:
                    if dst is src or self._is_isolated(dst, generation):
                        continue
                    dst.replace_worst(migrants[:1])
                self.history.append(dict(generation=generation, burst_from=src.id, isolated_until=src.isolated_until))
        active = [isl for isl in self.islands if not self._is_isolated(isl, generation)]
        if len(active) < 2:
            return
        migrants_per_island={}
        for isl in active:
            migrants_per_island[isl.id]=isl.get_migrants(self.n_migrants, diverse=True)
        n=len(active)
        for i, isl in enumerate(active):
            src_id=active[(i-1)%n].id
            incoming=migrants_per_island[src_id]
            isl.replace_worst(incoming)
        self.history.append(dict(generation=generation, migrant_ids=list(migrants_per_island.keys()),
                                 isolated_skipped=[isl.id for isl in self.islands if self._is_isolated(isl,generation)]))
        if generation % (self.interval*3)==0:
            self.global_shuffle(generation)

    def global_shuffle(self, generation=None):
        pool_islands = self.islands
        if generation is not None:
            pool_islands = [isl for isl in self.islands if not self._is_isolated(isl, generation)]
        if len(pool_islands) < 2:
            return
        pool=[isl.rng.choice(isl.population).copy() for isl in pool_islands]
        for isl in pool_islands:
            idx=isl.rng.integers(2, len(isl.population))
            donor=isl.rng.choice(pool).copy()
            adapted=__import__('shuttle_sim.ga.genome', fromlist=['Genome']).Genome(donor.theta, isl.strategy)
            adapted.fitness=None
            isl.population[idx]=adapted

class StarMigration(RingMigration):
    """Star: all islands send to hub (R robust) and hub broadcasts."""
    def migrate(self, generation):
        hub=self.islands[3] if len(self.islands)>3 else self.islands[0]
        # collect from all
        all_migrants=[]
        for isl in self.islands:
            if isl is hub: continue
            all_migrants.extend(isl.get_migrants(1, diverse=False))
        # hub replaces worst with each
        for m in all_migrants:
            hub.replace_worst([m])
        # hub broadcasts its best to all
        bests=hub.get_migrants(1, diverse=False)
        for isl in self.islands:
            if isl is hub: continue
            isl.replace_worst(bests)
        self.history.append(dict(generation=generation, star=True))
