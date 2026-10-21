"""Migration topologies for island GA."""
import numpy as np

class RingMigration:
    """Directed ring S→A→V→R→H→S, every `interval` gens, `n_migrants` per island."""
    def __init__(self, islands, interval=5, n_migrants=2):
        self.islands=islands
        self.interval=interval
        self.n_migrants=n_migrants
        self.history=[]

    def should_migrate(self, generation):
        return generation>0 and generation % self.interval ==0

    def migrate(self, generation):
        # collect migrants first (parallel)
        migrants_per_island={}
        for isl in self.islands:
            migrants_per_island[isl.id]=isl.get_migrants(self.n_migrants, diverse=True)
        # ring: island i sends to (i+1)%N
        n=len(self.islands)
        for i, isl in enumerate(self.islands):
            src_id=self.islands[(i-1)%n].id
            incoming=migrants_per_island[src_id]
            isl.replace_worst(incoming)
        self.history.append(dict(generation=generation, migrant_ids=list(migrants_per_island.keys())))
        # occasional global shuffle every 3 intervals
        if generation % (self.interval*3)==0:
            self.global_shuffle()

    def global_shuffle(self):
        # pick one random genome from each island and broadcast
        import random
        pool=[isl.rng.choice(isl.population).copy() for isl in self.islands]
        for isl in self.islands:
            # replace a random non-elite
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
