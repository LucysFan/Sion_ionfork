"""Island GA — one population with its own strategy."""
import numpy as np
from .genome import Genome
from .fitness import evaluate_genome

class Island:
    def __init__(self, island_id, strategy, pop_size=20, rng=None, baseline='sta', isolated_until=0):
        self.id = island_id
        self.strategy = strategy  # name S/A/V/R/H/Q...
        self.pop_size = pop_size
        self.baseline = baseline
        self.isolated_until = int(isolated_until)
        self.rng = np.random.default_rng(rng)
        self.population = []
        self.generation = 0
        self.history = []  # list of {gen, best, mean, worst, diversity}
        # strategy params
        self.params = self._strategy_params(strategy)

    def _strategy_params(self, s):
        base = dict(mutation_sigma=0.06, cx_rate=0.7, tour_k=3, elite=2)
        if s in ('S','smooth'):
            return dict(mutation_sigma=0.04, cx_rate=0.7, tour_k=3, elite=2, desc="smooth explorer, low sigma (не STA)")
        if s in ('A','sta'):
            return dict(mutation_sigma=0.06, cx_rate=0.9, tour_k=5, elite=2, desc="STA explorer, high cx")
        if s in ('V','variable'):
            return dict(mutation_sigma=0.08, cx_rate=0.9, tour_k=3, elite=2, desc="variable ΔV 10 генов, 4 базиса×2 (env*sin2πs), гибкая форма не в середине")
        if s in ('R','robust'):
            return dict(mutation_sigma=0.06, cx_rate=0.7, tour_k=5, elite=3, desc="robust, larger elite")
        if s in ('H','hardware'):
            return dict(mutation_sigma=0.05, cx_rate=0.7, tour_k=3, elite=2, desc="hardware penalty lowpass")
        if s in ('F','free'):
            return dict(mutation_sigma=0.07, cx_rate=0.8, tour_k=3, elite=2, desc="free-form 12 генов ΔV (5 базисов×2 + sin4πs), максимум гибкости")
        return base

    def initialize(self, seed_genome=None):
        self.population=[]
        # keep one zero genome as safety
        self.population.append(Genome.zeros(self.strategy))
        # one from seed if provided
        if seed_genome is not None:
            try:
                self.population.append(Genome(seed_genome.theta, self.strategy))
            except: pass
        while len(self.population)<self.pop_size:
            self.population.append(Genome.random(self.strategy, self.rng))
        self.generation=0

    def evaluate(self, runner):
        for g in self.population:
            if g.fitness is None:
                evaluate_genome(g, runner, baseline=self.baseline, exact_nodes=None)

    def tournament(self, k=3):
        k_eff = min(int(k), len(self.population))
        # if k > pop, sample with replacement
        replace = k > len(self.population)
        cand=self.rng.choice(self.population, size=k_eff, replace=replace)
        return min(cand, key=lambda g: g.fitness if g.fitness is not None else 1e9)

    def evolve_one_generation(self, runner):
        # evaluate
        self.evaluate(runner)
        # sort (handle None)
        self.population.sort(key=lambda g: g.fitness if g.fitness is not None else 1e9)
        best=self.population[0]
        # stats
        fits=np.array([g.fitness for g in self.population])
        # diversity: mean pairwise distance
        divs=[]
        for i in range(min(10, len(self.population))):
            for j in range(i+1, min(10,len(self.population))):
                try:
                    divs.append(self.population[i].distance(self.population[j]))
                except: pass
        diversity=float(np.mean(divs)) if divs else 0.0
        self.history.append(dict(generation=self.generation, best=float(best.fitness), mean=float(fits.mean()), worst=float(fits.max()), diversity=diversity, best_status=best.status, best_theta=best.theta.tolist(), strategy=self.strategy))
        # elitism
        elite=self.population[:self.params['elite']]
        # breed
        children=[]
        while len(children) < self.pop_size - len(elite):
            p1=self.tournament(self.params['tour_k'])
            p2=self.tournament(self.params['tour_k'])
            if self.rng.random() < self.params['cx_rate']:
                child=p1.crossover(p2, self.rng)
            else:
                child=p1.copy()
            # mutate
            if self.rng.random() < 0.8:  # mutation chance
                child=child.mutate(sigma=self.params['mutation_sigma'], rng=self.rng)
                child.island=self.strategy  # keep island label
            # need re-evaluate
            child.fitness=None
            child.metrics={}
            child.status="unevaluated"
            children.append(child)
        self.population = elite + children
        self.generation+=1
        return best

    def best(self):
        return min(self.population, key=lambda g: g.fitness if g.fitness is not None else 1e9)

    def worst(self):
        return max(self.population, key=lambda g: g.fitness if g.fitness is not None else -1)

    def get_migrants(self, n=2, diverse=True):
        # best + most diverse
        self.population.sort(key=lambda g: g.fitness if g.fitness is not None else 1e9)
        migrants=[self.population[0].copy()]
        if diverse and n>1:
            # farthest from best
            best=self.population[0]
            far_sorted=sorted(self.population[1:], key=lambda g: best.distance(g), reverse=True)
            migrants.append(far_sorted[0].copy())
            # if need more, random
            while len(migrants)<n:
                migrants.append(self.rng.choice(self.population).copy())
        else:
            migrants.extend([p.copy() for p in self.population[1:n]])
        for m in migrants:
            m.island=self.strategy
        return migrants[:n]

    def replace_worst(self, migrants):
        self.population.sort(key=lambda g: g.fitness if g.fitness is not None else 1e9)
        # replace worst n
        for i, m in enumerate(migrants):
            # adapt migrant to this island dim
            adapted=Genome(m.theta, self.strategy)
            adapted.fitness=None  # force re-eval in new island context (freq coeffs etc)
            self.population[-(i+1)] = adapted

    def snapshot(self):
        fits=[g.fitness for g in self.population if g.fitness is not None]
        isolated = self.isolated_until > self.generation
        return dict(id=self.id, strategy=self.strategy, generation=self.generation,
                    pop_size=self.pop_size, best_fitness=float(min(fits)) if fits else None,
                    mean_fitness=float(np.mean(fits)) if fits else None,
                    diversity=self.history[-1]['diversity'] if self.history else 0,
                    isolated_until=self.isolated_until,
                    is_isolated=isolated,
                    params=self.params)
