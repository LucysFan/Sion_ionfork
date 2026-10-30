"""Genome for shuttling — low-dim control.

Base 8: 2 warp +6 voltage (3 огибающие×2 SVD).
Расширено до 12: +4 voltage-гармоники (env*sin2πs, env*sin4πs) → гибкая
форма ΔV с несколькими провалами не обязательно в середине.
10=2+8 (4 базиса×2), 12=2+10 (5×2, +sin2πs+sin4πs) — все гены активны для напряжения,
что исправляет дефект R6 (10/12 гены игнорировались).
Island V: pure voltage 10 генов (4 базиса×2 ΔV, g8/g9 = env·sin2πs) — крутит напряжение.
Island W: pure frequency 10 генов (3 базиса×2 ΔV + 2 freq) — g8/g9 = a·env + b·env·(2s-1)
  для ω(s)=ω0·(1+a·env+b·env·(2s-1)), клип ±15% via FrequencyProfile variable mode.
  Частотная мода FrequencyProfile остаётся глобальной, но W локально варьирует
  confinement через отдельный остров, как просили: один остров напряжение, один частоту.
Bounds — как в pipeline.optimize.
"""
import numpy as np
from ..field import finite

# Powell bounds
BOUNDS_8 = [(-0.10,0.10),(-0.10,0.10)] + [(-0.4,0.4)]*6
BOUNDS_10 = BOUNDS_8 + [(-0.15,0.15), (-0.15,0.15)]
BOUNDS_12 = BOUNDS_8 + [(-0.15,0.15), (-0.15,0.15), (-0.12,0.12), (-0.12,0.12)]
# 10 = 2 warp +8 voltage (4 базиса×2, +env*sin2πs), 12 = 2+10 (5×2, +sin2πs+sin4πs) — все voltage

ISLAND_GENE_DIMS = {
    "S": 8, "A": 8, "V": 10, "W": 10, "R": 8, "H": 8, "F": 12, "M": 12,
    "smooth":8, "sta":8, "variable":10, "frequency":10, "voltage":10, "robust":8, "hardware":8, "free":12, "multi":12,
    "K": 8, "K2":8, "K3":8, "K5":8, "K10":8, "C":8, "STITCH":8, "KRAT":8,
    "P":8, "P2":8, "P3":8, "P5":8, "P10":8, "PERIODIC":8,
}

def clip_to_bounds(theta, bounds):
    th=np.asarray(theta,float)
    lo=np.array([b[0] for b in bounds]); hi=np.array([b[1] for b in bounds])
    return np.clip(th, lo, hi)

class Genome:
    """Wraps theta vector with island-aware bounds."""
    def __init__(self, theta, island="S"):
        dim = ISLAND_GENE_DIMS.get(island, 8)
        if dim==10: b=BOUNDS_10
        elif dim==12: b=BOUNDS_12
        else: b=BOUNDS_8
        th = finite(theta, "theta")
        if th.shape != (dim,):
            # pad/truncate for cross-island migration
            if len(th) < dim:
                th = np.r_[th, np.zeros(dim-len(th))]
            else:
                th = th[:dim]
        self.theta = clip_to_bounds(th, b)
        self.island = island
        self.dim = dim
        self.bounds = b
        self.fitness = None  # set after evaluate
        self.metrics = {}    # full Runner result + observables
        self.status = "unevaluated"

    @classmethod
    def random(cls, island="S", rng=None):
        rng = np.random.default_rng() if rng is None else rng
        dim = ISLAND_GENE_DIMS.get(island, 8)
        if dim==10: b=BOUNDS_10
        elif dim==12: b=BOUNDS_12
        else: b=BOUNDS_8
        th = np.array([rng.uniform(lo,hi) for lo,hi in b])
        return cls(th, island)

    @classmethod
    def zeros(cls, island="S"):
        dim = ISLAND_GENE_DIMS.get(island, 8)
        return cls(np.zeros(dim), island)

    def copy(self):
        g=Genome(self.theta.copy(), self.island)
        g.fitness=self.fitness
        g.metrics=self.metrics.copy() if isinstance(self.metrics, dict) else {}
        g.status=self.status
        return g

    def distance(self, other):
        # Euclidean in normalized [0,1] space to compare 8 vs 10 fairly
        a = (self.theta - np.array([b[0] for b in self.bounds])) / (np.array([b[1] for b in self.bounds]) - np.array([b[0] for b in self.bounds]))
        # pad other to same dim for distance
        if len(other.theta) != len(self.theta):
            # compare only common prefix
            m=min(len(a), len(other.theta))
            # normalize other similarly
            b2= other.bounds
            other_norm=(other.theta[:m]-np.array([b[0] for b in b2[:m]]))/(np.array([b[1] for b in b2[:m]])-np.array([b[0] for b in b2[:m]]))
            a=a[:m]
            return float(np.linalg.norm(a-other_norm))
        b2=other.bounds
        other_norm=(other.theta - np.array([b[0] for b in b2]))/(np.array([b[1] for b in b2])-np.array([b[0] for b in b2]))
        return float(np.linalg.norm(a-other_norm))

    def mutate(self, sigma=0.06, rng=None):
        rng=np.random.default_rng() if rng is None else rng
        sigmas=np.array([sigma*0.7 if i<2 else sigma for i in range(self.dim)])
        if self.dim>=10:
            sigmas[8:]=sigma*0.5
        delta=rng.normal(0, sigmas)
        th=clip_to_bounds(self.theta+delta, self.bounds)
        child=Genome(th, self.island)
        return child

    def crossover(self, other, rng=None):
        rng=np.random.default_rng() if rng is None else rng
        # uniform crossover with island-aware dim
        dim=min(self.dim, other.dim)
        mask=rng.random(dim) < 0.5
        th=np.where(mask, self.theta[:dim], other.theta[:dim])
        # if dims differ, keep extra genes from self
        if self.dim>dim:
            th=np.r_[th, self.theta[dim:]]
        return Genome(th, self.island)

    def to_theta8(self):
        """Runner ждёт 8 (или теперь до 12). Отдаём весь вектор; pipeline обрежет/паддит сам."""
        return self.theta.copy()

    def to_theta_for_runner(self, runner_dim=8):
        if len(self.theta) >= runner_dim:
            return self.theta[:runner_dim]
        return np.r_[self.theta, np.zeros(runner_dim-len(self.theta))]

    def __repr__(self):
        return f"Genome(island={self.island} dim={self.dim} f={self.fitness} theta={self.theta.round(3).tolist()} status={self.status})"
