"""Control synthesis parameterizations.

Spec refs: §5 parameterization, §5 frequency modes, §12 low-dim closed-loop.

Four physically meaningful families are supported; closed-loop must stay low-dim.
The Runner currently implements two baselines (smooth, STA) plus a corrective
basis – this module documents them and adds the missing voltage-knot and
frequency-profile forms so the architecture explicitly shows all four.

STA — лишь частный случай: smooth+STA — два полинома, поверх них 6-мерная
коррекция в R_svd (острова S/A/H) и 10-мерная с variable ω (острова V/F)
генерит уже произвольное семейство не-STA траекторий, отбираемое по n_total.
"""
from dataclasses import dataclass
from typing import Literal
import numpy as np
from numpy.polynomial import Polynomial as P
from .field import finite

Baseline = Literal["smooth", "sta"]

@dataclass
class TrajectoryParameterization:
    """Trajectory-based form: time-law s(t) with polynomial envelope.

    theta[0:2] – time-law warp coefficients on envelope 16s^2(1-s)^2.
    The envelope vanishes with value and slope at leg boundaries, so
    boundary conditions are preserved for any theta.
    """
    theta: np.ndarray  # shape (2,) for pure trajectory; extended to 8 with voltage corrections

    @staticmethod
    def envelope(s: np.ndarray) -> np.ndarray:
        # 16 s^2 (1-s)^2  -> zero at 0,1 with zero derivative
        return 16 * s**2 * (1 - s)**2

    @staticmethod
    def time_warp(s: np.ndarray, coeff: np.ndarray) -> np.ndarray:
        env = PolynomialEnvelope.envelope(s)
        # pipeline uses P([0,0,16,-32,16]) which is 16s^2 -32s^3 +16s^4 == 16 s^2 (1-s)^2, same as envelope
        from numpy.polynomial import Polynomial as Poly
        env_poly = Poly([0,0,16,-32,16])
        base = Poly([0,1])
        shape = base + coeff[0]*env_poly + coeff[1]*env_poly*Poly([-1,2])
        return shape(s)

class PolynomialEnvelope:
    @staticmethod
    def envelope(s):
        return 16*s**2*(1-s)**2

@dataclass
class FrequencyProfile:
    """Confinement-profile parameterization.

    Fixed: ω(s)=ω0
    Variable: ω(s)=ω0*(1+ Σ coeff_k * basis_k(s)), где basis_k — огибающие,
    обнуляющиеся на концах. 2 коэффициента = один центральный провал
    (a*env + b*env*(2s-1), провал можно сдвинуть с середины),
    4 коэффициента = два независимых провала (добавляются
    env*sin(2πs) и env*sin(4πs)), и т.д. — профиль может иметь
    несколько провалов не обязательно в середине.
    """
    mode: Literal["fixed", "variable"] = "fixed"
    coeff: np.ndarray = None  # shape (N,) variable: 2 → один провал, 4 → два, 6 → три

    def _basis(self, s: np.ndarray, k: int) -> np.ndarray:
        s = np.asarray(s, float)
        env = 16*s**2*(1-s)**2
        if k == 0:
            return env
        if k == 1:
            return env*(2*s-1)
        if k == 2:
            return env*np.sin(2*np.pi*s)
        if k == 3:
            return env*np.sin(4*np.pi*s)
        if k == 4:
            return env*np.cos(2*np.pi*s) - env
        if k == 5:
            return env*np.sin(6*np.pi*s)
        return env*np.sin(2*np.pi*(k-1)*s)

    def omega_scaling(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s)
        if self.mode == "fixed":
            return np.ones_like(s)
        if self.coeff is None or len(self.coeff) == 0:
            raise ValueError("variable mode requires >=1 coeff")
        total = np.zeros_like(s, dtype=float)
        for k, c in enumerate(self.coeff):
            total += float(c) * self._basis(s, k)
        return 1 + total

@dataclass
class VoltageKnotParameterization:
    """Direct voltage-knot form: piecewise linear / spline knots at fixed s.

    Provided as a comparable baseline. Knots are placed at route support nodes
    and interpolated with CubicSpline, demonstrating a third parameterization family.
    """
    knots_s: np.ndarray
    knots_V: np.ndarray  # shape (len(s), channels)

    def to_waveform(self, route_s: np.ndarray, voltage_spline=None):
        from scipy.interpolate import CubicSpline
        if self.knots_s is None or self.knots_V is None:
            raise ValueError("knots not defined")
        cs = CubicSpline(self.knots_s, self.knots_V, axis=0, extrapolate=True)
        if voltage_spline is not None:
            # If a nominal spline is given, add knot correction on top
            return voltage_spline(route_s) + cs(route_s)
        return cs(route_s)

    @classmethod
    def from_route(cls, sroute: np.ndarray, route_V: np.ndarray, decimate: int = 2):
        """Create a decimated knot set from the supported route (for comparison)."""
        idx = np.arange(0, len(sroute), decimate)
        return cls(knots_s=sroute[idx], knots_V=route_V[idx])

@dataclass
class CorrectiveBasis:
    """Low-dimensional corrective basis over nominal waveform.

    База 8: theta[2:8] – 6 коэфф. для 3 огибающих ×2 SVD-моды.
    Расширено: если theta_tail длиннее 6, добавляются высшие гармоники
    e3=env*sin(2πs), e4=env*sin(4πs) → профиль ΔV(s) может иметь
    несколько провалов/горбов не только в середине (аналогично ω(s)).
    e0=env, e1=env*(2s-1), e2=env*(6s²-6s+1) — все 0 на краях.

    Геометрия: 2 SVD-моды R_svd (null-space лапласиана).
    8 dim — 3×2, 12 dim — 5×2 (extra 2 гармоники для множественных провалов).
    """
    geometry_modes: np.ndarray  # shape (2, channels)

    def correction(self, s: np.ndarray, theta_tail: np.ndarray) -> np.ndarray:
        theta_tail = finite(theta_tail, "theta_tail")
        n = len(theta_tail)
        if n % 2 != 0:
            raise ValueError("theta_tail must be 2*K (K envelopes ×2 modes)")
        K = n // 2
        if K < 3 or K > 5:
            raise ValueError("K must be 3..5 (6..10 coeffs) for 8..12 dim")
        s = np.asarray(s, float)
        env = 16*s**2*(1-s)**2
        e0 = env
        e1 = env*(2*s-1)
        e2 = env*(6*s**2-6*s+1)
        basis_list = [e0, e1, e2]
        if K >= 4:
            basis_list.append(env*np.sin(2*np.pi*s))
        if K >= 5:
            basis_list.append(env*np.sin(4*np.pi*s))
        basis = np.column_stack(basis_list[:K])
        coeff = theta_tail.reshape(K, 2)
        return (basis @ coeff) @ self.geometry_modes

# Helpers matching pipeline grid logic for documentation
def smooth_poly(shape):
    # 10 s^3 -15 s^4 +6 s^5  (smooth step, zero vel/acc at ends)
    return 10*shape**3 -15*shape**4 + 6*shape**5

def sta_poly(shape, duration_s, omega_rad_s):
    # STA-inspired: s + s''/ (T*omega)^2 ; implemented via polynomial + second derivative
    # full form: P = 35s^4-84s^5+70s^6-20s^7 plus correction
    poly = 35*shape**4 -84*shape**5 +70*shape**6 -20*shape**7
    # second derivative term approximated elsewhere; keep same as pipeline
    return poly
