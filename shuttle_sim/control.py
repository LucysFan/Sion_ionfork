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
        env = PolynomialEnvelope.envelope(s) if False else 16*s**2*(1-s)**2
        # Actually use 16 s^2 (1-s)^2? pipeline uses P([0,0,16,-32,16]) which is 16s^2 -32s^3 +16s^4
        # Simpler: use same polynomial as pipeline for exact compatibility.
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

    Fixed-frequency mode: axial secular frequency ≈ constant along route
      (voltage inversion holds it, used in baseline).
    Variable-frequency mode: frequency is allowed to dip or peak mid-route,
      e.g. parameterized as omega(s)=omega0 * (1 + a*env(s) + b*env(s)*(2s-1)).
    The solver currently holds axial frequency constant (baseline fixed); this
    class provides the variable mode for future advanced optimization and
    documents the choice explicitly in diagnostics.
    """
    mode: Literal["fixed", "variable"] = "fixed"
    coeff: np.ndarray = None  # shape (2,) for variable mode

    def omega_scaling(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s)
        if self.mode == "fixed":
            return np.ones_like(s)
        if self.coeff is None or len(self.coeff) != 2:
            raise ValueError("variable mode requires 2 coeffs")
        env = 16*s**2*(1-s)**2
        # small modulation ±20% to keep confinement
        return 1 + self.coeff[0]*env + self.coeff[1]*env*(2*s-1)

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

    theta[2:8] – 6 coefficients for 3 temporal envelopes × 2 geometry SVD modes.
    Envelopes: e0=env, e1=env*(2s-1), e2=env*(6s^2-6s+1) – orthogonal-ish, all zero at ends.
    Geometry modes: top-2 SVD modes of route_V- mean(route_V) – captures dominant null-space.

    This is the closed-loop-ready low-dim space (8 dims total, not thousands).
    """
    geometry_modes: np.ndarray  # shape (2, channels)

    def correction(self, s: np.ndarray, theta_tail: np.ndarray) -> np.ndarray:
        theta_tail = finite(theta_tail, "theta_tail")
        if theta_tail.shape != (6,):
            raise ValueError("6 corrective params required")
        # envelopes
        env = 16*s**2*(1-s)**2
        e0 = env
        e1 = env*(2*s-1)
        e2 = env*(6*s**2-6*s+1)
        basis = np.column_stack([e0, e1, e2])  # (Ns,3)
        coeff = theta_tail.reshape(3,2)  # 3 temporal × 2 modes
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
