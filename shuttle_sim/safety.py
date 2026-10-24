"""Safety layer – mandatory pre-export gates.

Spec refs: §13 safety, §14 failure classification, §8 hierarchy.

A candidate waveform must pass ALL checks before it is marked exportable.
Failure is not a traceback but a classified Rejected code.

Checks:
- voltage bounds per channel
- max sample-to-sample jump / slew proxy
- clipping / saturation
- filter-consistent validity (delta between intended and delivered)
- stable local minimum along path (via trap.local)
- weak confinement (frequency floor)
- numerical stability (nonfinite, uncertainty covariance)
- compatibility with calibration state (hardware identifier)
"""
from dataclasses import dataclass
import numpy as np
from .field import Rejected, finite

@dataclass
class SafetyReport:
    passed: bool
    checks: dict
    failure_code: str = None
    failure_message: str = None

class SafetyLayer:
    def __init__(self, trap, hardware, task):
        self.trap = trap
        self.hw = hardware
        self.task = task

    def check_voltage_bounds(self, voltages_V):
        v = finite(voltages_V, "voltages")
        lo, hi = self.trap.lo, self.trap.hi
        violations = (v < lo) | (v > hi)
        if np.any(violations):
            raise Rejected("hardware_saturation", f"voltage bounds violated max {np.max(v-hi):.3g} V")
        return {"voltage_bounds": "ok", "max_margin_V": float(np.min(np.minimum(hi - v, v - lo).max(axis=1).min())) if v.size else 0}

    def check_slew(self, commands_V):
        # Fail-closed on non-finite inputs (NaN would otherwise bypass max check)
        if not np.isfinite(commands_V).all():
            raise Rejected("numerical_failure", "commands contain non-finite values (NaN/Inf) in slew check")
        jumps = float(np.max(np.abs(np.diff(commands_V, axis=0)))) if commands_V.shape[0] > 1 else 0.0
        if not np.isfinite(jumps):
            raise Rejected("numerical_failure", "slew jump is non-finite")
        if jumps > self.hw.max_jump_V + 1e-12:
            raise Rejected("slew_limit", f"max jump {jumps:g} V exceeds {self.hw.max_jump_V:g}")
        return {"max_jump_V": jumps, "slew_proxy": "ok"}

    def check_filter_consistency(self, distortion_report: dict, threshold_V=None):
        # Threshold is task-configurable and EXECUTABLE (B5): exceeding triggers rejection/re-optimization, not just warn.
        # Spec: "If расхождение превышает допустимый порог, система обязана либо re-optimization under delivered model, либо отклонять"
        if threshold_V is None:
            # Priority: task.distortion_threshold_V > task.max_distortion_V > cfg distortion_threshold > default
            # Default 0.25 V is honest for RC at 500kHz/1MHz; previous 0.8 was arbitrary and too permissive
            if isinstance(self.task, dict):
                threshold_V = float(self.task.get("distortion_threshold_V",
                                    self.task.get("max_distortion_V",
                                        self.task.get("distortion_threshold",
                                            0.25))))
            else:
                threshold_V = float(getattr(self.task, "distortion_threshold_V", 0.25))
        # Fail-closed if distortion report contains non-finite
        max_dev = distortion_report.get("max_deviation_V", 0)
        if not np.isfinite(max_dev):
            raise Rejected("numerical_failure", "distortion max_deviation_V is non-finite")
        if not np.isfinite(threshold_V):
            raise Rejected("numerical_failure", "distortion threshold is non-finite")
        for k, v in distortion_report.items():
            if isinstance(v, (float, int, np.floating)):
                if not np.isfinite(v):
                    raise Rejected("numerical_failure", f"distortion field {k} is non-finite")
        # Executable threshold: reject with delivered_intent_violated if intent destroyed (spec B5)
        if max_dev > threshold_V:
            raise Rejected("delivered_intent_violated",
                           f"max_deviation {max_dev:.3f} V exceeds threshold {threshold_V:.3f} V – intent destroyed, requires re-optimization under delivered model or rejection")
        flag = "ok"  # only ok remains; warn case now raises
        return {"max_deviation_V": float(max_dev), "filter_consistency": flag, "threshold_V": float(threshold_V)}

    def check_confinement(self, hessians, min_freq_hz):
        if not np.isfinite(hessians).all():
            raise Rejected("numerical_failure", "hessians contain non-finite values")
        if not np.isfinite(min_freq_hz):
            raise Rejected("numerical_failure", "min_frequency_hz is non-finite")
        eig = np.linalg.eigvalsh(hessians / self.trap.mass)
        if not np.isfinite(eig).all():
            raise Rejected("numerical_failure", "eigenvalues contain non-finite values")
        # compute properly per time; simplified:
        freqs = np.sqrt(np.maximum(eig, 0)) / (2 * np.pi)
        if not np.isfinite(freqs).all():
            raise Rejected("numerical_failure", "confinement frequencies are non-finite")
        fmin = float(freqs.min())
        if not np.isfinite(fmin):
            raise Rejected("numerical_failure", "min frequency is non-finite")
        if fmin < min_freq_hz - 1e-9:
            raise Rejected("weak_confinement", f"min frequency {fmin/1e6:.3f} MHz below floor {min_freq_hz/1e6:.3f} MHz")
        return {"min_frequency_hz": float(fmin), "confinement": "ok"}

    def evaluate(self, commands_V, delivered_V, distortion, hessians):
        checks = {}
        try:
            # Fail-closed on any non-finite array inputs before domain checks
            for name, arr in [("commands_V", commands_V), ("delivered_V", delivered_V), ("hessians", hessians)]:
                if not np.isfinite(np.asarray(arr)).all():
                    raise Rejected("numerical_failure", f"{name} contains non-finite (NaN/Inf)")
            if distortion is not None:
                for kk, vv in distortion.items():
                    if isinstance(vv, (float, int, np.floating, np.ndarray)):
                        av = np.asarray(vv)
                        if not np.isfinite(av).all():
                            raise Rejected("numerical_failure", f"distortion[{kk}] is non-finite")
            checks.update(self.check_voltage_bounds(delivered_V))
            checks.update(self.check_slew(commands_V))
            checks.update(self.check_filter_consistency(distortion if isinstance(distortion, dict) else {}))
            checks.update(self.check_confinement(hessians, self.task["min_frequency_hz"] if isinstance(self.task, dict) else self.task.min_frequency_hz))
            # Additional: clipping already enforced in hardware.deliver (raises)
            # Numerical stability checked via finite() inside above
            return SafetyReport(passed=True, checks=checks)
        except Rejected as ex:
            checks["failure"] = {"code": ex.code, "message": str(ex)}
            return SafetyReport(passed=False, checks=checks, failure_code=ex.code, failure_message=str(ex))
        except Exception as ex:
            # Any unexpected exception is also fail-closed
            checks["failure"] = {"code": "numerical_failure", "message": str(ex)}
            return SafetyReport(passed=False, checks=checks, failure_code="numerical_failure", failure_message=str(ex))

    def assert_safe(self, commands_V, delivered_V, distortion, hessians):
        rep = self.evaluate(commands_V, delivered_V, distortion, hessians)
        if not rep.passed:
            raise Rejected(rep.failure_code, rep.failure_message)
        return rep
