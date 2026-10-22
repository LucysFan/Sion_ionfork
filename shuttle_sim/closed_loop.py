"""Closed-loop refinement – safe, low-dimensional, budget-limited.

Spec refs: §11 closed-loop refinement, §12 low-dim, §13 safety.

The closed-loop layer:
- accepts nominal or hardware-aware waveform (theta)
- uses an 8-dim corrective space (2 time-warp + 6 voltage basis), NOT thousands of samples
- runs budgeted iterations (e.g., Powell with max evaluations = budget)
- collects measured observables (or simulated stand-in when no hardware)
- recomputes scalar loss via observables pipeline
- applies SafetyLayer before each candidate
- stores full history

When no experimental apparatus is present, the optimizer can run in
"simulation-emulated closed loop" where measurements are emulated by evaluating
with a perturbed hardware model (to demonstrate that the loop reacts to inputs).
This is NOT a fake score – it's a declared simulation of what would happen with
real measurements, with provenance clearly labeled.

Experiment-facing mode would replace `measure_fn(theta) -> scalar loss` with a
call to the real apparatus (blocked_by_missing_external_data if not connected).
"""
from dataclasses import dataclass, field
import numpy as np
import time
from typing import Callable, List, Dict
from scipy.optimize import minimize
from .field import finite, Rejected
from .safety import SafetyLayer
from .observables import processing_chain

@dataclass
class ClosedLoopConfig:
    budget_evaluations: int = 30
    initial_theta: np.ndarray = None
    bounds: List[tuple] = None  # length 8
    optimizer: str = "Powell"
    safety_enabled: bool = True

@dataclass
class ClosedLoopHistoryEntry:
    iteration: int
    theta: List[float]
    loss: float
    observables: dict
    safety_passed: bool
    failure_code: str = None

class ClosedLoopRefiner:
    def __init__(self, runner, safety: SafetyLayer, task: dict):
        """
        runner: Runner instance (provides evaluate())
        safety: SafetyLayer instance
        task: task dict for loss scaling
        """
        self.runner = runner
        self.safety = safety
        self.task = task

    def emulated_measure(self, theta, hardware_perturb=None, stray=None, frequency_scale=1.0):
        """Emulated measurement: evaluate with perturbed model and convert to observables.

        This stands in for a real experiment when hardware is not attached.
        Provenance is explicitly labeled as emulated.
        """
        # Use runner.evaluate but with hardware perturbation if given
        from .hardware import Hardware
        hw = hardware_perturb
        # If no perturbation, use nominal hardware
        result = self.runner.evaluate(theta, baseline="sta", hardware=hw,
                                      stray=stray, frequency_scale=frequency_scale,
                                      exact_nodes=321, subdivisions=4)
        chain = processing_chain(result, self.task)
        return {
            "sim_metrics": result,
            "observables": chain["observables"],
            "loss": chain["loss"],
            "provenance": "emulated_closed_loop: simulated hardware perturbation + observable mapping",
            "hardware_perturb": hardware_perturb,
        }

    def optimize(self, measure_fn: Callable[[np.ndarray], Dict],
                 initial_theta: np.ndarray,
                 bounds=None,
                 budget=30,
                 checkpoint_path=None):
        """Budgeted, safety-gated optimization with safety BEFORE measurement.

        Strict ordering per spec §13: synthesize -> hardware deliver ->
        safety evaluate -> authorize -> measure.  No measurement occurs for
        a safety-rejected candidate.  Budget counts only authorized
        measurements (including the initial point).  If all candidates are
        rejected, no best_theta is chosen (all_failed / blocked state) and
        last-safe fallback is preserved separately.

        measure_fn(theta) must return dict with at least 'loss' and optionally
        'observables', and may raise Rejected for safety violations (treated
        as measurement failure, not safety bypass).

        Returns dict with best theta, history, and metadata.  When no safe
        candidate exists best_theta is None and all_failed=True; the caller
        must use the last-safe theta.
        """
        if bounds is None:
            bounds = [(-0.10, 0.10), (-0.10, 0.10)] + [(-0.4, 0.4)]*6
        finite(initial_theta, "initial_theta")
        if not np.isfinite(initial_theta).all():
            raise ValueError("initial_theta contains non-finite values")
        if len(initial_theta) != 8:
            raise ValueError("closed-loop must stay low-dim (8 params)")
        if not isinstance(budget, int) or budget < 1:
            raise ValueError("budget must be positive integer (counts measurement calls including initial)")
        history: List[ClosedLoopHistoryEntry] = []
        # best tracks only safety-passed, measurement-successful candidates
        best = {"loss": float("inf"), "theta": None}
        last_safe = {"theta": None, "loss": float("inf"), "delivered_sha256": None}
        measurement_count = 0  # number of authorized measure_fn calls so far
        all_failed = False

        def _safety_gate(x):
            """Run synthesize -> hardware -> full safety (including Hessians). Raises Rejected if unsafe."""
            # finite already checked by caller
            if not np.isfinite(x).all():
                raise Rejected("numerical_failure", "theta contains NaN/Inf")
            # 1) synthesize commands
            ct, cmd, route = self.runner.commands(x, baseline="sta")
            if not np.isfinite(cmd).all():
                raise Rejected("numerical_failure", "commands contain non-finite")
            # 2) hardware deliver
            t, delivered, codes, distortion = self.runner.hw.deliver(cmd, subdivisions=4)
            if not np.isfinite(delivered).all():
                raise Rejected("numerical_failure", "delivered voltages contain non-finite")
            # 3) Hessian-based confinement via ROM (fast path, no exact_nodes)
            try:
                # use ROM to get H for full safety confinement check
                _, H, _, _ = self.runner.rom(t, delivered, ct, route, stray=None, exact_nodes=None)
            except Rejected:
                raise
            except Exception as ex:
                raise Rejected("numerical_failure", f"ROM/Hessian failed: {ex}")
            if self.safety is not None:
                rep = self.safety.evaluate(cmd, delivered, distortion, H)
                if not rep.passed:
                    raise Rejected(rep.failure_code or "safety_reject", rep.failure_message or "safety rejected")
                # additionally compute checksum of delivered for last-safe provenance
                import hashlib, json as _json
                delivered_sha = hashlib.sha256(delivered.tobytes()).hexdigest()
                return {
                    "ct": ct, "cmd": cmd, "route": route,
                    "t": t, "delivered": delivered, "codes": codes,
                    "distortion": distortion, "H": H,
                    "safety_report": rep,
                    "delivered_sha256": delivered_sha,
                }
            return {
                "ct": ct, "cmd": cmd, "route": route,
                "t": t, "delivered": delivered, "codes": codes,
                "distortion": distortion, "H": H,
                "delivered_sha256": hashlib.sha256(delivered.tobytes()).hexdigest(),
            }

        def loss_wrapped(x):
            nonlocal measurement_count
            # --- Safety gate BEFORE measurement (fail-closed) ---
            try:
                gate = _safety_gate(x)
                safety_ok = True
                code = None
                safety_checks = gate.get("safety_report", None)
                safety_checks = safety_checks.checks if hasattr(safety_checks, "checks") else {}
            except Rejected as ex:
                loss_val = 1e6
                obs = {"failure": ex.code, "message": str(ex)}
                safety_ok = False
                code = ex.code
                entry = ClosedLoopHistoryEntry(
                    iteration=len(history),
                    theta=x.tolist(),
                    loss=loss_val,
                    observables=obs,
                    safety_passed=False,
                    failure_code=code,
                )
                history.append(entry)
                # safety-rejected does NOT consume budget (no measurement call)
                if checkpoint_path is not None:
                    import json, pathlib
                    pathlib.Path(checkpoint_path).write_text(
                        __import__("json").dumps({"best": {"loss": best["loss"], "theta": best["theta"].tolist() if isinstance(best["theta"], np.ndarray) else best["theta"]},
                                                  "last_safe": last_safe, "history": [e.__dict__ for e in history]}, indent=2, default=str)
                    )
                return loss_val
            except Exception as ex:
                loss_val = 1e6
                obs = {"failure": "numerical_failure", "message": str(ex)}
                safety_ok = False
                code = "numerical_failure"
                entry = ClosedLoopHistoryEntry(
                    iteration=len(history),
                    theta=x.tolist(),
                    loss=loss_val,
                    observables=obs,
                    safety_passed=False,
                    failure_code=code,
                )
                history.append(entry)
                if checkpoint_path is not None:
                    import json, pathlib
                    pathlib.Path(checkpoint_path).write_text(
                        __import__("json").dumps({"best": {"loss": best["loss"], "theta": best["theta"].tolist() if isinstance(best["theta"], np.ndarray) else best["theta"]},
                                                  "last_safe": last_safe, "history": [e.__dict__ for e in history]}, indent=2, default=str)
                    )
                return loss_val

            # --- Budget enforcement: measurement only if budget remains ---
            if measurement_count >= budget:
                # Budget exhausted: do not call hardware, return penalty and mark blocked
                loss_val = 1e6
                obs = {"failure": "budget_exhausted", "message": f"budget {budget} measurements already consumed"}
                entry = ClosedLoopHistoryEntry(
                    iteration=len(history),
                    theta=x.tolist(),
                    loss=loss_val,
                    observables=obs,
                    safety_passed=False,
                    failure_code="budget_exhausted",
                )
                history.append(entry)
                return loss_val

            # --- Authorized measurement ---
            try:
                out = measure_fn(x)
                # measure_fn may still raise Rejected -> measurement failure
                if not isinstance(out, dict) or "loss" not in out:
                    raise ValueError("measure_fn must return dict with 'loss'")
                loss_val = float(out["loss"])
                if not np.isfinite(loss_val):
                    raise ValueError("measure_fn returned non-finite loss")
                obs = out.get("observables", {})
                safety_ok = True
                code = None
                measurement_count += 1
                # update last_safe (most recent safety-passed & measurement-successful)
                if loss_val < last_safe["loss"]:
                    last_safe["theta"] = x.copy()
                    last_safe["loss"] = loss_val
                    last_safe["delivered_sha256"] = gate.get("delivered_sha256")
                # update best (lowest loss among safe & measured)
                if loss_val < best["loss"]:
                    best["loss"] = loss_val
                    best["theta"] = x.copy()
            except Rejected as ex:
                loss_val = 1e6
                obs = {"failure": ex.code, "message": str(ex)}
                safety_ok = False
                code = ex.code
                measurement_count += 1  # measurement was attempted but failed via apparatus rejection
            except Exception as ex:
                loss_val = 1e6
                obs = {"failure": "numerical_failure", "message": str(ex)}
                safety_ok = False
                code = "numerical_failure"
                measurement_count += 1
            entry = ClosedLoopHistoryEntry(
                iteration=len(history),
                theta=x.tolist(),
                loss=loss_val,
                observables=obs if isinstance(obs, dict) else {"loss_proxy": obs},
                safety_passed=safety_ok,
                failure_code=code,
            )
            history.append(entry)
            if checkpoint_path is not None:
                import json, pathlib
                serial_best = best["theta"].tolist() if isinstance(best["theta"], np.ndarray) else best["theta"]
                serial_last = last_safe["theta"].tolist() if isinstance(last_safe["theta"], np.ndarray) else last_safe["theta"]
                pathlib.Path(checkpoint_path).write_text(
                    __import__("json").dumps({"best": {"loss": best["loss"], "theta": serial_best},
                                              "last_safe": {"theta": serial_last, "loss": last_safe["loss"], "sha256": last_safe["delivered_sha256"]},
                                              "measurement_count": measurement_count,
                                              "budget": budget,
                                              "history": [e.__dict__ for e in history]}, indent=2, default=str)
                )
            return loss_val

        # Seed history with initial point – counts toward budget via loss_wrapped
        loss_wrapped(initial_theta)

        # Remaining budget for optimizer (Powell calls loss_wrapped at most remaining times)
        remaining = budget - measurement_count
        # If budget already exhausted by initial point, do not launch optimizer
        if remaining > 0 and best["theta"] is not None:
            # Powell's maxfev counts function evaluations; we set it to remaining
            # to guarantee total measurements <= budget (safety-rejected evals don't consume budget)
            # Use best theta as starting point (last safe); if best is None, optimizer would start from rejected – not allowed
            res = minimize(loss_wrapped, best["theta"], method="Powell", bounds=bounds,
                           options={"maxfev": remaining, "xtol": 1e-3, "ftol": 1e-4})
            success = bool(res.success)
            termination = str(res.message)
        elif remaining > 0 and best["theta"] is None:
            # Initial point was safety-rejected -> no safe starting point; do not optimize from unsafe theta
            success = False
            termination = "blocked: initial theta safety-rejected; no safe starting point within budget"
        else:
            # No remaining budget – only initial point was evaluated
            success = True
            termination = "budget exhausted after initial evaluation"

        # Determine blocked / all_failed state: no safe measurement ever succeeded
        if best["theta"] is None:
            all_failed = True
            best_theta_out = None
            best_loss_out = None
        else:
            best_theta_out = best["theta"].tolist() if isinstance(best["theta"], np.ndarray) else best["theta"]
            best_loss_out = float(best["loss"])

        # Enforce invariant: total authorized measurements never exceeds budget
        assert measurement_count <= budget, f"budget invariant violated: {measurement_count} > {budget}"

        # Serialize safely
        last_safe_serial = last_safe["theta"].tolist() if isinstance(last_safe["theta"], np.ndarray) else last_safe["theta"]
        return {
            "best_theta": best_theta_out,
            "best_loss": best_loss_out,
            "all_failed": bool(all_failed),
            "blocked": bool(all_failed),
            "fallback_theta": [0.0]*8,  # last known safe zero-correction
            "last_safe_theta": last_safe_serial,
            "last_safe_loss": float(last_safe["loss"]) if last_safe["theta"] is not None else None,
            "last_safe_sha256": last_safe["delivered_sha256"],
            "optimizer_success": bool(success),
            "termination": termination,
            "evaluations": len(history),  # total safety-gated attempts (incl safety-rejected)
            "measurements": int(measurement_count),  # authorized measurement calls (must be <= budget)
            "budget": int(budget),
            "history": [e.__dict__ for e in history],
            "bounds": bounds,
            "parameterization": "closed-loop low-dim: 2 time-law + 6 voltage corrections (2 SVD modes)",
            "provenance": "safety-gated (synthesize->hardware->safety->authorize->measure), budget-limited, fail-closed on NaN, no best from rejected",
        }

    def demo_emulated_run(self, initial_theta=None, true_perturb=None, budget=20):
        """Convenience: run emulated closed-loop with a synthetic true perturbation.

        Demonstrates that the loop improves loss under model mismatch.
        """
        if initial_theta is None:
            initial_theta = np.zeros(8)
        # Define true perturbed hardware (unknown to optimizer)
        from .hardware import Hardware
        # Use a gain mismatch if not provided
        if true_perturb is None:
            # 2% gain error
            n = len(self.runner.trap.indices)
            true_perturb = Hardware(
                sample_rate_hz=self.runner.hw.sample_rate_hz,
                dac_bits=self.runner.hw.dac_bits,
                dac_min_V=self.runner.hw.dac_min_V,
                dac_max_V=self.runner.hw.dac_max_V,
                lowpass_hz=self.runner.hw.lowpass_hz,
                gain=[1.02]*n,
                offset_V=[0.01]*n,
                delay_s=[5e-9]*n,
                max_jump_V=self.runner.hw.max_jump_V,
            )
        def measure(theta):
            out = self.emulated_measure(theta, hardware_perturb=true_perturb)
            return {"loss": out["loss"], "observables": out["observables"]}

        return self.optimize(measure, np.asarray(initial_theta, float), budget=budget)
