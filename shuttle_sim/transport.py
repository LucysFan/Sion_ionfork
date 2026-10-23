"""Transport task definition – canonical representation of single-ion shuttling.

Spec refs: §4 Canonical path, §4 Transport task, §11 transport modes.

The TransportTask is a pure data container + validation. It explicitly distinguishes:
- user route (logical start→target vector)
- nominal path of desired trap minimum (via voltage inversion along s)
- realized path after hardware delivery (computed in Runner.rom / dynamics)

SI units everywhere; conversions at interface boundaries only.
"""
from dataclasses import dataclass, field
from typing import Literal, Tuple, List
import numpy as np
from .field import finite

Protocol = Literal["one_way", "round_trip", "repeated"]

@dataclass
class TransportTask:
    """Single-ion transport specification.

    Attributes are SI unless noted.
    """
    start_m: np.ndarray
    target_m: np.ndarray
    direction: np.ndarray  # unit vector, 3D
    distance_m: float
    duration_s: float
    hold_s: float
    protocol: Protocol = "one_way"
    cycles: int = 1
    min_frequency_hz: float = 0.5e6
    max_n_total: float = 0.05
    position_tolerance_m: float = 50e-9
    corridor_halfwidth_m: float = 20e-6
    hold_scan_s: List[float] = field(default_factory=list)
    durations_s: List[float] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: dict, initial_position_m: np.ndarray):
        """Build from pipeline config['task'] and initial trap minimum."""
        direction = finite(cfg["direction"], "direction")
        if direction.shape != (3,) or np.linalg.norm(direction) == 0:
            raise ValueError("nonzero 3D direction required")
        direction = direction / np.linalg.norm(direction)
        distance = float(cfg["distance_m"])
        duration = float(cfg["duration_s"])
        hold = float(cfg["hold_s"])
        finite([distance, duration, hold, cfg["min_frequency_hz"],
                cfg["max_n_total"], cfg["position_tolerance_m"]], "task scales")
        if min(distance, duration) <= 0 or hold < 0 or cfg["min_frequency_hz"] <= 0:
            raise ValueError("invalid task scales")
        if cfg["protocol"] not in ("one_way", "round_trip", "repeated"):
            raise ValueError("unknown protocol")
        cycles = int(cfg.get("cycles", 1))
        if cycles < 1:
            raise ValueError("positive cycles required")
        start = finite(initial_position_m, "start_m")
        target = start + distance * direction
        corridor = float(cfg.get("corridor_halfwidth_m", 20e-6))
        # Support both canonical and alias keys (hold_scan_s at cfg root vs task, durations_scan_s alias)
        hold_scan = list(cfg.get("hold_scan_s", [hold]))
        # alias durations_scan_s used in configs (task holds durations_scan_s), canonical durations_s also supported
        if "durations_scan_s" in cfg:
            durations_src = cfg["durations_scan_s"]
        elif "durations_s" in cfg:
            durations_src = cfg["durations_s"]
        elif "duration_scan_s" in cfg:
            durations_src = cfg["duration_scan_s"]
        else:
            durations_src = [duration]
        durations = list(durations_src)
        return cls(
            start_m=start, target_m=target, direction=direction,
            distance_m=distance, duration_s=duration, hold_s=hold,
            protocol=cfg["protocol"], cycles=cycles,
            min_frequency_hz=float(cfg["min_frequency_hz"]),
            max_n_total=float(cfg["max_n_total"]),
            position_tolerance_m=float(cfg["position_tolerance_m"]),
            corridor_halfwidth_m=corridor,
            hold_scan_s=hold_scan, durations_s=durations,
        )

    def validate(self):
        finite(self.start_m, "start_m")
        finite(self.target_m, "target_m")
        finite(self.direction, "direction")
        if self.start_m.shape != (3,) or self.target_m.shape != (3,):
            raise ValueError("start/target must be 3-vector")
        if abs(np.linalg.norm(self.direction) - 1) > 1e-9:
            raise ValueError("direction must be unit")
        finite([self.distance_m, self.duration_s, self.min_frequency_hz,
                self.corridor_halfwidth_m], "transport-task scales")
        if (self.distance_m <= 0 or self.duration_s <= 0 or self.min_frequency_hz <= 0
                or self.corridor_halfwidth_m <= 0):
            raise ValueError("positive distance/duration/frequency/corridor required")
        if self.protocol not in ("one_way", "round_trip", "repeated"):
            raise ValueError("unknown protocol")
        if self.cycles < 1:
            raise ValueError("positive cycles")

    def summary(self) -> dict:
        return {
            "start_m": self.start_m.tolist(),
            "target_m": self.target_m.tolist(),
            "direction": self.direction.tolist(),
            "distance_m": self.distance_m,
            "duration_s": self.duration_s,
            "hold_s": self.hold_s,
            "protocol": self.protocol,
            "cycles": self.cycles,
            "min_frequency_hz": self.min_frequency_hz,
            "max_n_total": self.max_n_total,
            "position_tolerance_m": self.position_tolerance_m,
            "corridor_halfwidth_m": self.corridor_halfwidth_m,
            "durations_s": self.durations_s,
            "hold_scan_s": self.hold_scan_s,
            "path_convention": "user_route: start + s*direction; nominal well path via voltage inversion; realized path via hardware-delivered ROM",
        }

    def leg_count(self) -> int:
        if self.protocol == "one_way":
            return 1
        return 2 * (self.cycles if self.protocol == "repeated" else 1)
