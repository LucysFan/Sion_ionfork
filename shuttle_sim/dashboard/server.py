"""Live dashboard server — serves static UI + JSON API for setup, trap preview, and GA/pipeline runs.

Run: python -m shuttle_sim.dashboard.server --out output/ga_run --port 8000
Preview in E2B: https://8000-<sandboxId>.e2b.app
"""
import argparse
import base64
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# --- fix for `python shuttle_sim/dashboard/server.py` on Windows/macOS/Linux ---
# When run as a script, sys.path[0] is .../shuttle_sim/dashboard, not the repo root,
# so `import tools` and `import shuttle_sim` fail. Insert repo root automatically.
try:
    _ROOT = pathlib.Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
except Exception:
    pass

# Global job tracker for unified dashboard
_JOBS = {}
_JOBS_lock = threading.Lock()


def _resolve_gds_path(gds_path_str, cfg_dir=None):
    """Resolve a GDS path safely against repo root, config dir, configs/, uploads/, examples/gds/."""
    if not gds_path_str:
        return None
    raw = pathlib.Path(str(gds_path_str).strip())
    root = pathlib.Path.cwd().resolve()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(root / raw)
        if cfg_dir is not None:
            candidates.append(pathlib.Path(cfg_dir).resolve() / raw)
        candidates.append(root / "configs" / raw.name)
        candidates.append(root / "configs" / "uploads" / raw.name)
        candidates.append(root / "examples" / "gds" / raw.name)
    for cand in candidates:
        try:
            resolved = cand.resolve()
            if resolved.is_file():
                try:
                    return resolved.relative_to(root)
                except ValueError:
                    return resolved
        except Exception:
            continue
    return raw


def _collect_gds_files():
    """Return sorted relative paths of all available .gds / .GDS files in workspace."""
    root = pathlib.Path.cwd()
    seen = set()
    items = []
    for folder in (
        root / "configs" / "uploads",
        root / "configs",
        root / "examples" / "gds",
    ):
        if not folder.exists():
            continue
        for pat in ("*.gds", "*.GDS"):
            for p in sorted(folder.glob(pat)):
                rel = str(p.relative_to(root)) if p.is_relative_to(root) else str(p)
                if rel not in seen:
                    seen.add(rel)
                    items.append(rel)
    return items


def _polygon_area_and_centroid(pts):
    """Signed shoelace area and geometric area centroid for a 2D simple polygon."""
    n = len(pts)
    if n < 3:
        xs = [p[0] for p in pts] or [0.0]
        ys = [p[1] for p in pts] or [0.0]
        return 0.0, [sum(xs) / len(xs), sum(ys) / len(ys)]
    area2 = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        area2 += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(area2) < 1e-12:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return 0.0, [sum(xs) / n, sum(ys) / n]
    inv = 1.0 / (3.0 * area2)
    return abs(area2) * 0.5, [cx * inv, cy * inv]


def _point_in_poly(px, py, pts):
    """Ray-casting point-in-polygon check."""
    inside = False
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / ((yj - yi) or 1e-18) + xi):
            inside = not inside
        j = i
    return inside


def _min_edge_dist_sq(px, py, pts):
    """Minimum squared Euclidean distance from (px, py) to polygon edges."""
    n = len(pts)
    best = float("inf")
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        if denom <= 1e-18:
            dsq = (px - ax) ** 2 + (py - ay) ** 2
        else:
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
            projx, projy = ax + t * dx, ay + t * dy
            dsq = (px - projx) ** 2 + (py - projy) ** 2
        if dsq < best:
            best = dsq
    return best


def _interior_label_point(pts):
    """Find a guaranteed interior point of any (possibly thin non-convex L/U/T) polygon."""
    area, centroid = _polygon_area_and_centroid(pts)
    n = len(pts)
    if n < 3:
        return [round(centroid[0], 3), round(centroid[1], 3)], area

    candidates = []
    if _point_in_poly(centroid[0], centroid[1], pts):
        candidates.append((centroid[0], centroid[1]))

    # Exact scanline chord midpoints between sorted vertex coordinates:
    # guaranteed strictly inside even ultra-thin L/U/T/meander strips!
    uniq_y = sorted({p[1] for p in pts})
    test_ys = [0.5 * (uniq_y[i] + uniq_y[i + 1]) for i in range(len(uniq_y) - 1)]
    for ym in test_ys:
        xs_cross = []
        for i in range(n):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % n]
            if (y0 <= ym < y1) or (y1 <= ym < y0):
                xs_cross.append(x0 + (ym - y0) * (x1 - x0) / (y1 - y0))
        xs_cross.sort()
        for k in range(0, len(xs_cross) - 1, 2):
            candidates.append((0.5 * (xs_cross[k] + xs_cross[k + 1]), ym))

    uniq_x = sorted({p[0] for p in pts})
    test_xs = [0.5 * (uniq_x[i] + uniq_x[i + 1]) for i in range(len(uniq_x) - 1)]
    for xm in test_xs:
        ys_cross = []
        for i in range(n):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % n]
            if (x0 <= xm < x1) or (x1 <= xm < x0):
                ys_cross.append(y0 + (xm - x0) * (y1 - y0) / (x1 - x0))
        ys_cross.sort()
        for k in range(0, len(ys_cross) - 1, 2):
            candidates.append((xm, 0.5 * (ys_cross[k] + ys_cross[k + 1])))

    best_pt = None
    best_score = -1.0
    for px, py in candidates:
        if _point_in_poly(px, py, pts):
            sc = _min_edge_dist_sq(px, py, pts)
            if sc > best_score:
                best_score = sc
                best_pt = [px, py]

    if best_pt is None:
        return [round(centroid[0], 3), round(centroid[1], 3)], area
    return [round(best_pt[0], 3), round(best_pt[1], 3)], area


def _compute_focus_bbox(all_polys, full_bbox):
    """Compute central active trapping region bounding box inside outer bond-pad border."""
    xmin, xmax = full_bbox["xmin"], full_bbox["xmax"]
    ymin, ymax = full_bbox["ymin"], full_bbox["ymax"]
    span_x = xmax - xmin
    span_y = ymax - ymin
    if span_x <= 0 or span_y <= 0:
        return dict(full_bbox)

    margin_x = span_x * 0.04
    margin_y = span_y * 0.04
    inner_x = []
    inner_y = []
    for poly in all_polys:
        for x, y in poly:
            if (xmin + margin_x) < x < (xmax - margin_x):
                inner_x.append(x)
            if (ymin + margin_y) < y < (ymax - margin_y):
                inner_y.append(y)

    if len(inner_x) >= 4 and len(inner_y) >= 4:
        fx0, fx1 = min(inner_x), max(inner_x)
        fy0, fy1 = min(inner_y), max(inner_y)
        pad_x = max(80.0, (fx1 - fx0) * 0.14)
        pad_y = max(80.0, (fy1 - fy0) * 0.22)
        cand = {
            "xmin": max(xmin, fx0 - pad_x),
            "xmax": min(xmax, fx1 + pad_x),
            "ymin": max(ymin, fy0 - pad_y),
            "ymax": min(ymax, fy1 + pad_y),
        }
        # Only use focus box if significantly more zoomed-in than full chip
        if (cand["xmax"] - cand["xmin"]) < 0.78 * span_x or (cand["ymax"] - cand["ymin"]) < 0.78 * span_y:
            return cand
    return dict(full_bbox)


def build_trap_preview(gds_path=None, cfg=None):
    """Build JSON-serializable trap topology preview in micrometres."""
    cfg = cfg or {}
    trap_cfg = cfg.get("trap", {})
    task_cfg = cfg.get("task", {})

    target_gds = gds_path if gds_path is not None else (
        trap_cfg.get("path") if trap_cfg.get("kind") == "gds" else None
    )

    if target_gds:
        resolved = _resolve_gds_path(target_gds)
        if resolved is None or not pathlib.Path(resolved).exists():
            raise FileNotFoundError(f"GDS file not found: {target_gds}")

        from tools.gds_reader import read_polygons

        cell_filter = trap_cfg.get("cell") or None
        raw_polys = read_polygons(resolved, cell_name=cell_filter)
        if not raw_polys and cell_filter:
            raw_polys = read_polygons(resolved, cell_name=None)
        if not raw_polys:
            raise ValueError(f"No polygons found in GDS file: {resolved}")

        # Check whether config's electrode_map matches the layers present in THIS GDS file
        mapping_entries = trap_cfg.get("electrode_map") or []
        rf_names = set((trap_cfg.get("rf_electrodes") or {}).keys())
        by_spec = {}
        for p in raw_polys:
            by_spec.setdefault((int(p.layer), int(p.datatype)), []).append(p)

        mapped_specs = [(int(m["layer"]), int(m.get("datatype", 0))) for m in mapping_entries if "layer" in m]
        is_mapped = bool(mapped_specs) and all(sp in by_spec for sp in mapped_specs)

        electrodes = []
        flat_polys = []

        if is_mapped:
            for idx, m in enumerate(mapping_entries):
                spec = (int(m["layer"]), int(m.get("datatype", 0)))
                name = str(m["name"])
                is_rf = (name in rf_names) or ("rf" in name.lower())
                typ = "rf" if is_rf else "dc"
                paths = []
                areas = []
                best_lbl = [0.0, 0.0]
                best_area = -1.0
                ex0, ex1 = float("inf"), float("-inf")
                ey0, ey1 = float("inf"), float("-inf")
                for poly in by_spec.get(spec, []):
                    pts = [[round(float(x), 4), round(float(y), 4)] for x, y in poly.xy_um]
                    paths.append(pts)
                    flat_polys.append(pts)
                    lbl, area = _interior_label_point(pts)
                    areas.append(area)
                    if area > best_area:
                        best_area = area
                        best_lbl = lbl
                    for x, y in pts:
                        if x < ex0:
                            ex0 = x
                        if x > ex1:
                            ex1 = x
                        if y < ey0:
                            ey0 = y
                        if y > ey1:
                            ey1 = y
                electrodes.append({
                    "index": idx,
                    "name": name,
                    "type": typ,
                    "layer": spec[0],
                    "datatype": spec[1],
                    "paths": paths,
                    "label_pt": best_lbl,
                    "area_um2": round(sum(areas), 2),
                    "bbox": {
                        "xmin": round(ex0, 3),
                        "xmax": round(ex1, 3),
                        "ymin": round(ey0, 3),
                        "ymax": round(ey1, 3),
                    },
                })
        else:
            for idx, poly in enumerate(raw_polys):
                pts = [[round(float(x), 4), round(float(y), 4)] for x, y in poly.xy_um]
                flat_polys.append(pts)
                lbl, area = _interior_label_point(pts)
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                electrodes.append({
                    "index": idx,
                    "name": f"[{idx}]",
                    "type": "polygon",
                    "layer": int(poly.layer),
                    "datatype": int(poly.datatype),
                    "cell": poly.cell,
                    "paths": [pts],
                    "label_pt": lbl,
                    "area_um2": round(area, 2),
                    "bbox": {
                        "xmin": round(min(xs), 3),
                        "xmax": round(max(xs), 3),
                        "ymin": round(min(ys), 3),
                        "ymax": round(max(ys), 3),
                    },
                })

        all_x = [pt[0] for poly in flat_polys for pt in poly]
        all_y = [pt[1] for poly in flat_polys for pt in poly]
        bbox = {
            "xmin": round(min(all_x), 3),
            "xmax": round(max(all_x), 3),
            "ymin": round(min(all_y), 3),
            "ymax": round(max(all_y), 3),
        }
        focus_bbox = _compute_focus_bbox(flat_polys, bbox)
        layers_list = sorted({f"{p.layer}/{p.datatype}" for p in raw_polys})
        cells_list = sorted({p.cell for p in raw_polys if p.cell})
        seed_m = trap_cfg.get("minimum_seed_m", [0.0, 0.0, 90e-6])
        start_m = seed_m if is_mapped and isinstance(seed_m, list) and len(seed_m) == 3 else [0.0, 0.0, 90e-6]
        dist_m = float(task_cfg.get("distance_m", 30e-6))
        direction = task_cfg.get("direction", [1.0, 0.0, 0.0])

        return {
            "kind": "gds",
            "filename": pathlib.Path(resolved).name,
            "path": str(resolved),
            "mapped": is_mapped,
            "polygon_count": len(raw_polys),
            "electrode_count": len(electrodes),
            "layers": layers_list,
            "cells": cells_list,
            "bbox": bbox,
            "focus_bbox": focus_bbox,
            "electrodes": electrodes,
            "start_m": start_m,
            "distance_m": dist_m,
            "direction": direction,
            "height_m": float(start_m[2]) if len(start_m) >= 3 else 90e-6,
            "rf_frequency_hz": trap_cfg.get("rf_frequency_hz"),
        }

    # Analytic fallback (e.g. sion_five_wire)
    if trap_cfg.get("kind") == "sion_five_wire":
        from shuttle_sim.field import load_trap

        adapter = load_trap(trap_cfg)
        system = adapter.system
        electrodes = []
        flat_polys = []
        for idx, ele in enumerate(system):
            rf_val = float(system.rfs[idx]) if idx < len(system.rfs) else 0.0
            typ = "rf" if abs(rf_val) > 1e-12 or "rf" in ele.name.lower() else "dc"
            paths = []
            areas = []
            best_lbl = [0.0, 0.0]
            best_area = -1.0
            ex0, ex1 = float("inf"), float("-inf")
            ey0, ey1 = float("inf"), float("-inf")
            for p in getattr(ele, "paths", []):
                pts = [[round(float(v[0]), 4), round(float(v[1]), 4)] for v in p]
                paths.append(pts)
                flat_polys.append(pts)
                lbl, area = _interior_label_point(pts)
                areas.append(area)
                if area > best_area:
                    best_area = area
                    best_lbl = lbl
                for x, y in pts:
                    ex0, ex1 = min(ex0, x), max(ex1, x)
                    ey0, ey1 = min(ey0, y), max(ey1, y)
            electrodes.append({
                "index": idx,
                "name": ele.name,
                "type": typ,
                "layer": idx,
                "datatype": 0,
                "paths": paths,
                "label_pt": best_lbl,
                "area_um2": round(sum(areas), 2),
                "bbox": {"xmin": round(ex0, 3), "xmax": round(ex1, 3), "ymin": round(ey0, 3), "ymax": round(ey1, 3)},
            })
        all_x = [pt[0] for poly in flat_polys for pt in poly] or [0.0]
        all_y = [pt[1] for poly in flat_polys for pt in poly] or [0.0]
        bbox = {"xmin": min(all_x), "xmax": max(all_x), "ymin": min(all_y), "ymax": max(all_y)}
        seed_m = trap_cfg.get("minimum_seed_m", [0.0, 0.0, 90e-6])
        return {
            "kind": "sion_five_wire",
            "filename": "sion_five_wire",
            "path": "",
            "mapped": True,
            "polygon_count": len(flat_polys),
            "electrode_count": len(electrodes),
            "layers": [],
            "cells": [],
            "bbox": bbox,
            "focus_bbox": _compute_focus_bbox(flat_polys, bbox),
            "electrodes": electrodes,
            "start_m": seed_m,
            "distance_m": float(task_cfg.get("distance_m", 30e-6)),
            "direction": task_cfg.get("direction", [1.0, 0.0, 0.0]),
            "height_m": float(seed_m[2]) if isinstance(seed_m, list) and len(seed_m) >= 3 else 90e-6,
            "rf_frequency_hz": trap_cfg.get("rf_frequency_hz"),
            "lower_V": trap_cfg.get("lower_V"),
            "upper_V": trap_cfg.get("upper_V"),
            "dac_min_V": cfg.get("hardware",{}).get("dac_min_V"),
            "dac_max_V": cfg.get("hardware",{}).get("dac_max_V"),
            "rf_peak_V": trap_cfg.get("rf_peak_V"),
        }

    raise ValueError("No trap path or configuration provided")


def _load_active_config(out_dir):
    """Load active config JSON and normalize relative GDS path if present."""
    for cand in (
        pathlib.Path(out_dir) / "config.json",
        pathlib.Path("configs/unified_live.json"),
        pathlib.Path("configs/shuttle_gds.json"),
        pathlib.Path("configs/shuttle_reference.json"),
    ):
        if cand.exists():
            cfg = json.loads(cand.read_text(encoding="utf-8"))
            if cfg.get("trap", {}).get("kind") == "gds" and cfg["trap"].get("path"):
                resolved = _resolve_gds_path(cfg["trap"]["path"], cfg_dir=cand.parent)
                if resolved is not None:
                    cfg["trap"]["path"] = str(resolved)
            return cfg, cand
    return None, None


class Handler(SimpleHTTPRequestHandler):

    def __init__(self, *args, out_dir=None, **kwargs):
        self.out_dir = pathlib.Path(out_dir) if out_dir else pathlib.Path("output/ga_run")
        super().__init__(*args, directory=str(pathlib.Path(__file__).parent / "static"), **kwargs)

    def _send_json(self, obj, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        if self.path.startswith("/api/state"):
            out_dir = self._resolve_out_dir()
            p = out_dir / "state.json"
            if p.exists():
                try:
                    # merge live migration if newer so ring is live even before heavy state
                    state_bytes = p.read_bytes()
                    try:
                        live_path = out_dir / "live.json"
                        if live_path.exists():
                            st_json = json.loads(state_bytes.decode("utf-8"))
                            live_json = json.loads(live_path.read_text(encoding="utf-8"))
                            mig = live_json.get("migration_live")
                            if mig:
                                # only merge if live is at same or newer generation
                                if live_json.get("timestamp", 0) >= st_json.get("timestamp", 0) - 2:
                                    st_json["migration_live"] = mig
                                    # also enrich islands with live isolation (more timely)
                                    # keep islands from state but update is_isolated from live if available
                                    try:
                                        live_islands = {x["id"]: x for x in mig.get("islands", [])}
                                        for isl in st_json.get("islands", []):
                                            li = live_islands.get(isl.get("id"))
                                            if li is not None:
                                                isl["is_isolated"] = li.get("is_isolated", isl.get("is_isolated"))
                                                isl["isolated_until"] = li.get("isolated_until", isl.get("isolated_until"))
                                    except Exception:
                                        pass
                                    state_bytes = json.dumps(st_json, ensure_ascii=False).encode("utf-8")
                    except Exception:
                        pass
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(state_bytes)
                    return
                except Exception:
                    # fallback to raw
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(p.read_bytes())
                    return
            pp = out_dir / "status.json"
            if pp.exists():
                try:
                    st = json.loads(pp.read_text(encoding="utf-8"))
                    self._send_json({
                        "status": "pipeline_done",
                        "pipeline_status": st,
                        "message": "pipeline finished",
                        "global_generation": 0,
                    })
                    return
                except Exception:
                    pass
            self._send_json({"status": "waiting", "message": "waiting"})
            return

        if self.path.startswith("/api/history"):
            out_dir = self._resolve_out_dir()
            hist = [f.name for f in sorted((out_dir / "history").glob("gen_*.json"))] if (out_dir / "history").exists() else []
            self._send_json({"history": hist})
            return

        if self.path.startswith("/api/cuda"):
            out_dir = self._resolve_out_dir()
            p = out_dir / "state.json"
            cuda = {}
            if p.exists():
                try:
                    cuda = json.loads(p.read_text(encoding="utf-8")).get("cuda", {})
                except Exception:
                    pass
            if not cuda:
                try:
                    from shuttle_sim.ga.cuda import cuda_info, get_xp
                    cuda = cuda_info()
                    try:
                        xp = get_xp()
                        cuda["xp"] = getattr(xp, "__name__", str(xp))
                    except Exception:
                        pass
                except Exception as e:
                    cuda = {"error": str(e)}
            self._send_json({"cuda": cuda, "ok": True})
            return

        if self.path.startswith("/api/runs"):
            root = pathlib.Path("output")
            runs = []
            if root.exists():
                for d in sorted(root.iterdir()):
                    if d.is_dir():
                        runs.append({
                            "name": d.name,
                            "path": str(d),
                            "has_state": (d / "state.json").exists(),
                            "has_status": (d / "status.json").exists(),
                            "has_config": (d / "config.json").exists(),
                        })
            self._send_json({"runs": runs, "current": str(self.out_dir)})
            return

        if self.path.startswith("/api/trap_preview"):
            out_dir = self._resolve_out_dir()
            qs = parse_qs(urlparse(self.path).query)
            req_path = qs.get("path", [None])[0]
            cfg, _ = _load_active_config(out_dir)
            try:
                preview = build_trap_preview(gds_path=req_path, cfg=cfg)
                self._send_json({"ok": True, "trap": preview})
            except FileNotFoundError as e:
                self._send_json({"ok": False, "error": str(e)}, 404)
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 400)
            return

        if self.path.startswith("/api/config"):
            out_dir = self._resolve_out_dir()
            cfg, cfg_path = _load_active_config(out_dir)
            if cfg is not None:
                preview = None
                try:
                    preview = build_trap_preview(cfg=cfg)
                except Exception:
                    preview = None
                self._send_json({
                    "config": cfg,
                    "config_path": str(cfg_path),
                    "gds_list": _collect_gds_files(),
                    "out_dir": str(out_dir),
                    "trap_preview": preview,
                })
                return
            self._send_json({"error": "no config found"}, 404)
            return

        if self.path.startswith("/api/pipeline_status"):
            out_dir = self._resolve_out_dir()
            pp = out_dir / "status.json"
            if pp.exists():
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(pp.read_bytes())
            else:
                self._send_json({"status": "no pipeline yet"})
            return

        if self.path.startswith("/api/log"):
            out_dir = self._resolve_out_dir()
            log_path = out_dir / "run.log"
            if log_path.exists():
                try:
                    txt = log_path.read_text(encoding="utf-8", errors="replace")
                    tail = "\n".join(txt.splitlines()[-250:])
                except Exception as e:
                    tail = f"read error: {e}"
            else:
                tail = ""
            self._send_json({"log": tail, "log_path": str(log_path), "exists": log_path.exists()})
            return

        if self.path.startswith("/api/live"):
            out_dir = self._resolve_out_dir()
            p = out_dir / "live.json"
            if p.exists():
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(p.read_bytes())
                    return
                except Exception:
                    pass
            self._send_json({"status": "no live yet"})
            return

        if self.path.startswith("/api/sion_export"):
            out_dir = self._resolve_out_dir()
            qs = parse_qs(urlparse(self.path).query)
            # allow ?run=... override
            p = out_dir / "sion_export.json"
            if p.exists():
                # support ?format=csv&island=S etc for direct CSV download
                fmt = qs.get("format", [None])[0]
                if fmt == "csv":
                    island = qs.get("island", [None])[0]
                    baseline = qs.get("baseline", [None])[0]
                    # try to find specific CSV
                    cand = None
                    if island:
                        for suf in [f"algo_{island}_{baseline}.csv" if baseline else f"algo_{island}_*.csv", f"baseline_{island}.csv", f"baseline_{baseline}.csv" if baseline else ""]:
                            import glob
                            matches = list((out_dir).glob(suf)) if "*" not in suf else list((out_dir).glob(f"algo_{island}_*.csv"))
                            if matches:
                                cand = matches[0]
                                break
                    if cand and cand.exists():
                        self.send_response(200)
                        self.send_header("Content-Type", "text/csv; charset=utf-8")
                        self.send_header("Content-Disposition", f'attachment; filename="{cand.name}"')
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.wfile.write(cand.read_bytes())
                        return
                    # fallback to JSON if not found
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(p.read_bytes())
                return
            self._send_json({"error": "no sion_export yet, run GA to generate", "path": str(p)}, 404)
            return

        if self.path.startswith("/api/robustness"):
            out_dir = self._resolve_out_dir()
            p = out_dir / "state.json"
            if p.exists():
                try:
                    j = json.loads(p.read_text(encoding="utf-8"))
                    rob = j.get("robustness")
                    if rob:
                        self._send_json({"robustness": rob, "ok": True})
                        return
                except Exception as e:
                    self._send_json({"error": str(e)}, 500)
                    return
            self._send_json({"error": "no robustness yet"}, 404)
            return

        if self.path.startswith("/api/download"):
            out_dir = self._resolve_out_dir()
            qs = parse_qs(urlparse(self.path).query)
            fname = qs.get("file", [None])[0] or qs.get("path", [None])[0]
            if fname:
                # prevent directory traversal
                fname = pathlib.Path(fname).name
                p = out_dir / fname
                if p.exists() and p.is_file():
                    # guess content type
                    ctype = "application/octet-stream"
                    if p.suffix == ".json": ctype = "application/json; charset=utf-8"
                    elif p.suffix == ".csv": ctype = "text/csv; charset=utf-8"
                    elif p.suffix == ".md": ctype = "text/markdown; charset=utf-8"
                    self.send_response(200)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Disposition", f'attachment; filename="{p.name}"')
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(p.read_bytes())
                    return
            self._send_json({"error": "file not found", "available": [str(x.name) for x in out_dir.glob("*") if x.is_file()]}, 404)
            return

        if self.path.startswith("/api/jobs"):
            running = False
            running_job = None
            running_out = None
            running_start = None
            progress = {}
            with _JOBS_lock:
                jobs = {}
                for k, v in _JOBS.items():
                    alive = v["proc"].poll() is None
                    jobs[k] = {
                        "cmd": v["cmd"],
                        "started": v["started"],
                        "alive": alive,
                        "out": v.get("out", ""),
                        "mode": v["cmd"][2] if len(v["cmd"]) > 2 else "",
                    }
                    if alive:
                        running = True
                        running_job = k
                        running_out = v.get("out")
                        running_start = v.get("started")
                if running and running_out:
                    try:
                        p = pathlib.Path(running_out) / "state.json"
                        cfg_p = pathlib.Path(running_out) / "config.json"
                        tg_cfg = 0
                        if cfg_p.exists():
                            try:
                                cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
                                tg_cfg = int(cfg.get("ga", {}).get("generations", 0) or cfg.get("total_generations", 0) or 0)
                            except Exception:
                                tg_cfg = 0
                        # merge live.json for real-time fractional progress (robust against heavy state stalls)
                        live = None
                        live_path = pathlib.Path(running_out) / "live.json"
                        if live_path.exists():
                            try:
                                live = json.loads(live_path.read_text(encoding="utf-8"))
                            except Exception:
                                live = None
                        if p.exists():
                            st = json.loads(p.read_text(encoding="utf-8"))
                            gg = st.get("global_generation", 0)
                            tg = st.get("total_generations", 0) or st.get("ga", {}).get("generations", 0) or tg_cfg or 12
                            elapsed = st.get("elapsed_s", time.time() - (running_start or time.time()))
                            perc_state = (gg / tg * 100) if tg else 0
                            perc = perc_state
                            best = (st.get("hall_of_fame", [{}])[0].get("fitness") if st.get("hall_of_fame") else None)
                            status = "running"
                            progress_extra = {}
                            if live is not None:
                                try:
                                    lg = live.get("live_generation")
                                    op = live.get("overall_progress")
                                    if lg is not None and op is not None and tg:
                                        live_perc = op * 100
                                        if live_perc > perc_state - 0.5:
                                            perc = live_perc
                                            progress_extra = {
                                                "live_generation": lg,
                                                "live_generation_raw": live.get("live_generation_raw", lg),
                                                "gen_progress": live.get("gen_progress", 0),
                                                "overall_progress": op,
                                                "evaluated_in_gen": live.get("evaluated_in_gen", 0),
                                                "total_in_gen": live.get("total_in_gen", 0),
                                                "current_island": live.get("current_island"),
                                                "migration_live": live.get("migration_live"),
                                            }
                                except Exception:
                                    progress_extra = {}
                            progress = {
                                "global_generation": gg,
                                "total_generations": tg,
                                "percent": perc,
                                "elapsed_s": live.get("elapsed_s", elapsed) if live else elapsed,
                                "best": live.get("live_best", best) if live and live.get("live_best") is not None else best,
                                "status": "running",
                            }
                            progress.update(progress_extra)
                        else:
                            elapsed = time.time() - (running_start or time.time()) if running_start else 0
                            # prefer live if available even when state.json missing (early gen)
                            if live is not None and live.get("overall_progress") is not None:
                                tg = tg_cfg or live.get("total_generations", 12) or 12
                                lg = live.get("live_generation", 0)
                                op = live.get("overall_progress", 0)
                                progress = {
                                    "global_generation": int(lg) if lg is not None else 0,
                                    "live_generation": lg,
                                    "total_generations": tg,
                                    "percent": op*100 if op else 0,
                                    "overall_progress": op,
                                    "gen_progress": live.get("gen_progress", 0),
                                    "evaluated_in_gen": live.get("evaluated_in_gen", 0),
                                    "total_in_gen": live.get("total_in_gen", 0),
                                    "elapsed_s": live.get("elapsed_s", elapsed),
                                    "best": live.get("live_best"),
                                    "status": "initializing",
                                    "migration_live": live.get("migration_live"),
                                }
                            else:
                                gg_hist = 0
                                hist_dir = pathlib.Path(running_out) / "history"
                                if hist_dir.exists():
                                    gg_hist = len(list(hist_dir.glob("gen_*.json")))
                                tg = tg_cfg or 12
                                perc = (gg_hist / tg * 100) if tg else 0
                                progress = {
                                    "global_generation": gg_hist,
                                    "total_generations": tg,
                                    "percent": perc,
                                    "elapsed_s": elapsed,
                                    "status": "initializing",
                                }
                    except Exception:
                        progress = {
                            "global_generation": 0,
                            "total_generations": 0,
                            "percent": 0,
                            "elapsed_s": time.time() - (running_start or time.time()) if running_start else 0,
                            "status": "initializing",
                        }
            self._send_json({
                "jobs": jobs,
                "running": running,
                "running_job": running_job,
                "running_out": running_out,
                "progress": progress,
            })
            return

        if self.path == "/" or self.path.startswith("/?"):
            self.path = "/index.html"
        return super().do_GET()

    def _resolve_out_dir(self):
        try:
            qs = parse_qs(urlparse(self.path).query)
            run = qs.get("run", [None])[0]
            if run:
                p = pathlib.Path(run)
                if p.exists() and p.is_dir():
                    return p
        except Exception:
            pass
        return self.out_dir

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        path = self.path.split("?")[0]
        try:
            data = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            data = {}

        if path == "/api/upload":
            fname = data.get("filename", "uploaded.gds")
            fname = re.sub(r"[^a-zA-Z0-9_.-]", "_", pathlib.Path(fname).name)
            if not fname.lower().endswith(".gds"):
                fname += ".gds"
            out_path = pathlib.Path("configs/uploads")
            out_path.mkdir(parents=True, exist_ok=True)
            dest = out_path / fname
            try:
                if "content_base64" in data:
                    raw = base64.b64decode(data["content_base64"])
                    dest.write_bytes(raw)
                elif "content_text" in data:
                    dest.write_text(data["content_text"], encoding="utf-8")
                else:
                    self._send_json({"error": "no content_base64 or content_text"}, 400)
                    return
                cfg, _ = _load_active_config(self.out_dir)
                preview = build_trap_preview(gds_path=dest, cfg=cfg)
                self._send_json({
                    "ok": True,
                    "path": str(dest),
                    "filename": fname,
                    "trap": preview,
                    "gds_list": _collect_gds_files(),
                })
            except Exception as e:
                self._send_json({"error": str(e)}, 400)
            return

        if path == "/api/config":
            try:
                base_cfg, _ = _load_active_config(self.out_dir)
                if base_cfg is None:
                    base_cfg = json.loads(pathlib.Path("configs/shuttle_gds.json").read_text(encoding="utf-8"))
                if "trap" in data and "task" in data:
                    cfg = data
                else:
                    cfg = base_cfg
                    if "distance_m" in data:
                        cfg["task"]["distance_m"] = float(data["distance_m"])
                    if "duration_s" in data:
                        cfg["task"]["duration_s"] = float(data["duration_s"])
                    if "hold_s" in data:
                        cfg["task"]["hold_s"] = float(data["hold_s"])
                    if "protocol" in data:
                        cfg["task"]["protocol"] = str(data["protocol"])
                    if "gds_path" in data or "trap_path" in data:
                        gpath = data.get("gds_path") or data.get("trap_path")
                        if gpath:
                            resolved = _resolve_gds_path(gpath)
                            cfg["trap"]["path"] = str(resolved)
                            cfg["trap"]["kind"] = "gds"
                    if "electrode_map" in data:
                        cfg["trap"]["electrode_map"] = data["electrode_map"]
                    if "rf_electrodes" in data:
                        cfg["trap"]["rf_electrodes"] = data["rf_electrodes"]
                    if "dc_indices" in data:
                        cfg["trap"]["dc_indices"] = data["dc_indices"]
                    if "lower_V" in data:
                        try: cfg["trap"]["lower_V"] = float(data["lower_V"])
                        except: pass
                    if "upper_V" in data:
                        try: cfg["trap"]["upper_V"] = float(data["upper_V"])
                        except: pass
                    if "generations" in data:
                        cfg.setdefault("ga", {})["generations"] = int(data["generations"])
                    if "pop_size" in data or "pop" in data:
                        try:
                            _pop = int(data.get("pop_size", data.get("pop")))
                            if _pop>0:
                                cfg.setdefault("ga", {})["pop_size"] = _pop
                                cfg["ga"]["pop"] = _pop
                                if "islands" in cfg["ga"] and isinstance(cfg["ga"]["islands"], list):
                                    for ic in cfg["ga"]["islands"]:
                                        ic["pop"] = _pop
                        except: pass
                    if "use_cuda" in data:
                        try:
                            cfg.setdefault("ga", {})["use_cuda"] = bool(data["use_cuda"])
                        except: pass
                    if "islands" in data and isinstance(data["islands"], list):
                        cfg.setdefault("ga", {})["islands"] = data["islands"]
                    if "migration_interval" in data:
                        try: cfg.setdefault("ga", {})["migration_interval"] = int(data["migration_interval"])
                        except: pass
                    if "n_migrants" in data:
                        try: cfg.setdefault("ga", {})["n_migrants"] = int(data["n_migrants"])
                        except: pass
                    if "start_m" in data:
                        try:
                            sm = [float(x) for x in data["start_m"]]
                            if len(sm) == 3:
                                cfg["trap"]["minimum_seed_m"] = sm
                        except:
                            pass
                    if "minimum_seed_m" in data:
                        try:
                            sm = [float(x) for x in data["minimum_seed_m"]]
                            if len(sm) == 3:
                                cfg["trap"]["minimum_seed_m"] = sm
                        except:
                            pass
                    if "rf_voltage" in data:
                        try:
                            cfg["trap"]["rf_voltage"] = float(data["rf_voltage"])
                        except:
                            pass
                dest = pathlib.Path("configs/unified_live.json")
                dest.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                preview = None
                try:
                    preview = build_trap_preview(cfg=cfg)
                except Exception:
                    preview = None
                self._send_json({"ok": True, "path": str(dest), "config": cfg, "trap_preview": preview})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
            return

        if path == "/api/stop":
            try:
                target_id = data.get("job_id") if data else None
                stopped = []
                with _JOBS_lock:
                    for jid, info in list(_JOBS.items()):
                        if target_id and jid != target_id:
                            continue
                        proc = info.get("proc")
                        if proc and proc.poll() is None:
                            try:
                                proc.terminate()
                                try:
                                    proc.wait(timeout=5)
                                except subprocess.TimeoutExpired:
                                    proc.kill()
                                    try:
                                        proc.wait(timeout=3)
                                    except Exception:
                                        pass
                                stopped.append(jid)
                            except Exception:
                                pass
                if stopped:
                    self._send_json({"ok": True, "stopped": stopped})
                else:
                    self._send_json({"ok": False, "error": "no running job", "stopped": []}, 404)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
            return

        if path == "/api/launch":
            with _JOBS_lock:
                for jid, info in list(_JOBS.items()):
                    if info.get("proc") and info["proc"].poll() is None:
                        self._send_json({
                            "ok": False,
                            "error": "already running",
                            "running_job": jid,
                            "out": info.get("out"),
                            "cmd": info.get("cmd"),
                        }, 409)
                        return
            try:
                mode = data.get("mode", "ga")
                distance = data.get("distance_m")
                duration = data.get("duration_s")
                gds_path = data.get("gds_path") or data.get("trap_path")
                gens = int(data.get("generations", 8))
                out_dir_str = data.get("out", "output/live_run")
                protocol = data.get("protocol", "one_way")

                cfg, _ = _load_active_config(self.out_dir)
                if cfg is None:
                    cfg = json.loads(pathlib.Path("configs/shuttle_gds.json").read_text(encoding="utf-8"))

                if distance is not None:
                    cfg["task"]["distance_m"] = float(distance)
                if duration is not None:
                    cfg["task"]["duration_s"] = float(duration)
                if gds_path:
                    resolved = _resolve_gds_path(gds_path)
                    cfg["trap"]["path"] = str(resolved)
                    cfg["trap"]["kind"] = "gds"
                elif cfg.get("trap", {}).get("kind") == "gds" and cfg["trap"].get("path"):
                    resolved = _resolve_gds_path(cfg["trap"]["path"])
                    cfg["trap"]["path"] = str(resolved)
                if protocol:
                    cfg["task"]["protocol"] = str(protocol)
                if "start_m" in data:
                    try:
                        sm = [float(x) for x in data["start_m"]]
                        if len(sm) == 3:
                            cfg["trap"]["minimum_seed_m"] = sm
                    except:
                        pass
                if "pop_size" in data or "pop" in data:
                    try:
                        _pop = int(data.get("pop_size", data.get("pop")))
                        if _pop>0:
                            cfg.setdefault("ga", {})["pop_size"]=_pop
                            cfg["ga"]["pop"]=_pop
                            if "islands" in cfg["ga"] and isinstance(cfg["ga"]["islands"], list):
                                for ic in cfg["ga"]["islands"]:
                                    ic["pop"]=_pop
                    except: pass
                if "use_cuda" in data:
                    try: cfg.setdefault("ga", {})["use_cuda"]=bool(data["use_cuda"])
                    except: pass
                if "islands" in data and isinstance(data["islands"], list):
                    cfg.setdefault("ga", {})["islands"]=data["islands"]
                if "migration_interval" in data:
                    try: cfg.setdefault("ga", {})["migration_interval"]=int(data["migration_interval"])
                    except: pass

                out_path = pathlib.Path(out_dir_str)
                out_path.mkdir(parents=True, exist_ok=True)
                # clean stale state from previous run (otherwise dashboard shows old 2/2 100% immediately — stale after break/finish)
                for _p in (out_path / "state.json", out_path / "live.json", out_path / "checkpoint.json", out_path / "final.json"):
                    try:
                        if _p.exists():
                            _p.unlink()
                    except Exception:
                        pass
                if (out_path / "history").exists():
                    for _p in (out_path / "history").glob("gen_*.json"):
                        try:
                            _p.unlink()
                        except Exception:
                            pass
                try:
                    (out_path / "run.log").write_text("", encoding="utf-8")
                except Exception:
                    pass
                try:
                    (out_path / "history").mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass

                cfg_path = out_path / "config.json"
                if mode == "pipeline":
                    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                    cmd = [sys.executable, "-m", "shuttle_sim", "--config", str(cfg_path), "--out", str(out_path)]
                else:
                    cfg.setdefault("ga", {})
                    cfg["ga"]["generations"] = gens
                    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                    cmd = [sys.executable, "-m", "shuttle_sim.ga", "--config", str(cfg_path), "--out", str(out_path), "--generations", str(gens)]

                log_path = out_path / "run.log"
                try:
                    log_f = open(log_path, "w", buffering=1, encoding="utf-8", errors="replace")
                except Exception:
                    log_f = subprocess.PIPE
                env = dict(os.environ)
                # GA benefits from JIT cache (fast per-genome) but first compile is ~48s; use JIT for GA, keep pipeline fast-preview without JIT.
                if mode == "ga":
                    env["NUMBA_DISABLE_JIT"] = "0"
                else:
                    env.setdefault("NUMBA_DISABLE_JIT", "1")
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_f if not isinstance(log_f, int) and log_f != subprocess.PIPE else subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                )
                job_id = f"{mode}_{int(time.time())}"
                with _JOBS_lock:
                    _JOBS[job_id] = {
                        "proc": proc,
                        "cmd": cmd,
                        "started": time.time(),
                        "out": str(out_path),
                        "log": str(log_path),
                    }
                self._send_json({"ok": True, "job_id": job_id, "cmd": cmd, "out": str(out_path), "mode": mode})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
            return

        self._send_json({"error": "unknown POST path"}, 404)

    def log_message(self, format, *args):
        sys.stderr.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), format % args))


def main():
    ap = argparse.ArgumentParser(description="Shuttling live dashboard server")
    ap.add_argument("--out", default="output/ga_run")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    static = pathlib.Path(__file__).parent / "static"
    static.mkdir(parents=True, exist_ok=True)

    def handler(*a, **kw):
        return Handler(*a, out_dir=args.out, **kw)

    server = HTTPServer((args.host, args.port), handler)
    print(f"Dashboard serving {static} and API from {out} at http://{args.host}:{args.port}")
    try:
        from shuttle_sim.ga.cuda import cuda_info
        print("CUDA:", cuda_info())
    except Exception as e:
        print("CUDA info:", e)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
