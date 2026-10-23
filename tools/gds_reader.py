"""Minimal pure-Python GDSII reader (BOUNDARY, PATH, SREF, AREF).

Used by :mod:`tools.make_docs_assets` and :mod:`shuttle_sim.dashboard.server`
to read real GDSII electrode layouts without requiring binary ``gdspy``.

Returns polygons in **micrometres**, honouring the IBM System/360 floating-point
``UNITS`` record of the file (database unit -> micrometres).  Trailing points
that repeat the first vertex (GDS closing convention) are dropped.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

# GDSII record types
_HEADER = 0x00
_BGNLIB = 0x01
_LIBNAME = 0x02
_UNITS = 0x03
_ENDLIB = 0x04
_BGNSTR = 0x05
_STRNAME = 0x06
_ENDSTR = 0x07
_BOUNDARY = 0x08
_PATH = 0x09
_SREF = 0x0A
_AREF = 0x0B
_LAYER = 0x0D
_DATATYPE = 0x0E
_WIDTH = 0x0F
_XY = 0x10
_ENDEL = 0x11
_SNAME = 0x12
_COLROW = 0x13
_STRANS = 0x1A
_MAG = 0x1B
_ANGLE = 0x1C


def ibm_to_float(raw8: bytes) -> float:
    """Decode an 8-byte GDSII REAL8 (IBM System/360 base-16 float)."""
    if len(raw8) != 8 or raw8 == b"\x00" * 8:
        return 0.0
    sign = -1.0 if (raw8[0] & 0x80) else 1.0
    exp = (raw8[0] & 0x7F) - 64
    mant = int.from_bytes(raw8[1:8], "big") / float(1 << 56)
    return sign * mant * (16.0 ** exp)


@dataclass
class Polygon:
    layer: int
    datatype: int
    cell: str
    xy_um: list[tuple[float, float]]


def _path_to_polygon(pts: list[tuple[float, float]], width: float) -> list[tuple[float, float]]:
    """Expand a GDS PATH centerline with given width into a closed polygon."""
    if len(pts) < 2 or width <= 0:
        return pts if len(pts) >= 3 else []
    half = abs(width) * 0.5
    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    n = len(pts)
    for idx in range(n):
        if idx == 0:
            dx = pts[1][0] - pts[0][0]
            dy = pts[1][1] - pts[0][1]
        elif idx == n - 1:
            dx = pts[-1][0] - pts[-2][0]
            dy = pts[-1][1] - pts[-2][1]
        else:
            dx = pts[idx + 1][0] - pts[idx - 1][0]
            dy = pts[idx + 1][1] - pts[idx - 1][1]
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length * half, dx / length * half
        left.append((pts[idx][0] + nx, pts[idx][1] + ny))
        right.append((pts[idx][0] - nx, pts[idx][1] - ny))
    return left + list(reversed(right))


def _transform_pts(
    pts: list[tuple[float, float]],
    reflect_x: bool,
    mag: float,
    angle_deg: float,
    ox: float,
    oy: float,
) -> list[tuple[float, float]]:
    rad = math.radians(angle_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    out: list[tuple[float, float]] = []
    for x, y in pts:
        if reflect_x:
            y = -y
        x *= mag
        y *= mag
        rx = x * ca - y * sa + ox
        ry = x * sa + y * ca + oy
        out.append((rx, ry))
    return out


def read_polygons(
    path: str | Path,
    scale: float | None = None,
    cell_name: str | None = None,
) -> list[Polygon]:
    """Parse all polygons (BOUNDARY, PATH, and resolved SREF/AREF) of a GDSII file.

    Coordinates are multiplied by ``scale``.  By default the GDSII ``UNITS``
    record is decoded (database unit expressed in micrometres); pass an
    explicit ``scale`` when raw database-unit scaling is desired.
    """
    data = Path(path).read_bytes()
    i = 0
    db_unit_um = 1e-3  # standard 1 nm database unit fallback

    cells_polys: dict[str, list[tuple[int, int, list[tuple[float, float]]]]] = {}
    cells_refs: dict[str, list[dict]] = {}
    referenced: set[str] = set()

    cur_cell: str | None = None
    cur: dict | None = None

    while i + 4 <= len(data):
        ln = struct.unpack(">H", data[i : i + 2])[0]
        if ln < 4:
            break
        rec_type = data[i + 2]
        payload = data[i + 4 : i + ln]
        i += ln

        if rec_type == _UNITS and scale is None and len(payload) >= 16:
            db_m = ibm_to_float(payload[8:16])
            if 1e-15 < db_m < 1.0:
                db_unit_um = db_m * 1e6
        elif rec_type == _BGNSTR:
            cur_cell = ""
            cur = None
        elif rec_type == _STRNAME:
            cur_cell = payload.rstrip(b"\x00").decode("ascii", "ignore")
            cells_polys.setdefault(cur_cell, [])
            cells_refs.setdefault(cur_cell, [])
        elif rec_type == _BOUNDARY:
            cur = {"kind": "boundary", "layer": 0, "datatype": 0, "xy": []}
        elif rec_type == _PATH:
            cur = {"kind": "path", "layer": 0, "datatype": 0, "width": 0, "xy": []}
        elif rec_type in (_SREF, _AREF):
            cur = {
                "kind": "aref" if rec_type == _AREF else "sref",
                "sname": "",
                "strans": 0,
                "mag": 1.0,
                "angle": 0.0,
                "cols": 1,
                "rows": 1,
                "xy": [],
            }
        elif cur is not None:
            if rec_type == _LAYER and len(payload) >= 2:
                cur["layer"] = struct.unpack(">h", payload[:2])[0]
            elif rec_type == _DATATYPE and len(payload) >= 2:
                cur["datatype"] = struct.unpack(">h", payload[:2])[0]
            elif rec_type == _WIDTH and len(payload) >= 4:
                cur["width"] = struct.unpack(">i", payload[:4])[0]
            elif rec_type == _SNAME:
                cur["sname"] = payload.rstrip(b"\x00").decode("ascii", "ignore")
            elif rec_type == _STRANS and len(payload) >= 2:
                cur["strans"] = struct.unpack(">H", payload[:2])[0]
            elif rec_type == _MAG and len(payload) >= 8:
                cur["mag"] = ibm_to_float(payload[:8])
            elif rec_type == _ANGLE and len(payload) >= 8:
                cur["angle"] = ibm_to_float(payload[:8])
            elif rec_type == _COLROW and len(payload) >= 4:
                cols, rows = struct.unpack(">hh", payload[:4])
                cur["cols"] = max(1, cols)
                cur["rows"] = max(1, rows)
            elif rec_type == _XY:
                n = len(payload) // 8
                cur["xy"] = [
                    struct.unpack(">ii", payload[8 * k : 8 * k + 8]) for k in range(n)
                ]
            elif rec_type == _ENDEL and cur_cell is not None:
                kind = cur["kind"]
                if kind == "boundary" and len(cur["xy"]) >= 3:
                    pts = [(float(x), float(y)) for x, y in cur["xy"]]
                    if pts[0] == pts[-1]:
                        pts = pts[:-1]
                    if len(pts) >= 3:
                        cells_polys[cur_cell].append((cur["layer"], cur["datatype"], pts))
                elif kind == "path" and len(cur["xy"]) >= 2:
                    raw_pts = [(float(x), float(y)) for x, y in cur["xy"]]
                    expanded = _path_to_polygon(raw_pts, float(cur.get("width", 0)))
                    if len(expanded) >= 3:
                        cells_polys[cur_cell].append((cur["layer"], cur["datatype"], expanded))
                elif kind in ("sref", "aref") and cur.get("sname") and cur.get("xy"):
                    referenced.add(cur["sname"])
                    cells_refs[cur_cell].append(cur)
                cur = None
        elif rec_type in (_ENDSTR, _ENDLIB):
            cur = None
            cur_cell = None

    mult = float(scale) if scale is not None else db_unit_um

    def _collect(cname: str, depth: int = 0) -> list[tuple[int, int, list[tuple[float, float]]]]:
        if depth > 16 or cname not in cells_polys:
            return []
        result = list(cells_polys[cname])
        for ref in cells_refs.get(cname, []):
            sub = _collect(ref["sname"], depth + 1)
            if not sub:
                continue
            reflect_x = bool(ref["strans"] & 0x8000)
            mag = float(ref["mag"])
            ang = float(ref["angle"])
            xy = ref["xy"]
            if ref["kind"] == "aref" and len(xy) >= 3:
                cols, rows = ref["cols"], ref["rows"]
                ox0, oy0 = float(xy[0][0]), float(xy[0][1])
                col_dx = (float(xy[1][0]) - ox0) / cols
                col_dy = (float(xy[1][1]) - oy0) / cols
                row_dx = (float(xy[2][0]) - ox0) / rows
                row_dy = (float(xy[2][1]) - oy0) / rows
                for ci in range(cols):
                    for ri in range(rows):
                        ox = ox0 + ci * col_dx + ri * row_dx
                        oy = oy0 + ci * col_dy + ri * row_dy
                        for lyr, dt, ppts in sub:
                            result.append((lyr, dt, _transform_pts(ppts, reflect_x, mag, ang, ox, oy)))
            else:
                ox, oy = float(xy[0][0]), float(xy[0][1])
                for lyr, dt, ppts in sub:
                    result.append((lyr, dt, _transform_pts(ppts, reflect_x, mag, ang, ox, oy)))
        return result

    if cell_name and cell_name in cells_polys:
        roots = [cell_name]
    else:
        top = [c for c in cells_polys if c not in referenced]
        roots = top if top else list(cells_polys.keys())

    out: list[Polygon] = []
    for root in roots:
        for lyr, dt, pts in _collect(root):
            scaled = [(x * mult, y * mult) for x, y in pts]
            out.append(Polygon(lyr, dt, root, scaled))
    return out


def _main(argv: list[str]) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Print layer/bounds of a GDSII file")
    ap.add_argument("path", nargs="?", default="configs/generated_fivewire.gds")
    ap.add_argument(
        "--scale",
        type=float,
        default=None,
        help="explicit multiplier for database units",
    )
    args = ap.parse_args(argv)

    polys = read_polygons(args.path, scale=args.scale)
    print(f"{args.path}: {len(polys)} polygons")
    for p in polys:
        xs = [x for x, _ in p.xy_um]
        ys = [y for _, y in p.xy_um]
        print(
            f"layer {p.layer:3d} datatype {p.datatype} n={len(p.xy_um):3d} "
            f"x=[{min(xs):10.3f},{max(xs):10.3f}] y=[{min(ys):10.3f},{max(ys):10.3f}]"
        )


if __name__ == "__main__":
    import sys

    _main(sys.argv[1:])
