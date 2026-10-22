"""Minimal pure-Python GDSII reader (boundaries only).

Used by :mod:`tools.make_docs_assets` to draw the *real* electrode layout of
``configs/generated_fivewire.gds`` in the README figures.  No gdspy / no
python-gdsii dependency, so it also works in a bare docs environment.

Returns polygons in **micrometres**, honouring the UNITS record of the file
(user unit -> metres).  Trailing points that repeat the first vertex (GDS
closing convention) are dropped.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

# GDSII record types (only the ones we need)
_HEADER = 0x00
_BGNLIB = 0x01
_LIBNAME = 0x02
_UNITS = 0x03
_BGNSTR = 0x05
_STRNAME = 0x06
_BOUNDARY = 0x08
_PATH = 0x09
_SREF = 0x0A
_LAYER = 0x0D
_DATATYPE = 0x0E
_XY = 0x10
_ENDEL = 0x11
_ENDSTR = 0x07
_ENDLIB = 0x04


@dataclass
class Polygon:
    layer: int
    datatype: int
    cell: str
    xy_um: list[tuple[float, float]]


def read_polygons(path: str | Path, scale: float | None = None) -> list[Polygon]:
    """Parse all BOUNDARY elements of a GDSII file.

    Coordinates are multiplied by ``scale``.  By default the UNITS record is
    used (database unit expressed in micrometres); pass an explicit ``scale``
    when the file carries a degenerate/placeholder UNITS record.
    """
    data = Path(path).read_bytes()
    i = 0
    polys: list[Polygon] = []
    cur: dict | None = None
    cell = ""
    db_unit_um = 1e-3  # fallback: 1 nm
    if scale is not None:
        db_unit_um = float(scale)

    while i + 4 <= len(data):
        ln = struct.unpack(">H", data[i : i + 2])[0]
        if ln == 0:  # padding at end of stream
            break
        rec_type, data_type = data[i + 2], data[i + 3]
        payload = data[i + 4 : i + ln]
        i += ln

        if rec_type == _UNITS and scale is None:
            _user, db = struct.unpack(">dd", payload)
            db_unit_um = db * 1e6
        if rec_type == _STRNAME:
            cell = payload.rstrip(b"\x00").decode("ascii", "ignore")
        elif rec_type == _BOUNDARY:
            cur = {"layer": 0, "datatype": 0, "xy": []}
        elif rec_type == _LAYER and cur is not None:
            cur["layer"] = struct.unpack(">h", payload)[0]
        elif rec_type == _DATATYPE and cur is not None:
            cur["datatype"] = struct.unpack(">h", payload)[0]
        elif rec_type == _XY and cur is not None:
            n = len(payload) // 8
            cur["xy"] = [
                struct.unpack(">ii", payload[8 * k : 8 * k + 8]) for k in range(n)
            ]
        elif rec_type == _ENDEL:
            if cur is not None and len(cur["xy"]) >= 3:
                pts = [(x * db_unit_um, y * db_unit_um) for x, y in cur["xy"]]
                if pts[0] == pts[-1]:
                    pts = pts[:-1]
                polys.append(Polygon(cur["layer"], cur["datatype"], cell, pts))
            cur = None
        elif rec_type in (_ENDSTR, _ENDLIB):
            cur = None

    return polys


def _main(argv: list[str]) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Print layer/layer bounds of a GDSII file")
    ap.add_argument("path", nargs="?", default="configs/generated_fivewire.gds")
    ap.add_argument(
        "--scale",
        type=float,
        default=None,
        help="explicit multiplier for database units (needed when the UNITS record is a placeholder)",
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
