import base64
import html.parser
import json
import pathlib
import threading
import urllib.request
from http.server import HTTPServer

from shuttle_sim.dashboard.server import (
    Handler,
    _point_in_poly,
    build_trap_preview,
)
from tools.gds_reader import read_polygons


def test_gds_reader_units_and_scale_compat():
    polys_um = read_polygons("configs/generated_fivewire.gds")
    assert len(polys_um) == 17
    xs = [pt[0] for p in polys_um for pt in p.xy_um]
    ys = [pt[1] for p in polys_um for pt in p.xy_um]
    assert abs(min(xs) - (-10000.0)) < 1e-3
    assert abs(max(xs) - 10000.0) < 1e-3
    assert abs(min(ys) - (-8000.0)) < 1e-3
    assert abs(max(ys) - 8000.0) < 1e-3

    # Backward compatibility for explicit scale callers (e.g. tools/make_docs_assets.py)
    polys_scaled = read_polygons("configs/generated_fivewire.gds", scale=1e-6)
    xs_s = [pt[0] for p in polys_scaled for pt in p.xy_um]
    assert abs(max(xs_s) - 10.0) < 1e-4


def test_trap_preview_all_repo_gds_and_interior_labels():
    cfg_gds = json.loads(pathlib.Path("configs/shuttle_gds.json").read_text())

    # Mapped 17-channel GDS
    prev_mapped = build_trap_preview("configs/generated_fivewire.gds", cfg=cfg_gds)
    assert prev_mapped["mapped"] is True
    assert prev_mapped["electrode_count"] == 17
    assert any(e["type"] == "rf" for e in prev_mapped["electrodes"])
    assert any(e["type"] == "dc" for e in prev_mapped["electrodes"])

    # Unmapped multi-polygon GDS layouts (including non-convex L/T rails)
    expected_counts = {
        "examples/gds/test.GDS": 8,
        "examples/gds/gapped_test.GDS": 8,
        "examples/gds/t_junction.GDS": 29,
    }
    for gds_file, expected_n in expected_counts.items():
        prev = build_trap_preview(gds_file, cfg=cfg_gds)
        assert prev["mapped"] is False
        assert prev["polygon_count"] == expected_n
        assert prev["electrode_count"] == expected_n
        for ele in prev["electrodes"]:
            lx, ly = ele["label_pt"]
            assert _point_in_poly(lx, ly, ele["paths"][0]), (
                f"Label point {(lx, ly)} outside polygon {ele['name']} in {gds_file}"
            )


def test_trap_preview_analytic_five_wire():
    cfg_ref = json.loads(pathlib.Path("configs/shuttle_reference.json").read_text())
    prev = build_trap_preview(cfg=cfg_ref)
    assert prev["kind"] == "sion_five_wire"
    assert prev["electrode_count"] >= 5


class _TagBalanceChecker(html.parser.HTMLParser):
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__()
        self.stack = []
        self.mismatches = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.mismatches.append((list(self.stack), tag))
        else:
            self.stack.pop()


def test_dashboard_http_upload_and_clean_main_page():
    srv = HTTPServer(("127.0.0.1", 0), lambda *a, **kw: Handler(*a, out_dir="output/ga_traj", **kw))
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    uploaded_file = pathlib.Path("configs/uploads/_pytest_t_junction.gds")
    try:
        base = f"http://127.0.0.1:{port}"
        html_text = urllib.request.urlopen(f"{base}/").read().decode("utf-8")

        checker = _TagBalanceChecker()
        checker.feed(html_text)
        assert not checker.mismatches, f"Unbalanced closing tags: {checker.mismatches}"
        assert not checker.stack, f"Unclosed tags: {checker.stack}"

        # Main page has trap canvas and no extraneous non-dashboard commentary
        assert 'id="cSetupTrap"' in html_text
        for clutter in (
            "Как это работает",
            "Быстрый старт для большой ловушки",
            "GA writes",
            "одновременно — только одно обучение",
            "тот самый общий файл",
        ):
            assert clutter not in html_text, f"Extraneous text still present: {clutter}"

        # Config endpoint includes trap_preview immediately
        cfg_resp = json.loads(urllib.request.urlopen(f"{base}/api/config").read().decode("utf-8"))
        assert cfg_resp["trap_preview"] is not None
        assert cfg_resp["trap_preview"]["electrode_count"] == 17

        # Upload a real GDS file and verify immediate trap preview payload in response
        raw_bytes = pathlib.Path("examples/gds/t_junction.GDS").read_bytes()
        payload = json.dumps({
            "filename": "_pytest_t_junction.gds",
            "content_base64": base64.b64encode(raw_bytes).decode("ascii"),
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/api/upload",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        up_resp = json.loads(urllib.request.urlopen(req).read().decode("utf-8"))
        assert up_resp["ok"] is True
        assert up_resp["trap"]["polygon_count"] == 29
        assert up_resp["trap"]["filename"] == "_pytest_t_junction.gds"
    finally:
        srv.shutdown()
        srv.server_close()
        if uploaded_file.exists():
            uploaded_file.unlink()
