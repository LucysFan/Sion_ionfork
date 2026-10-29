#!/usr/bin/env python3
"""
Generate Island GA architecture assets:
 - assets/fig-ga-islands.svg  (static vector overview, dark theme, for README)
 - assets/anim-ga-islands.gif (animated GIF: evolution, migrations, quarantine bursts)

No external deps beyond matplotlib, numpy, pillow (already in requirements-shuttle.txt).
Run: python tools/make_ga_islands_assets.py
"""

import math
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared island definitions (must match docs/ga_architecture.md & engine.py)
# ---------------------------------------------------------------------------
ISLANDS = [
    # active ring S -> A -> V -> W -> R -> H -> S
    dict(id="S", name="S", full="S smooth", dim=8, baseline="smooth",
         sigma="0.04", role="адиабатика", color="#5ac8ff", desc="env без STA", pos_angle=-90),
    dict(id="A", name="A", full="A STA", dim=8, baseline="sta",
         sigma="0.06", role="STA короткий T", color="#ffbe55", desc="s''/(Tω)²", pos_angle=-30),
    dict(id="V", name="V", full="V voltage", dim=10, baseline="sta",
         sigma="0.08", role="напряжение", color="#f0883e", desc="4×2 ΔV  sin2πs", pos_angle=30),
    dict(id="W", name="W", full="W frequency", dim=10, baseline="sta",
         sigma="0.08", role="частота ω(s)", color="#3fb950", desc="2 freq  ±18%", pos_angle=90),
    dict(id="R", name="R", full="R robust", dim=8, baseline="sta",
         sigma="0.06", role="worst-case", color="#c58aff", desc="0.35mean+0.65max", pos_angle=150),
    dict(id="H", name="H", full="H hardware", dim=8, baseline="smooth",
         sigma="0.05", role="low-pass", color="#ff7a7a", desc="штраф rms/dist", pos_angle=210),
]
QUARANTINE = [
    dict(id="Q1", name="Q1", full="Q1 карантин", dim=8, baseline="smooth",
         sigma="0.05", until=12, color="#ff6b9e", desc="smooth·изол.12"),
    dict(id="Q2", name="Q2", full="Q2 карантин", dim=8, baseline="sta",
         sigma="0.07", until=18, color="#5affda", desc="sta·изол.18"),
    dict(id="F",  name="F",  full="F free", dim=12, baseline="smooth",
         sigma="0.07", until=15, color="#ffd05a", desc="free 12·изол.15"),
]

# ---------------------------------------------------------------------------
# 1) Static SVG
# ---------------------------------------------------------------------------
def make_static_svg():
    W, H = 1200, 720
    cx, cy, R = 600, 340, 185
    svg = []
    def add(s): svg.append(s)

    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">')
    add('<title id="title">Island GA — 6 активных + 3 карантинные острова</title>')
    add('<desc id="desc">Кольцо S→A→V→W→R→H, миграции каждые 5 поколений, Q1/Q2/F карантин с burst в 12/18/15</desc>')
    add('<rect width="100%" height="100%" rx="18" fill="#0f141c"/>')
    # subtle grid
    add('<g opacity="0.06" stroke="#5ac8ff" stroke-width="0.7">')
    for x in range(0, W, 80):
        add(f'<line x1="{x}" y1="0" x2="{x}" y2="{H}"/>')
    for y in range(0, H, 80):
        add(f'<line x1="0" y1="{y}" x2="{W}" y2="{y}"/>')
    add('</g>')
    # title
    add(f'<text x="{W/2}" y="44" text-anchor="middle" font-family="Inter,ui-sans-serif,sans-serif" font-size="22" font-weight="800" fill="#e6efff">Архипелаг эволюции — островная GA</text>')
    add(f'<text x="{W/2}" y="66" text-anchor="middle" font-family="Inter,ui-sans-serif,sans-serif" font-size="12.5" fill="#8aa0c8">6 ниш держат диверсити · 3 карантина ищут без навязанного профиля · кольцо + burst</text>')
    # central hub
    add(f'<circle cx="{cx}" cy="{cy}" r="58" fill="#17202f" stroke="#243556" stroke-width="1.6"/>')
    add(f'<circle cx="{cx}" cy="{cy}" r="42" fill="none" stroke="#2a3f62" stroke-width="1" stroke-dasharray="4 6" opacity="0.9"/>')
    add(f'<text x="{cx}" y="{cy-10}" text-anchor="middle" font-family="Inter,sans-serif" font-size="10.5" font-weight="700" fill="#e6efff">HALL OF</text>')
    add(f'<text x="{cx}" y="{cy+6}" text-anchor="middle" font-family="Inter,sans-serif" font-size="10.5" font-weight="700" fill="#e6efff">FAME</text>')
    add(f'<text x="{cx}" y="{cy+22}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="8.5" fill="#8aa0c8">min_dist 0.12  θ</text>')
    add(f'<text x="{cx}" y="{cy+54}" text-anchor="middle" font-family="Inter,sans-serif" font-size="8.5" fill="#5ac8ff">S→A→V→W→R→H→S</text>')

    # quarantine satellites (top-right cluster)
    q_pos = {
        "Q1": (980, 150),
        "Q2": (980, 430),
        "F":  (220, 430),
    }
    # subtle connectors from center to quarantine (dashed, closed)
    for q in QUARANTINE:
        qx, qy = q_pos[q["id"]]
        add(f'<line x1="{cx}" y1="{cy}" x2="{qx}" y2="{qy}" stroke="{q["color"]}" stroke-width="1.1" stroke-dasharray="8 7" opacity="0.18"/>')

    # ring positions
    pos = {}
    for isl in ISLANDS:
        ang = math.radians(isl["pos_angle"])
        x = cx + R*math.cos(ang)
        y = cy + R*math.sin(ang)
        pos[isl["id"]] = (x,y)

    # migration arrows along ring (directed)
    order = ["S","A","V","W","R","H"]
    for i in range(len(order)):
        src = order[i]
        dst = order[(i+1) % len(order)]
        x1,y1 = pos[src]
        x2,y2 = pos[dst]
        # shorten arrow to not overlap circles
        vx, vy = x2-x1, y2-y1
        L = math.hypot(vx,vy)
        ux, uy = vx/L, vy/L
        x1s, y1s = x1 + ux*62, y1 + uy*62
        x2s, y2s = x2 - ux*62, y2 - uy*62
        # shaft
        add(f'<line x1="{x1s:.1f}" y1="{y1s:.1f}" x2="{x2s:.1f}" y2="{y2s:.1f}" stroke="#2a3f62" stroke-width="2.1" stroke-linecap="round" opacity="0.95"/>')
        # Animated dot will be via CSS on GIF, but for SVG add arrowhead
        add(f'<polygon points="{x2s:.1f},{y2s:.1f} {x2s-10*ux+6*uy:.1f},{y2s-10*uy-6*ux:.1f} {x2s-10*ux-6*uy:.1f},{y2s-10*uy+6*ux:.1f}" fill="#5ac8ff" opacity="0.9"/>')
        # label every 2nd edge
        if src in ("S","V","R"):
            mx, my = (x1s+x2s)/2, (y1s+y2s)/2
            add(f'<rect x="{mx-32}" y="{my-10}" width="64" height="13" rx="6" fill="#0f141c" stroke="#243556" stroke-width="0.7"/>')
            add(f'<text x="{mx}" y="{my}" dy="3.5" text-anchor="middle" font-family="ui-monospace,monospace" font-size="7.5" fill="#8aa0c8">5 gen · 2</text>')

    # draw islands (circles with glow)
    for isl in ISLANDS:
        x,y = pos[isl["id"]]
        col = isl["color"]
        # glow
        add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="68" fill="{col}" opacity="0.08"/>')
        add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="58" fill="#17202f" stroke="{col}" stroke-width="2.2"/>')
        add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="56" fill="none" stroke="{col}" stroke-width="0.8" opacity="0.22"/>')
        # dim badge
        add(f'<rect x="{x+34:.1f}" y="{y-58:.1f}" width="28" height="16" rx="8" fill="{col}" />')
        add(f'<text x="{x+48:.1f}" y="{y-47:.1f}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="9" font-weight="800" fill="#0f141c">{isl["dim"]}</text>')
        # island letter
        add(f'<text x="{x:.1f}" y="{y-12:.1f}" text-anchor="middle" font-family="Inter,sans-serif" font-size="22" font-weight="900" fill="#e6efff">{isl["id"]}</text>')
        add(f'<text x="{x:.1f}" y="{y+3:.1f}" text-anchor="middle" font-family="Inter,sans-serif" font-size="7.5" font-weight="700" fill="{col}" letter-spacing="0.04em">{isl["baseline"].upper()}</text>')
        # role
        add(f'<text x="{x:.1f}" y="{y+18:.1f}" text-anchor="middle" font-family="Inter,sans-serif" font-size="8.2" font-weight="600" fill="#c8d7f0">{isl["role"]}</text>')
        add(f'<text x="{x:.1f}" y="{y+29:.1f}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="7" fill="#8aa0c8">{isl["desc"]}</text>')
        add(f'<text x="{x:.1f}" y="{y+39:.1f}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="7" fill="#8aa0c8">σ{isl["sigma"]} · {isl["dim"]}θ</text>')

    # quarantine islands
    for q in QUARANTINE:
        qx,qy = q_pos[q["id"]]
        col = q["color"]
        # lock card
        add(f'<rect x="{qx-64}" y="{qy-46}" width="128" height="92" rx="14" fill="#111a2b" stroke="{col}" stroke-width="1.6" stroke-dasharray="7 5" opacity="0.98"/>')
        add(f'<rect x="{qx-64}" y="{qy-46}" width="128" height="22" rx="14" fill="{col}" opacity="0.92"/>')
        # clip top radius for header
        add(f'<rect x="{qx-64}" y="{qy-32}" width="128" height="10" fill="{col}" opacity="0.92"/>')
        add(f'<text x="{qx}" y="{qy-31}" text-anchor="middle" font-family="Inter,sans-serif" font-size="9" font-weight="800" fill="#0f141c">🔒  {q["full"]}</text>')
        add(f'<text x="{qx}" y="{qy-6}" text-anchor="middle" font-family="Inter,sans-serif" font-size="18" font-weight="900" fill="#e6efff">{q["id"]}</text>')
        add(f'<text x="{qx}" y="{qy+10}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="7.5" fill="#c8d7f0">{q["desc"]}</text>')
        add(f'<rect x="{qx-34}" y="{qy+16}" width="68" height="16" rx="8" fill="#0f141c" stroke="{col}" stroke-width="1"/>')
        add(f'<text x="{qx}" y="{qy+27}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="8" font-weight="700" fill="{col}">до {q["until"]} gen</text>')

    # bottom legend timeline
    add(f'<rect x="56" y="560" width="{W-112}" height="96" rx="14" fill="#17202f" stroke="#243556" stroke-width="1.2"/>')
    add(f'<text x="78" y="584" font-family="Inter,sans-serif" font-size="10" font-weight="700" fill="#e6efff">ЭВОЛЮЦИЯ ВО ВРЕМЕНИ · от изоляции к обмену</text>')
    # timeline line
    tl_y = 612
    add(f'<line x1="78" y1="{tl_y}" x2="{W-78}" y2="{tl_y}" stroke="#2a3f62" stroke-width="2" stroke-linecap="round"/>')
    # ticks 0..30
    for g in [0,5,10,12,15,18,20,25,30]:
        x = 78 + (g/30)*(W-156)
        add(f'<line x1="{x:.1f}" y1="{tl_y-8}" x2="{x:.1f}" y2="{tl_y+8}" stroke="#5ac8ff" stroke-width="1.6"/>')
        add(f'<text x="{x:.1f}" y="{tl_y+22}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="8" fill="#8aa0c8">{g}</text>')
    # colored quarantine bars
    for q in QUARANTINE:
        until = q["until"]
        x0 = 78
        x1 = 78 + (until/30)*(W-156)
        x2 = W-78
        # isolated segment
        add(f'<line x1="{x0:.1f}" y1="{tl_y}" x2="{x1:.1f}" y2="{tl_y}" stroke="{q["color"]}" stroke-width="6" stroke-linecap="round" opacity="0.95"/>')
        add(f'<circle cx="{x1:.1f}" cy="{tl_y}" r="7" fill="{q["color"]}" stroke="#0f141c" stroke-width="2"/>')
        add(f'<text x="{x1:.1f}" y="{tl_y-14}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="7" font-weight="700" fill="{q["color"]}">burst {q["id"]}</text>')
    # annotations
    add(f'<text x="78" y="648" font-family="Inter,sans-serif" font-size="8.5" fill="#8aa0c8">0 gen — Q1/Q2/F живут сами, не отдают и не принимают (isolated_until). Каждые 5 gen: кольцо шлёт <tspan fill="#e6efff" font-weight="700">best + самый дальний по θ</tspan> → 2 худших соседа. В 12/15/18 — <tspan fill="#7af0a0" font-weight="700">burst</tspan> проснувшегося острова → всем активным + global shuffle.</text>')

    # footer code refs
    add(f'<text x="{W/2}" y="{H-10}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="7.5" fill="#5a6b8a">shuttle_sim/ga/{chr(123)}genome,island,fitness,migration,engine{"}"}  ·  topology S→A→V→W→R→H→S  every 5 gen  n_migrants=2  hall min_dist 0.12  ·  pipeline.prepare clip ω ±18%  ·  dashboard: is_isolated → серая иконка</text>')
    add('</svg>')

    out = ASSETS / "fig-ga-islands.svg"
    out.write_text("\n".join(svg), encoding="utf-8")
    print(f"[svg] wrote {out} ({len(svg)} lines)")

# ---------------------------------------------------------------------------
# 2) Animated GIF
# ---------------------------------------------------------------------------
def make_animated_gif():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
        from matplotlib.collections import PatchCollection
        from PIL import Image
    except Exception as e:
        print(f"[gif] skip, missing deps: {e}")
        return

    W, H = 1200, 720
    cx, cy, R = 600, 340, 185
    order = ["S","A","V","W","R","H"]
    # positions
    pos = {}
    for isl in ISLANDS:
        ang = math.radians(isl["pos_angle"])
        pos[isl["id"]] = (cx + R*math.cos(ang), cy + R*math.sin(ang))
    q_pos = {"Q1": (980,150), "Q2": (980,430), "F": (220,430)}

    # color map
    cmap = {isl["id"]: isl["color"] for isl in ISLANDS}
    cmap.update({q["id"]: q["color"] for q in QUARANTINE})

    def hex_to_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2],16)/255 for i in (0,2,4))

    # fake evolution: fitness improves, diversity decays
    gens = list(range(0, 31))
    # For each island, generate fake best fitness trajectory
    rng = np.random.default_rng(42)
    base_fits = {"S": 4.5, "A": 4.2, "V": 3.9, "W": 3.7, "R": 4.0, "H": 4.3, "Q1": 5.0, "Q2": 5.2, "F": 4.8}
    traces = {}
    for k, bf in base_fits.items():
        # exponential decay + noise
        arr = []
        v = bf
        for g in gens:
            # improvement slows
            v = v*0.985 + rng.normal(0, 0.06) - 0.02
            v = max(0.45, v)
            # Q islands improve faster early because isolated
            if k in ("Q1","Q2","F") and g < {"Q1":12,"Q2":18,"F":15}[k]:
                v += 0.25 # they lag before burst?
            arr.append(v)
        # smooth
        traces[k] = np.array(arr)
    # Global best is min of active
    global_best = []
    for g in gens:
        active = ["S","A","V","W","R","H"]
        for q in QUARANTINE:
            if g >= q["until"]:
                active.append(q["id"])
        global_best.append(min(traces[k][g] for k in active))
    global_best = np.array(global_best)

    frames = []
    for g in gens:
        fig = plt.figure(figsize=(12, 7.2), dpi=100)
        fig.patch.set_facecolor("#0f141c")
        ax = fig.add_axes([0,0,1,1])
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor("#0f141c")

        # grid faint
        for x in range(0, W, 80):
            ax.plot([x,x],[0,H], color="#5ac8ff", lw=0.6, alpha=0.04)
        for y in range(0, H, 80):
            ax.plot([0,W],[y,y], color="#5ac8ff", lw=0.6, alpha=0.04)

        # title
        ax.text(W/2, 680, "Island GA  —  эволюция 30 поколений", ha="center", va="center",
                fontsize=16, fontweight=800, color="#e6efff", fontfamily="sans-serif")
        ax.text(W/2, 660, f"поколение {g:02d} / 30   ·   global best {global_best[g]:.2f}   ·   {'[x] изоляция' if g<12 else '[o] открыто' if g>=18 else '[o] Q1 открыт'}",
                ha="center", va="center", fontsize=10, color="#8aa0c8", fontfamily="monospace")

        # central hall
        hall = Circle((cx, cy), 58, facecolor="#17202f", edgecolor="#243556", lw=1.6)
        ax.add_patch(hall)
        hall2 = Circle((cx, cy), 42, facecolor="none", edgecolor="#2a3f62", lw=1, linestyle=(0,(4,6)), alpha=0.9)
        ax.add_patch(hall2)
        ax.text(cx, cy-8, "HALL OF", ha="center", va="center", color="#e6efff", fontsize=9, fontweight=700)
        ax.text(cx, cy+6, "FAME", ha="center", va="center", color="#e6efff", fontsize=9, fontweight=700)
        ax.text(cx, cy+22, "min_dist 0.12", ha="center", va="center", color="#8aa0c8", fontsize=7, fontfamily="monospace")

        # quarantine backgrounds (closed border until unlock)
        for q in QUARANTINE:
            qx, qy = q_pos[q["id"]]
            unlocked = g >= q["until"]
            # connector
            ax.plot([cx, qx], [cy, qy], color=q["color"], lw=1.1, linestyle=(0,(8,7)), alpha=0.16 if unlocked else 0.09)
            # card
            card_alpha = 1.0 if unlocked else 0.55
            rect = Rectangle((qx-64, qy-46), 128, 92, fc="#111a2b", ec=q["color"], lw=1.7 if unlocked else 1.2,
                             linestyle="--" if not unlocked else "-", alpha=card_alpha, capstyle="round")
            # matplotlib doesn't do dash for Rectangle easily, so we handle via patch
            ax.add_patch(rect)
            # header
            hdr = Rectangle((qx-64, qy-46), 128, 22, fc=q["color"], ec="none", alpha=0.92 if unlocked else 0.6)
            ax.add_patch(hdr)
            ax.add_patch(Rectangle((qx-64, qy-32), 128, 10, fc=q["color"], ec="none", alpha=0.92 if unlocked else 0.6))
            ax.text(qx, qy-31, f"{'[o]' if unlocked else '[x]'}  {q['full']}", ha="center", va="center",
                    fontsize=8, fontweight=800, color="#0f141c")
            ax.text(qx, qy-6, q["id"], ha="center", va="center", fontsize=14, fontweight=900, color="#e6efff", alpha=card_alpha)
            ax.text(qx, qy+9, q["desc"], ha="center", va="center", fontsize=6.5, color="#c8d7f0", fontfamily="monospace", alpha=0.9)
            # until badge
            badge_col = q["color"] if unlocked else "#2a3f62"
            badge_face = "#0f141c" if not unlocked else q["color"]
            badge_text_col = q["color"] if not unlocked else "#0f141c"
            # badge rect
            ax.add_patch(Rectangle((qx-34, qy+16), 68, 16, fc=badge_face, ec=badge_col, lw=1, alpha=0.95))
            ax.text(qx, qy+24, f"до {q['until']} gen" if not unlocked else f"burst {q['id']}", ha="center", va="center",
                    fontsize=7, fontweight=700, color=badge_text_col, fontfamily="monospace")
            # if burst this generation, flash
            if g == q["until"]:
                circ = Circle((qx, qy), 78, facecolor="none", edgecolor=q["color"], lw=2.5, alpha=0.85)
                ax.add_patch(circ)
                ax.text(qx, qy+54, "BURST → всем активным", ha="center", va="center", fontsize=7, fontweight=700, color=q["color"],
                        bbox=dict(facecolor="#0f141c", edgecolor=q["color"], boxstyle="round,pad=0.3", alpha=0.9))

        # ring migration arrows
        for i in range(len(order)):
            src = order[i]; dst = order[(i+1)%len(order)]
            x1,y1 = pos[src]; x2,y2 = pos[dst]
            vx, vy = x2-x1, y2-y1
            L = math.hypot(vx,vy)
            ux, uy = vx/L, vy/L
            x1s, y1s = x1 + ux*62, y1 + uy*62
            x2s, y2s = x2 - ux*62, y2 - uy*62
            # Only active ring always
            is_migr = (g != 0 and g % 5 == 0)  # migration every 5
            alpha = 0.95 if is_migr else 0.35
            lw = 2.6 if is_migr else 1.4
            col = "#5ac8ff" if is_migr else "#2a3f62"
            ax.plot([x1s, x2s], [y1s, y2s], color=col, lw=lw, alpha=alpha, solid_capstyle="round")
            # arrowhead
            if is_migr:
                arr = FancyArrowPatch((x1s, y1s), (x2s, y2s), arrowstyle='-|>', mutation_scale=10,
                                      color=col, lw=lw, alpha=alpha)
                ax.add_patch(arr)
                # moving dot
                t = (g % 5) / 5 if g %5 !=0 else 1.0
                # But for migration gen, show dot at tip
                dot_x = x1s*(1-t) + x2s*t
                dot_y = y1s*(1-t) + y2s*t
                ax.add_patch(Circle((dot_x, dot_y), 4, facecolor="#7af0a0", edgecolor="#0f141c", lw=1, alpha=0.95))
            else:
                # arrowhead small
                ax.plot([x2s], [y2s], marker=10, markersize=6, color=col, alpha=alpha)  # placeholder
                # use polygon manually
                hx1 = x2s - 10*ux + 6*uy
                hy1 = y2s - 10*uy - 6*ux
                hx2 = x2s - 10*ux - 6*uy
                hy2 = y2s - 10*uy + 6*ux
                ax.fill([x2s, hx1, hx2], [y2s, hy1, hy2], color=col, alpha=alpha*0.9, edgecolor="none")

        # islands with population clouds
        for isl in ISLANDS:
            x,y = pos[isl["id"]]
            col = isl["color"]
            fit = traces[isl["id"]][g]
            # glow
            ax.add_patch(Circle((x,y), 68, facecolor=hex_to_rgb(col), alpha=0.07, edgecolor="none"))
            ax.add_patch(Circle((x,y), 58, facecolor="#17202f", edgecolor=col, lw=2.3, alpha=0.98))
            # inner ring
            ax.add_patch(Circle((x,y), 56, facecolor="none", edgecolor=col, lw=0.8, alpha=0.22))
            # top badge dim
            ax.add_patch(Rectangle((x+34, y-58), 28, 16, fc=col, ec="none"))
            # but rectangle with rounded corners manual: use FancyBbox not available, so use patch
            # add dim text
            ax.text(x+48, y-49, f"{isl['dim']}", ha="center", va="center", fontsize=8, fontweight=800, color="#0f141c", fontfamily="monospace")
            ax.text(x, y-12, isl["id"], ha="center", va="center", fontsize=17, fontweight=900, color="#e6efff")
            ax.text(x, y+3, isl["baseline"].upper(), ha="center", va="center", fontsize=6.5, fontweight=700, color=col)
            ax.text(x, y+16, isl["role"], ha="center", va="center", fontsize=7.2, fontweight=600, color="#c8d7f0")
            ax.text(x, y+27, isl["desc"], ha="center", va="center", fontsize=6.2, color="#8aa0c8", fontfamily="monospace")
            ax.text(x, y+37, f"σ{isl['sigma']}  f{fit:.2f}", ha="center", va="center", fontsize=6.5, color="#e6efff", fontfamily="monospace",
                    bbox=dict(facecolor="#0f141c", edgecolor="#243556", boxstyle="round,pad=0.2", alpha=0.85))
            # population dots around
            # diversity = spread of dots, larger when early
            div = max(0.12, 0.45 - g*0.012 + rng.normal(0,0.015))
            n_dots = 18
            for k in range(n_dots):
                ang = rng.uniform(0, 2*math.pi)
                rad_hi = max(12, 84*div)
                rad = rng.uniform(4, rad_hi)
                dx = math.cos(ang)*rad
                dy = math.sin(ang)*rad
                # color jitter
                ax.add_patch(Circle((x+dx, y+dy), 2.1, facecolor=col, alpha=0.55 + 0.35* (1 - fit/5), edgecolor="none"))
            # elite star
            ax.text(x, y-44, "★", ha="center", va="center", fontsize=9, color=col, alpha=0.95)

        # quarantine island dots (they evolve alone before burst)
        for q in QUARANTINE:
            qx, qy = q_pos[q["id"]]
            unlocked = g >= q["until"]
            # show isolated evolution as small cloud inside card
            fit = traces[q["id"]][g]
            if not unlocked:
                # show locked evolution: grayish
                for k in range(10):
                    ang = rng.uniform(0, 2*math.pi)
                    rad = rng.uniform(4, 14)
                    dx = math.cos(ang)*rad*0.6
                    dy = math.sin(ang)*rad*0.6
                    ax.add_patch(Circle((qx+dx, qy-14+dy*0.35), 1.7, facecolor=q["color"], alpha=0.45, edgecolor="none"))
            else:
                for k in range(14):
                    ang = rng.uniform(0, 2*math.pi)
                    rad = rng.uniform(3, 18)
                    dx = math.cos(ang)*rad
                    dy = math.sin(ang)*rad*0.5
                    ax.add_patch(Circle((qx+dx, qy-14+dy), 1.8, facecolor=q["color"], alpha=0.7, edgecolor="none"))

        # bottom timeline bar
        bar_x0, bar_x1, bar_y = 56, W-56, 110
        ax.add_patch(Rectangle((bar_x0, bar_y-28), bar_x1-bar_x0, 96, fc="#17202f", ec="#243556", lw=1.2))
        ax.text(bar_x0+22, bar_y+46, "ЭВОЛЮЦИЯ ВО ВРЕМЕНИ  ·  от изоляции к обмену", ha="left", va="center",
                fontsize=9, fontweight=700, color="#e6efff")
        # timeline line
        tl_y = bar_y+18
        ax.plot([bar_x0+22, bar_x1-22], [tl_y, tl_y], color="#2a3f62", lw=2, solid_capstyle="round")
        for tg in [0,5,10,12,15,18,20,25,30]:
            tx = bar_x0+22 + (tg/30)*(bar_x1-44 - (bar_x0+22) +44)  # simplifies to bar_x0+22 + tg/30 * (bar_x1-44)
            # Actually width = bar_x1-22 - (bar_x0+22) = W-84
            tx = bar_x0+22 + (tg/30)*(bar_x1-22 - (bar_x0+22))
            ax.plot([tx, tx], [tl_y-8, tl_y+8], color="#5ac8ff", lw=1.6)
            ax.text(tx, tl_y+22, f"{tg}", ha="center", va="center", fontsize=7, color="#8aa0c8", fontfamily="monospace")
            # progress indicator
            if g == tg:
                ax.add_patch(Circle((tx, tl_y), 5, facecolor="#7af0a0", edgecolor="#0f141c", lw=1.5))
        # quarantine bars
        for q in QUARANTINE:
            until = q["until"]
            x0 = bar_x0+22
            x1 = bar_x0+22 + (until/30)*(bar_x1-22 - (bar_x0+22))
            ax.plot([x0, x1], [tl_y, tl_y], color=q["color"], lw=6, alpha=0.95, solid_capstyle="round")
            ax.add_patch(Circle((x1, tl_y), 7, facecolor=q["color"], edgecolor="#0f141c", lw=2))
            ax.text(x1, tl_y-14, f"burst {q['id']}", ha="center", va="center", fontsize=6, fontweight=700, color=q["color"], fontfamily="monospace")
        ax.text(bar_x0+22, bar_y-10, "0 gen — Q1/Q2/F живут сами, каждые 5 gen кольцо best+diverse → 2 худших соседа. В 12/15/18 — burst проснувшегося острова → всем активным",
                ha="left", va="center", fontsize=7.2, color="#8aa0c8")

        # footer
        ax.text(W/2, 14, "shuttle_sim/ga/{genome,island,fitness,migration,engine}  ·  S→A→V→W→R→H→S every 5 gen  hall min_dist 0.12  ·  pipeline.prepare clip ω ±18%  ·  dashboard: is_isolated → серая иконка",
                ha="center", va="center", fontsize=6.5, color="#5a6b8a", fontfamily="monospace")

        # generation highlight
        if g % 5 == 0 and g != 0:
            ax.text(W/2, 88, f"->  МИГРАЦИЯ  gen {g}  ·  2 мигранта / остров  ·  global shuffle каждые 15", ha="center", va="center",
                    fontsize=8, fontweight=800, color="#7af0a0",
                    bbox=dict(facecolor="#17202f", edgecolor="#7af0a0", boxstyle="round,pad=0.4", alpha=0.95))

        # Save frame to buffer (tostring_rgb removed in matplotlib 3.11)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        # buffer_rgba returns (h,w,4) uint8
        buf = np.asarray(fig.canvas.buffer_rgba())
        # drop alpha, keep RGB
        rgb = buf[:, :, :3].copy()
        frames.append(Image.fromarray(rgb))
        plt.close(fig)

    # Save GIF
    gif_path = ASSETS / "anim-ga-islands.gif"
    # Save with 250ms per frame (4fps), loop infinite, optimize
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=380, loop=0, optimize=False, disposal=2)
    print(f"[gif] wrote {gif_path} ({len(frames)} frames, {gif_path.stat().st_size/1024:.0f} kB)")
    # Also save first frame as png for fallback
    png_path = ASSETS / "fig-ga-islands-preview.png"
    frames[0].save(png_path, optimize=True)
    print(f"[png] wrote {png_path}")

if __name__ == "__main__":
    make_static_svg()
    make_animated_gif()
