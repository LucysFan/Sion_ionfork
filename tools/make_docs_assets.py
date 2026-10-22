"""Generate every image used by README.md from the run artifacts in ``output/``.

Nothing here is hand-drawn fake data: the curves come from the CSV/JSON files that
``python -m shuttle_sim`` wrote.  Only the trap *schematic* is explicitly labelled
as a schematic (electrode silhouettes, not a field solve).

Usage
-----
    python tools/make_docs_assets.py            # all PNG + GIF
    python tools/make_docs_assets.py --only ga  # a subset (wave, traj, freq,
                                                # valid, robust, ga, dashboard, anim)

Outputs go to ``assets/`` at the repository root.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"
RUN = ROOT / "output" / "shuttle_reference"
GA = ROOT / "output" / "ga_demo"

# ----------------------------------------------------------------------------
# shared styling — same palette as shuttle_sim/dashboard/static/index.html
# ----------------------------------------------------------------------------
BG = "#0f141c"
PANEL = "#17202f"
PANEL2 = "#1e2d47"
TEXT = "#e6efff"
MUTED = "#8aa0c8"
GRID = "#24324d"
ACCENT = "#5ac8ff"
ACCENT2 = "#7af0a0"
WARN = "#ffbe55"
BAD = "#ff6b7a"
ISLAND_COLORS = {
    "S": "#5ac8ff",
    "A": "#7af0a0",
    "V": "#ffbe55",
    "R": "#ff6b7a",
    "H": "#c79bff",
    "F": "#6ee7d5",
}

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "savefig.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "text.color": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "grid.alpha": 0.55,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.frameon": False,
        "legend.labelcolor": TEXT,
        "figure.autolayout": False,
    }
)


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.grid(True, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _save(fig, name, dpi=150):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=dpi, facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}  ({path.stat().st_size/1024:.0f} kB)")


def _selected_metrics(run_dir):
    """Metrics of the candidate the pipeline actually selected in that run."""
    status = json.loads((Path(run_dir) / "status.json").read_text())
    comp = json.loads((Path(run_dir) / "comparison.json").read_text())
    key = {
        "sta": "sta_delivered",
        "smooth": "smooth_delivered",
        "aware": "hardware_aware_candidate_delivered",
        "unaware": "hardware_unaware_optimized_delivered",
    }.get(status.get("selected"), f"{status.get('selected')}_delivered")
    if key not in comp:  # tolerate future naming
        key = next(k for k in comp if k.endswith("_delivered"))
    return comp[key], status, key


def _read_csv(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    cols = {}
    for key in rows[0]:
        cols[key] = np.array([float(r[key]) for r in rows])
    return cols


# ----------------------------------------------------------------------------
# 1. waveforms: ideal commands vs delivered (DAC + RC + gain + delay)
# ----------------------------------------------------------------------------
def fig_waveforms():
    ideal = _read_csv(RUN / "ideal_waveform.csv")
    deliv = _read_csv(RUN / "delivered_waveform.csv")
    dist = json.loads((RUN / "distortion.json").read_text())

    t_us = ideal["time_s"] * 1e6
    channels = [c for c in ideal if c.startswith("dc[")]
    spans = [(k, ideal[k].max() - ideal[k].min()) for k in channels]
    picked = [k for k, _ in sorted(spans, key=lambda kv: -kv[1])[:4]]

    fig = plt.figure(figsize=(12.6, 7.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.8, 1], hspace=0.55)

    ax = fig.add_subplot(gs[0])
    # the absolute levels (~±10 V) hide the motion: show the *change* w.r.t. t=0
    for key in channels:
        d0 = ideal[key] - ideal[key][0]
        if key in picked:
            continue
        ax.plot(t_us, d0, color=MUTED, lw=0.7, alpha=0.35)
    for i, key in enumerate(picked):
        color = plt.cm.viridis(0.18 + 0.62 * i / max(1, len(picked) - 1))
        ax.plot(t_us, ideal[key] - ideal[key][0], color=color, lw=2.1, label=f"команда {key}")
    _style(
        ax,
        "Команда на электроды: изменение напряжения относительно старта ΔV(t)  "
        "(15 каналов, 4 несущих выделены)",
        "время, мкс",
        "ΔV, В",
    )
    ax.legend(ncol=4, fontsize=8.8, loc="lower center", framealpha=0.0)
    ax.axhline(0, color=GRID, lw=0.8)
    turn = t_us[np.argmax(np.abs(ideal[picked[0]] - ideal[picked[0]][0]))]
    ax.axvline(turn, color=WARN, lw=1.0, ls=":", alpha=0.85)
    ax.annotate(
        "разворот: V перекидывается, ион останавливается",
        xy=(turn, ax.get_ylim()[1] * 0.80),
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        fontsize=9,
        color=WARN,
    )

    ax2 = fig.add_subplot(gs[1])
    ch_max = np.asarray(dist["channel_max_deviation_V"])
    idx = np.arange(len(ch_max))
    colors = [ACCENT if c not in picked else ACCENT2 for c in dist_channel_names(len(ch_max))]
    ax2.bar(idx, np.asarray(ch_max) * 1e3, color=colors, width=0.66, alpha=0.92)
    ax2.axhline(dist["rms_deviation_V"] * 1e3, color=WARN, lw=1.4, ls="--")
    ax2.text(
        len(ch_max) - 0.4,
        dist["rms_deviation_V"] * 1e3 * 1.08,
        f"rms {dist['rms_deviation_V']*1e3:.1f} мВ",
        color=WARN,
        ha="right",
        fontsize=9,
    )
    _style(
        ax2,
        f"Сколько съедает железо: |ΔV| max {dist['max_deviation_V']*1e3:.0f} мВ, "
        f"LSB {dist['lsb_V']*1e6:.0f} мкВ, клиппингов {dist['clipping_events']} (изменение масштаба ×1000)",
        "канал DC",
        "|ΔV|, мВ",
    )
    ax2.set_xticks(idx)
    ax2.set_xticklabels([k.replace("dc[", "dc").replace("]", "") for k in dist_channel_names(len(ch_max))], fontsize=8)
    _save(fig, "fig-waveforms.png")


def dist_channel_names(n):
    return [f"dc[{i}]" for i in range(n)]


# ----------------------------------------------------------------------------
# 2. trajectory: the ion actually travels out and back
# ----------------------------------------------------------------------------
def fig_trajectory():
    path = _read_csv(RUN / "delivered_path.csv")
    task = json.loads((RUN / "transport_task.json").read_text())
    diag = json.loads((RUN / "route_diagnostics.json").read_text())
    comp, _status, _key = _selected_metrics(RUN)
    deliv_diag = json.loads((RUN / "delivered_local_diagnostics.json").read_text())

    t_us = path["time_s"] * 1e6
    x_um = path["x_m"] * 1e6
    z_um = path["z_m"] * 1e6
    v_mm_s = np.gradient(path["x_m"], path["time_s"]) * 1e3
    t_d = np.array([d["time_s"] for d in deliv_diag]) * 1e6
    r_del_x = np.array([d["position_m"][0] for d in deliv_diag]) * 1e6
    f_min = np.array([min(d["omega_rad_s"]) for d in deliv_diag]) / (2 * np.pi) / 1e3
    dist_um = task["distance_m"] * 1e6

    fig = plt.figure(figsize=(12.6, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1], hspace=0.46, wspace=0.24)

    ax = fig.add_subplot(gs[0, :])
    ax.fill_between(t_us, 0, x_um, color=ACCENT, alpha=0.13, lw=0)
    ax.plot(t_us, x_um, color=ACCENT, lw=2.4, label="реализованный путь иона")
    ax.plot(
        t_d,
        r_del_x,
        color=ACCENT2,
        lw=1.1,
        ls="--",
        alpha=0.95,
        label="центр восстановленной ямы (ROM)",
    )
    ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    ax.axhline(dist_um, color=MUTED, lw=0.8, ls=":")
    ax.text(t_us.max(), dist_um + 1.1, "цель 30 мкм", color=MUTED, fontsize=9, ha="right")
    ax.text(-0.4, 1.4, "старт", color=MUTED, fontsize=9, ha="left")
    ax.set_ylim(-4, dist_um + 6)
    ax.set_xlim(0, t_us.max())
    _style(
        ax,
        f"Реализованный маршрут: {task['protocol']}, {dist_um:.0f} мкм за "
        f"{task['duration_s']*1e6:.0f} мкс на плечо",
        "время, мкс",
        "x, мкм",
    )
    ax.text(
        0.015,
        0.955,
        f"n_total = {comp['n_total']:.4f} фонона   ·   ошибка позиции "
        f"{comp['position_error_m']*1e9:.2f} нм   ·   пауза {task['hold_s']*1e6:.0f} мкс",
        transform=ax.transAxes,
        fontsize=9.6,
        color=ACCENT2,
        va="top",
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.90), ncol=1, fontsize=9.0)
    ax.annotate(
        "разворот",
        xy=(t_us[np.argmax(x_um)], dist_um),
        xytext=(0, -34),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        color=WARN,
        arrowprops=dict(arrowstyle="->", color=WARN, lw=1.0),
    )

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t_us, v_mm_s, color=WARN, lw=1.8)
    _style(
        ax2,
        f"Скорость иона: пик {np.abs(v_mm_s).max()/1e3:.2f} м/с",
        "время, мкс",
        "v, мм/с",
    )
    ax2.axhline(0, color=GRID, lw=0.8)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t_d, f_min, color=ACCENT2, lw=1.9)
    ax3.axhline(400, color=BAD, ls="--", lw=1.2)
    ax3.text(t_d.max(), 400 + 12, "порог 400 кГц", color=BAD, ha="right", fontsize=9)
    ax3.set_ylim(350, max(f_min.max() * 1.18, 780))
    _style(
        ax3,
        f"Конфайнмент: минимум {f_min.min():.0f} кГц > порога",
        "время, мкс",
        "ω₁/2π, кГц",
    )
    _save(fig, "fig-trajectory.png")

    # frequencies along the route, separate figure
    fig, ax = plt.subplots(figsize=(12.6, 3.6))
    s_route = np.array([d["s"] for d in diag])
    freqs = np.array([d["frequencies_hz"] for d in diag]) / 1e3
    for i in range(3):
        ax.plot(s_route, freqs[:, i], color=[ACCENT, ACCENT2, WARN][i], lw=2.0, label=["ω₁ (слабая ось)", "ω₂", "ω₃"][i])
    ax.axhline(freqs[:, 0].min(), color=BAD, ls="--", lw=1.2)
    ax.text(s_route[0], freqs[:, 0].min() + 45, f"минимум по маршруту {freqs[:, 0].min():.0f} кГц", color=BAD, fontsize=9)
    _style(
        ax,
        "Секулярные частоты вдоль маршрута — ROM-диагностика по 41 опорной точке",
        "s (доля маршрута)",
        "частота, кГц",
    )
    ax.legend(ncol=3, fontsize=9, loc="upper right")
    _save(fig, "fig-frequencies.png")


# ----------------------------------------------------------------------------
# 3. validation: cheap ROM vs exact nodes vs independent integrator
# ----------------------------------------------------------------------------
def fig_validation():
    val = json.loads((RUN / "validation.json").read_text())
    order = [
        ("cheap_search_model", "дешёвая ROM\n(поиск)"),
        ("midpoint_exact_fine", "точно\n321 узел"),
        ("refined_1281_nodes_16_substeps", "уточнённо\n1281 узел"),
        ("independent_dop853", "DOP853\nинтегратор"),
    ]
    rows = [(lab, val[k]["n_total"]) for k, lab in order if k in val]
    if not rows:
        rows = [(k, v["n_total"]) for k, v in val.items() if isinstance(v, dict) and "n_total" in v]

    fig, (ax, ax2) = plt.subplots(
        1,
        2,
        figsize=(12.6, 4.2),
        gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.26},
    )

    labels = [r[0] for r in rows]
    values = np.array([r[1] for r in rows])
    bars = ax.bar(range(len(rows)), values, color=[ACCENT, ACCENT2, ACCENT2, WARN][: len(rows)], width=0.62)
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + values.max() * 0.035,
            f"{v:.5f}",
            ha="center",
            fontsize=10,
            color=TEXT,
        )
    ax.set_ylim(0, values.max() * 1.28)
    _style(ax, "Один прогон, четыре уровня точности", "", "n_total, фононы")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=9)

    # deltas vs the cheap search model: what actually matters for trust
    ref = dict((k, val[k]["n_total"]) for k, _ in order if k in val).get("cheap_search_model", values[0])
    names, deltas, colors = [], [], []
    for key, short in (
        ("cheap_vs_validated_delta_n", "ROM → точный"),
        ("refinement_delta_n", "уточнение сетки"),
        ("coarse_to_fine_delta_n", "грубая → тонкая"),
        ("integrator_delta_n", "смена интегратора"),
    ):
        if key in val:
            names.append(short)
            deltas.append(abs(val[key]))
            colors.append(BAD if abs(val[key]) > val.get("n_tolerance", 0.01) else ACCENT2)
    y = np.arange(len(names))
    ax2.barh(y, deltas, color=colors, height=0.55)
    ax2.set_yticks(y)
    ax2.set_yticklabels(names, fontsize=9.5)
    ax2.invert_yaxis()
    tol = val.get("n_tolerance", val.get("validation_tolerance", 0.01))
    ax2.axvline(tol, color=BAD, ls="--", lw=1.2)
    ax2.text(tol, len(names) - 0.35, f"допуск {tol:g}", color=BAD, fontsize=9, ha="center")
    for yi, d in zip(y, deltas):
        ax2.text(d + tol * 0.03, yi, f"{d:.1e}", va="center", fontsize=9, color=TEXT)
    ax2.set_xlim(0, max(max(deltas) * 1.45, tol * 1.3))
    _style(
        ax2,
        "Дрейф ответа при смене модели (допуск 0.01 фонона)",
        "|Δn|, фононы",
        "",
    )
    ax2.set_ylim(len(names) - 0.2, -0.7)
    _save(fig, "fig-validation.png")


# ----------------------------------------------------------------------------
# 4. robustness: Monte-Carlo over declared hardware uncertainties
# ----------------------------------------------------------------------------
def fig_robust():
    rob = json.loads((RUN / "robust.json").read_text())
    rows = rob["rows"]
    n = np.array([r["n_total"] for r in rows])
    idx = np.arange(len(n))
    order = np.argsort(n)
    n = n[order]
    rows = [rows[i] for i in order]
    budget = 0.5

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12.6, 4.1), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.26}
    )

    colors = [BAD if v > 0.25 else (WARN if v > 0.05 else ACCENT2) for v in n]
    ax.bar(idx, n, color=colors, width=0.6)
    for i, v in enumerate(n):
        ax.text(i, v + budget * 0.028, f"{v:.4f}", ha="center", fontsize=9, color=TEXT)
    ax.set_xticks(idx)
    ax.set_xticklabels(
        [f"#{i}\n×{row['frequency_scale']:.3f}\nω" for i, row in enumerate(rows)], fontsize=8
    )
    ax.axhline(rob["worst_n"], color=BAD, ls="--", lw=1.2)
    ax.axhline(budget, color=TEXT, ls=":", lw=1.2)
    ax.text(-0.45, budget * 1.03, f"бюджет n ≤ {budget}", color=TEXT, fontsize=9, ha="left")
    ax.set_ylim(0, max(budget, n.max()) * 1.12)
    _style(
        ax,
        f"Робастность: {len(n)} сценариев шума, принято {rob['pass_fraction']*100:.0f}%",
        "сценарий (sorted по n) и масштаб ω",
        "n_total, фононы",
    )

    labels = ["mean", "CVaR 20%", "worst"]
    vals = [rob["mean_n"], rob["cvar_20pct_n"], rob["worst_n"]]
    y = np.arange(len(labels))
    ax2.barh(y, vals, color=[ACCENT2, WARN, BAD], height=0.55)
    for yi, v in zip(y, vals):
        ax2.text(v + budget * 0.015, yi - 0.30, f"{v:.4f}  ({v/budget*100:.0f}% бюджета)", va="center", fontsize=9, color=TEXT)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels)
    ax2.invert_yaxis()
    ax2.axvline(budget, color=TEXT, ls="--", lw=1.1)
    ax2.set_xlim(0, budget * 1.0)
    _style(ax2, "Запас до бюджета 0.5", "n_total", "")
    _save(fig, "fig-robust.png")


# ----------------------------------------------------------------------------
# 5. island GA: evolution, diversity, hall of fame
# ----------------------------------------------------------------------------
def _ga_history():
    frames = [json.loads(p.read_text()) for p in sorted((GA / "history").glob("gen_*.json"))]
    return frames


def fig_ga():
    frames = _ga_history()
    gens = [f["global_generation"] for f in frames]
    strategies = [i["strategy"] for i in frames[0]["islands"]]

    fig = plt.figure(figsize=(12.6, 7.4))
    gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.22)

    ax = fig.add_subplot(gs[0, 0])
    for si, strat in enumerate(strategies):
        best = [f["islands"][si]["best_fitness"] for f in frames]
        ax.semilogy(gens, best, marker="o", ms=3.5, lw=1.9, color=ISLAND_COLORS.get(strat, ACCENT), label=f"остров {strat}")
    _style(ax, "Лучший fitness по поколениям", "поколение", "fitness (лог)")
    ax.legend(ncol=3, fontsize=8.6, loc="upper right")

    ax2 = fig.add_subplot(gs[0, 1])
    for si, strat in enumerate(strategies):
        div = [f["islands"][si]["diversity"] for f in frames]
        ax2.plot(gens, div, marker="o", ms=3.5, lw=1.9, color=ISLAND_COLORS.get(strat, ACCENT), label=strat)
    _style(ax2, "Разнообразие популяции (θ-расстояние)", "поколение", "diversity")
    ax2.legend(ncol=5, fontsize=8.6, loc="lower right")

    hall = json.loads((GA / "ga_hall.json").read_text())
    ax3 = fig.add_subplot(gs[1, 0])
    pts = [(h["metrics"]["n_total"], h["fitness"], h["island"]) for h in hall if "metrics" in h and "n_total" in h["metrics"]]
    for strat in sorted({p[2] for p in pts}):
        sel = [(p[0], p[1]) for p in pts if p[2] == strat]
        ax3.scatter(
            [s[0] for s in sel],
            [s[1] for s in sel],
            s=80,
            color=ISLAND_COLORS.get(strat, ACCENT),
            edgecolor=BG,
            lw=1.3,
            label=strat,
            zorder=3,
        )
    _style(ax3, "Зал славы: чем левее и ниже, тем лучше", "n_total, фононы", "fitness")
    # Pareto frontier (non-dominated points)
    pts_sorted = sorted(pts, key=lambda p: p[0])
    frontier, best_f = [], np.inf
    for x_, y_, _s in pts_sorted:
        if y_ < best_f:
            frontier.append((x_, y_))
            best_f = y_
    if len(frontier) > 1:
        ax3.plot(
            [p[0] for p in frontier],
            [p[1] for p in frontier],
            color=ACCENT2,
            lw=1.3,
            ls="--",
            alpha=0.8,
            zorder=2,
            label="фронт",
        )
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlim(8e-4, 1.2e-2)
    ax3.set_ylim(1.5e-3, 2e-1)
    ax3.set_xticks([1e-3, 2e-3, 5e-3, 1e-2])
    ax3.set_xticklabels(["0.001", "0.002", "0.005", "0.01"])
    ax3.set_yticks([2e-3, 5e-3, 1e-2, 5e-2, 1e-1])
    ax3.set_yticklabels(["0.002", "0.005", "0.01", "0.05", "0.1"])
    ax3.legend(ncol=6, fontsize=8.2, loc="upper right")

    # theta "gene strip": 10 numbers, 2 decimals, readable cells
    best = json.loads((GA / "ga_best.json").read_text())
    theta = np.asarray(best["theta"])[:10]
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    names = ["warp 1", "warp 2"] + [f"SVD {i}" for i in range(6)] + ["ω a", "ω b"]
    n = len(theta)
    lim = max(0.4, float(np.abs(theta).max()) * 1.1)
    cmap = LinearSegmentedColormap.from_list("signed", [BAD, PANEL, ACCENT2])
    for i, val in enumerate(theta):
        frac = 0.5 + 0.5 * val / lim
        ax4.add_patch(plt.Rectangle((i, 0), 0.9, 1.0, facecolor=cmap(frac), edgecolor=GRID, lw=1.0))
        strong = abs(val) > lim * 0.32
        ax4.text(
            i + 0.45, 0.62, f"{val:+.2f}", ha="center", va="center", fontsize=8.6,
            color="#06121c" if strong else TEXT, fontweight="bold",
        )
        ax4.text(
            i + 0.45, 0.24, names[i], ha="center", va="center", fontsize=6.6,
            color="#06121c" if strong else MUTED,
        )
    ax4.set_xlim(-0.2, n)
    ax4.set_ylim(-0.35, 1.45)
    ax4.text(0, 1.30, "Геном θ* (10 чисел вместо 200 отсчётов V)", fontsize=12.5, fontweight="bold", color=TEXT)
    ax4.text(
        0,
        1.05,
        f"n_total {best['metrics']['n_total']:.5f}   ·   ошибка {best['metrics']['position_error_m']*1e9:.2f} нм   ·   "
        f"fitness {best['fitness']:.5f}   ·   остров {best.get('thetas_source', 'V')}",
        fontsize=8.8,
        color=ACCENT2,
    )
    _save(fig, "fig-ga.png")


# ----------------------------------------------------------------------------
# 6. dashboard-styled snapshot of the GA panels (static render of live data)
# ----------------------------------------------------------------------------
def fig_dashboard():
    frames = _ga_history()
    state = json.loads((GA / "state.json").read_text())
    gens = [f["global_generation"] for f in frames]
    strategies = [i["strategy"] for i in frames[0]["islands"]]

    fig = plt.figure(figsize=(13.2, 8.0))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(
        2, 3, hspace=0.55, wspace=0.30, left=0.06, right=0.985, top=0.775, bottom=0.075
    )

    # --- header ------------------------------------------------------------
    fig.text(0.06, 0.945, "Island GA — живой дашборд шаттлинга", fontsize=19, fontweight="bold", color=TEXT)
    fig.text(
        0.06,
        0.905,
        f"поколение {state['global_generation']}/{state['total_generations']}   ·   "
        f"{len(strategies)} островов   ·   {state['elapsed_s']:.0f} с   ·   "
        f"лучший fitness {min(i['best_fitness'] for i in state['islands']):.5f}",
        fontsize=10.5,
        color=MUTED,
    )
    fig.text(0.06, 0.868, "источник данных: output/ga_demo/state.json + history/gen_*.json", fontsize=9, color=MUTED)
    # fake "live" pill, same accent as the real dashboard
    fig.patches.append(
        plt.Rectangle((0.86, 0.926), 0.125, 0.043, transform=fig.transFigure, facecolor=PANEL2, edgecolor=GRID, lw=1)
    )
    fig.text(0.868, 0.940, "●  polling 500 мс", fontsize=9, color=ACCENT2, va="center")

    # --- fitness -----------------------------------------------------------
    ax = fig.add_subplot(gs[0, :2])
    for si, strat in enumerate(strategies):
        ax.semilogy(
            gens,
            [f["islands"][si]["best_fitness"] for f in frames],
            marker="o",
            ms=3,
            lw=2.0,
            color=ISLAND_COLORS.get(strat, ACCENT),
            label=f"остров {strat}",
        )
    _style(ax, "fitness лучшей особи (меньше — лучше)", "поколение", "fitness (лог)")
    ax.legend(ncol=5, fontsize=8.4, loc="upper right")

    # --- per-island bar ----------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 2])
    labels = [i["strategy"] for i in state["islands"]]
    vals = [i["best_fitness"] for i in state["islands"]]
    ax2.bar(range(len(vals)), vals, color=[ISLAND_COLORS.get(l, ACCENT) for l in labels], width=0.66)
    ax2.set_yscale("log")
    ax2.set_xticks(range(len(vals)))
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_yticks([1e-2, 1e-1])
    ax2.set_yticklabels(["0.01", "0.1"])
    _style(ax2, "лучший fitness / остров", "", "")

    # --- diversity ---------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    for si, strat in enumerate(strategies):
        ax3.plot(
            gens,
            [f["islands"][si]["diversity"] for f in frames],
            lw=1.8,
            color=ISLAND_COLORS.get(strat, ACCENT),
        )
    _style(ax3, "разнообразие (θ-расстояние)", "поколение", "")

    # --- Pareto ------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    hall = json.loads((GA / "ga_hall.json").read_text())
    pts = [(h["metrics"]["n_total"], h["fitness"], h["island"]) for h in hall if "metrics" in h]
    for strat in sorted({p[2] for p in pts}):
        sel = [p for p in pts if p[2] == strat]
        ax4.scatter(
            [s[0] for s in sel],
            [s[1] for s in sel],
            s=48,
            color=ISLAND_COLORS.get(strat, ACCENT),
            edgecolor=BG,
            lw=1.1,
            zorder=3,
        )
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlim(8e-4, 1.2e-2)
    ax4.set_ylim(1.5e-3, 2e-1)
    ax4.set_xticks([1e-3, 2e-3, 5e-3, 1e-2])
    ax4.set_xticklabels(["0.001", "0.002", "0.005", "0.01"], fontsize=8.5)
    ax4.set_yticks([2e-3, 1e-2, 1e-1])
    ax4.set_yticklabels(["0.002", "0.01", "0.1"], fontsize=8.5)
    _style(ax4, "Парето: n_total vs fitness", "n_total, фононы", "fitness")

    # --- genome ------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 2])
    best = json.loads((GA / "ga_best.json").read_text())
    theta = np.asarray(best["theta"])[:10]
    ax5.barh(range(len(theta)), theta, color=[ACCENT if v >= 0 else BAD for v in theta], height=0.62)
    ax5.set_yticks(range(len(theta)))
    ax5.set_yticklabels([f"θ{i}" for i in range(len(theta))], fontsize=8)
    ax5.invert_yaxis()
    ax5.axvline(0, color=MUTED, lw=0.8)
    _style(ax5, "геном θ* (зал славы)", "", "")

    _save(fig, "fig-dashboard.png", dpi=145)


# ----------------------------------------------------------------------------
# 7. animation: ion travelling in the trap while the DAC waveform plays
# ----------------------------------------------------------------------------
def fig_animation(n_frames=80, fps=13):
    """Ion travelling in the trap while the delivered DAC waveform plays.

    The GIF is the README "живая" картинка: слева ион в ловушке, справа
    доставленные напряжения и x(t).  Данные — те же CSV, что и у графиков.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    path = _read_csv(RUN / "delivered_path.csv")
    deliv = _read_csv(RUN / "delivered_waveform.csv")
    diag = json.loads((RUN / "delivered_local_diagnostics.json").read_text())
    task = json.loads((RUN / "transport_task.json").read_text())
    comp, _status, _key = _selected_metrics(RUN)

    t = path["time_s"]
    x = path["x_m"] * 1e6
    z = path["z_m"] * 1e6
    f_min = np.interp(
        t,
        [d["time_s"] for d in diag],
        [min(d["omega_rad_s"]) for d in diag],
    ) / (2 * np.pi) / 1e3

    chans = [c for c in deliv if c.startswith("dc[")]
    spans = [(k, deliv[k].max() - deliv[k].min()) for k in chans]
    picked = [k for k, _ in sorted(spans, key=lambda kv: -kv[1])[:4]]

    dist = task["distance_m"] * 1e6
    frames_idx = np.linspace(0, len(t) - 1, n_frames).astype(int)
    t_max_us = t.max() * 1e6

    fig = plt.figure(figsize=(11.6, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], hspace=0.52, wspace=0.26)

    # ---------- trap top view ----------
    axT = fig.add_subplot(gs[:, 0])
    axT.set_facecolor("#0b1220")
    axT.set_xlim(-18, 46)
    axT.set_ylim(-26, 26)
    axT.set_aspect("equal", adjustable="datalim")
    axT.axis("off")
    axT.set_title("Вид сверху: ион в яме (схема)", loc="left")

    axT.add_patch(plt.Rectangle((-16, 13), 62, 11, facecolor=PANEL2, edgecolor=GRID, lw=1))
    axT.add_patch(plt.Rectangle((-16, -24), 62, 11, facecolor=PANEL2, edgecolor=GRID, lw=1))
    for i in range(8):
        x0 = -15 + i * 7.6
        axT.add_patch(plt.Rectangle((x0, 2.6), 6.6, 7.2, facecolor=PANEL, edgecolor=GRID, lw=1))
        axT.add_patch(plt.Rectangle((x0, -9.8), 6.6, 7.2, facecolor=PANEL, edgecolor=GRID, lw=1))
    axT.text(15, 18.5, "RF-электроды", color=MUTED, ha="center", fontsize=9, fontweight="bold")
    axT.text(15, -18.5, "RF-электроды", color=MUTED, ha="center", fontsize=9, fontweight="bold")
    axT.text(15, 6.2, "DC-сегменты", color=MUTED, ha="center", fontsize=9)
    axT.plot([0, dist], [0, 0], color=GRID, lw=1.2, ls=":")
    axT.text(0, -3.4, "старт", color=MUTED, fontsize=8.5, ha="center")
    axT.text(dist, -3.4, "цель", color=MUTED, fontsize=8.5, ha="center")

    trail, = axT.plot([], [], color=ACCENT, lw=1.6, alpha=0.75)
    well, = axT.plot([], [], color=ACCENT2, lw=2.6, alpha=0.9)
    ion, = axT.plot([], [], "o", ms=13, color="#ffffff", markeredgecolor=ACCENT, markeredgewidth=2.6, zorder=6)
    glow = plt.Circle((0, 0), 3.4, color=ACCENT2, alpha=0.16, zorder=4)
    axT.add_patch(glow)
    label = axT.text(0.0, 1.0, "", transform=axT.transAxes, va="top", fontsize=10.5, color=TEXT)

    # ---------- voltages ----------
    axV = fig.add_subplot(gs[0, 1])
    t_us_d = deliv["time_s"] * 1e6
    colors = [plt.cm.viridis(0.18 + 0.62 * i / max(1, len(picked) - 1)) for i in range(len(picked))]
    for color, key in zip(colors, picked):
        axV.plot(t_us_d, deliv[key] - deliv[key][0], color=color, lw=1.2, alpha=0.55)
    cursor = axV.axvline(0, color=ACCENT, lw=0.9, ls="--", alpha=0.9)
    dots = [axV.plot([], [], "o", ms=5.5, color=c, zorder=5)[0] for c in colors]
    _style(axV, "ΔV доставленных напряжений (4 несущих канала)", "время, мкс", "ΔV, В")
    axV.set_xlim(0, t_max_us)

    # ---------- x(t) ----------
    axP = fig.add_subplot(gs[1, 1])
    axP.plot(t * 1e6, x, color=GRID, lw=1.3)
    prog, = axP.plot([], [], color=ACCENT, lw=2.4)
    posdot, = axP.plot([], [], "o", ms=7, color="#ffffff", markeredgecolor=ACCENT, markeredgewidth=2, zorder=5)
    _style(axP, "x(t) иона", "время, мкс", "x, мкм")
    axP.set_xlim(0, t_max_us)
    axP.set_ylim(-4, dist + 5)
    info = axP.text(0.975, 0.06, "", transform=axP.transAxes, fontsize=9.6, color=ACCENT2, va="bottom", ha="right")

    def update(k):
        i = frames_idx[k]
        tt = t[i]
        xx = x[i]
        trail.set_data(x[: i + 1], np.zeros(i + 1))
        well.set_data([xx - 5.5, xx + 5.5], [0, 0])
        ion.set_data([xx], [0])
        glow.center = (xx, 0)
        label.set_text(f"t = {tt*1e6:5.2f} мкс\nx = {xx:5.2f} мкм   z = {z[i]:.1f} мкм")
        cursor.set_xdata([tt * 1e6, tt * 1e6])
        for j, key in enumerate(picked):
            dots[j].set_data([tt * 1e6], [np.interp(tt, deliv["time_s"], deliv[key]) - deliv[key][0]])
        prog.set_data(t[: i + 1] * 1e6, x[: i + 1])
        posdot.set_data([tt * 1e6], [xx])
        info.set_text(f"f_min = {f_min[i]:.0f} кГц\nn_total = {comp['n_total']:.4f}")
        return ()

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    OUT.mkdir(parents=True, exist_ok=True)
    gif = OUT / "anim-shuttle.gif"
    anim.save(gif, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    print(f"  wrote {gif.relative_to(ROOT)}  ({gif.stat().st_size/1024/1024:.2f} MB)")


# ----------------------------------------------------------------------------
# 8. cheap "one number per panel" strip for the very top of the README
# ----------------------------------------------------------------------------
def fig_scorecard():
    order = [
        ("shuttle_reference", "reference\nround-trip 30 мкм"),
        ("shuttle_oneway", "one_way\n30 мкм"),
        ("shuttle_repeated", "repeated ×2\nфильтр не сбрасывается"),
        ("shuttle_gds", "GDS-геометрия\n17 каналов"),
        ("shuttle_variable_freq", "variable ω\n±15%"),
    ]
    cards = []
    for name, label in order:
        d = ROOT / "output" / name
        if not (d / "comparison.json").exists():
            continue
        metrics, status, key = _selected_metrics(d)
        cards.append((label, metrics, status, key))

    fig, axes = plt.subplots(1, len(cards), figsize=(13.4, 2.6))
    if len(cards) == 1:
        axes = [axes]
    for ax, (label, m, status, key) in zip(axes, cards):
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=PANEL, edgecolor=GRID, lw=1.2))
        ok = bool(status.get("simulation_acceptance_passed"))
        ax.add_patch(plt.Rectangle((0, 0.965), 1, 0.035, transform=ax.transAxes,
                                   facecolor=ACCENT2 if ok else BAD, lw=0))
        ax.text(0.07, 0.78, label, fontsize=8.8, color=MUTED, transform=ax.transAxes, va="top", linespacing=1.35)
        n = m["n_total"]
        ax.text(0.07, 0.40, f"{n:.4f}", fontsize=23, fontweight="bold", color=TEXT, transform=ax.transAxes)
        ax.text(0.07, 0.245, "фонона", fontsize=8.6, color=MUTED, transform=ax.transAxes)
        ax.text(
            0.07,
            0.10,
            f"позиция {m['position_error_m']*1e9:.2f} нм  ·  {'принят' if ok else 'отказ'}",
            fontsize=8.2,
            color=ACCENT2 if ok else BAD,
            transform=ax.transAxes,
        )
    _save(fig, "fig-scorecard.png", dpi=160)


# ----------------------------------------------------------------------------
# 9. architecture: the eight mandatory layers + what each one writes
# ----------------------------------------------------------------------------
def fig_architecture():
    """Static diagram of the pipeline: one box per layer, artifacts underneath."""
    import matplotlib.patches as mpatches

    layers = [
        ("1 · Поле", "field.py", "SionTrap, Local\nпсевдопотенциал + DC", "trap_snapshot.json", ACCENT),
        ("2 · Управление", "control.py", "θ ∈ ℝ⁸: warp + SVD\nfixed / variable ω", "config.json", "#6ee7d5"),
        ("3 · Инверсия", "voltage.py", "3 силы + 1 кривизна\nSLSQP, bounds", "route_diagnostics.json", ACCENT2),
        ("4 · Железо", "hardware.py", "DAC, ZOH, RC, gain,\ndelay, slew", "delivered_waveform.csv", WARN),
        ("5 · Динамика", "dynamics.py", "3D гаусс, exact/DOP853\nковариация", "delivered_path.csv", "#ffa94d"),
        ("6 · Робастность", "pipeline.robust", "ансамбль сценариев\nmean/worst/CVaR", "robust.json", BAD),
        ("7 · Калибровка", "calibration.py", "gain/offset/фильтр\n→ update_model", "calibration_*.json", "#f783ac"),
        ("8 · Closed-loop", "closed_loop.py", "8 параметров,\nSafetyLayer, бюджет", "closed_loop_*.json", "#c79bff"),
    ]

    fig = plt.figure(figsize=(13.0, 6.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    ax.set_facecolor(BG)

    ax.text(0.35, 5.95, "Восемь слоёв пайплайна: у каждого свой модуль, вход и артефакт", fontsize=15, fontweight="bold", color=TEXT)
    ax.text(
        0.35,
        5.55,
        "ни один слой не спрятан в ноутбуке: всё, что влияет на результат, лежит в отдельном файле с явным API",
        fontsize=10,
        color=MUTED,
    )

    # snake layout: 4 in the top row, 4 in the bottom row (right -> left)
    positions = []
    for i in range(4):
        positions.append((0.35 + i * 3.12, 3.55))
    for i in range(4):
        positions.append((0.35 + (3 - i) * 3.12, 1.30))

    w, h = 2.85, 1.55
    for i, ((name, mod, desc, artifact, color), (x, y)) in enumerate(zip(layers, positions)):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                facecolor=PANEL,
                edgecolor=color,
                lw=1.6,
            )
        )
        ax.add_patch(plt.Rectangle((x, y + h - 0.09), w, 0.09, facecolor=color, alpha=0.9))
        ax.text(x + 0.14, y + h - 0.34, name, fontsize=11.4, fontweight="bold", color=TEXT)
        ax.text(x + 0.14, y + h - 0.62, mod, fontsize=8.6, color=color, family="DejaVu Sans Mono")
        ax.text(x + 0.14, y + 0.56, desc, fontsize=8.6, color=MUTED, va="top", linespacing=1.35)
        ax.text(x + 0.14, y + 0.20, "→ " + artifact, fontsize=7.8, color=ACCENT2, va="top", family="DejaVu Sans Mono")

    arrow = dict(arrowstyle="-|>", color=MUTED, lw=1.4, shrinkA=0, shrinkB=0)
    for i in range(3):
        x0 = positions[i][0] + w
        ax.annotate("", xy=(x0 + 0.24, positions[i][1] + h / 2), xytext=(x0 + 0.03, positions[i][1] + h / 2), arrowprops=arrow)
    # down from 4 to 5
    ax.annotate(
        "",
        xy=(positions[4][0] + w / 2, positions[4][1] + h + 0.02),
        xytext=(positions[3][0] + w / 2, positions[3][1] - 0.02),
        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4, connectionstyle="arc3,rad=0.25"),
    )
    for i in range(4, 7):
        x0 = positions[i][0]
        ax.annotate("", xy=(x0 - 0.24, positions[i][1] + h / 2), xytext=(x0 - 0.03, positions[i][1] + h / 2), arrowprops=arrow)

    ax.text(
        0.35,
        0.72,
        "сквозные слои:  safety.py (коды отказа, fallback θ=0)   ·   observables.py (n → sideband → loss)",
        fontsize=9.2,
        color=MUTED,
    )
    ax.text(
        0.35,
        0.40,
        "всё в СИ: метры, секунды, герцы, вольты, кг  ·  ни одного «виртуального score» — только V(t) на электродах",
        fontsize=9.2,
        color=MUTED,
    )
    _save(fig, "fig-architecture.png", dpi=150)


# ----------------------------------------------------------------------------
# 10. GDS layout: the real polygons of configs/generated_fivewire.gds
# ----------------------------------------------------------------------------
def fig_gds():
    import sys

    sys.path.insert(0, str(ROOT / "tools"))
    from gds_reader import read_polygons

    # the shipped example file has a placeholder UNITS record -> work in raw
    # database units (1e6 units == 1 on the axis) and show only the topology
    polys = read_polygons(ROOT / "configs" / "generated_fivewire.gds", scale=1e-6)
    cfg = json.loads((ROOT / "configs" / "shuttle_gds.json").read_text())["trap"]
    mapping = {(m["layer"], m["datatype"]): m["name"] for m in cfg["electrode_map"]}
    rf_names = set(cfg["rf_electrodes"])

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12.6, 4.9), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.12}
    )

    for p in polys:
        name = mapping.get((p.layer, p.datatype), f"L{p.layer}")
        is_rf = name in rf_names
        color = WARN if is_rf else ACCENT
        ax.add_patch(
            plt.Polygon(
                p.xy_um,
                closed=True,
                facecolor=color,
                alpha=0.80 if is_rf else 0.30,
                edgecolor=color,
                lw=1.0,
            )
        )
        xs = [x for x, _ in p.xy_um]
        ys = [y for _, y in p.xy_um]
        w_, h_ = max(xs) - min(xs), max(ys) - min(ys)
        ax.text(
            (min(xs) + max(xs)) / 2,
            (min(ys) + max(ys)) / 2,
            name,
            ha="center",
            va="center",
            fontsize=6.8,
            color="#06121c" if is_rf else TEXT,
            fontweight="bold",
            rotation=90 if h_ > w_ else 0,
        )
    ax.set_aspect("equal")
    ax.set_xlim(-11.2, 11.2)
    ax.set_ylim(-8.9, 8.9)
    ax.grid(False)
    _style(ax, "Топология примера GDS: 17 полигонов → 17 каналов", "x, 10⁶ ед. GDS", "y, 10⁶ ед. GDS")
    ax.text(
        0.015,
        0.035,
        "жёлтые — RF (160 В), синие — DC; имя канала задаёт electrode_map, а не порядок полигонов",
        transform=ax.transAxes,
        fontsize=8.0,
        color=MUTED,
    )

    ax2.axis("off")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    rows = [
        ("файл-пример", "configs/generated_fivewire.gds"),
        ("полигонов", f"{len(polys)}"),
        ("слои / datatype", "10 … 26 / 0"),
        ("полосы", "RF0, RF1, центральный DC"),
        ("«пальцы»", "14 (7 сверху, 7 снизу)"),
        ("каналы DC", f"{sum(1 for m in cfg['electrode_map'] if m['name'] not in rf_names)}"),
        ("RF", ", ".join(f"{k} = {v:g} В" for k, v in cfg["rf_electrodes"].items())),
    ]
    ax2.text(0, 0.98, "Что берётся из GDS", fontsize=12.5, fontweight="bold", color=TEXT, va="top")
    for i, (k, v) in enumerate(rows):
        y = 0.855 - i * 0.095
        ax2.text(0.0, y, k, fontsize=9.2, color=MUTED, va="top")
        ax2.text(0.33, y, v, fontsize=8.8, color=TEXT, va="top", family="DejaVu Sans Mono")
    ax2.text(
        0,
        0.10,
        "Нет скрытого порядка: каждый канал — явная пара (layer, datatype).\n"
        "Масштаб этого файла-примера условный (запись UNITS — заглушка),\n"
        "он показывает маппинг слоёв, а не размеры реального чипа.",
        fontsize=8.2,
        color=MUTED,
        va="bottom",
        linespacing=1.6,
    )
    _save(fig, "fig-gds.png")


JOBS = {
    "wave": fig_waveforms,
    "traj": fig_trajectory,
    "valid": fig_validation,
    "robust": fig_robust,
    "ga": fig_ga,
    "dashboard": fig_dashboard,
    "anim": fig_animation,
    "scorecard": fig_scorecard,
    "arch": fig_architecture,
    "gds": fig_gds,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(JOBS), help="run a subset")
    args = ap.parse_args()
    todo = args.only or list(JOBS)
    for name in todo:
        print(f"[{name}]")
        JOBS[name]()


if __name__ == "__main__":
    main()
