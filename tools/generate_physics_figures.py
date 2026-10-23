"""Generate reproducible explanatory figures for docs/polnoe_opisanie.md."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from scipy.constants import atomic_mass

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 180,
    "axes.unicode_minus": False,
})


def draw_physical_flow():
    fig, ax = plt.subplots(figsize=(14, 4.3))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.3)
    ax.axis("off")
    boxes = [
        (0.25, 2.0, 1.75, 1.15, "Электроды и\nнапряжения", "геометрия, поле"),
        (2.55, 2.0, 1.85, 1.15, "$U_{\\mathrm{eff}}(r,t)$", "мгновенная яма $r_0(t)$"),
        (4.95, 2.0, 2.15, 1.15, "$\\mathcal{L}=T-U_{\\mathrm{eff}}$", "уравнение Эйлера—Лагранжа"),
        (7.65, 2.0, 2.1, 1.15, "Движение иона", "$\\bar{r}(t)$ и $\\Sigma(t)$"),
        (10.3, 2.0, 1.6, 1.15, "Физические\nпоказатели", "$n$, конечные ошибки, $e_{\\max}$"),
        (12.3, 2.0, 1.45, 1.15, "Безразмерная\nпотеря", "$\\ell_k$ и робастная оценка"),
    ]
    colors = ["#e8f1fb", "#e8f1fb", "#fff1d6", "#e4f4ea", "#e4f4ea", "#f2e8fb"]
    for (x, y, w, h, title, subtitle), color in zip(boxes, colors):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.08",
                               linewidth=1.2, edgecolor="#334155", facecolor=color)
        ax.add_patch(patch)
        ax.text(x + w/2, y + 0.72, title, ha="center", va="center", fontsize=11, weight="bold")
        ax.text(x + w/2, y + 0.27, subtitle, ha="center", va="center", fontsize=8.5)
    for left, right in [(2.0, 2.55), (4.4, 4.95), (7.1, 7.65), (9.75, 10.3), (11.9, 12.3)]:
        ax.add_patch(FancyArrowPatch((left, 2.58), (right, 2.58), arrowstyle="-|>",
                                     mutation_scale=14, linewidth=1.4, color="#475569"))
    ax.text(6.0, 1.25,
            "Лагранжиан задаёт движение; его действие не является целевой функцией оптимизатора.",
            ha="center", va="center", fontsize=11, color="#7c2d12", weight="bold")
    ax.text(8.7, 0.65,
            "Ион не прижимается к мгновенному минимуму: ускорение центра создаёт инерционное отставание.",
            ha="center", va="center", fontsize=9.5, color="#334155")
    fig.tight_layout()
    fig.savefig(FIGURES / "fizicheskaya_shema.png", bbox_inches="tight")
    plt.close(fig)


def draw_moving_well_example():
    from shuttle_sim.dynamics import propagate
    from shuttle_sim.field import Local

    mass = 40 * atomic_mass
    omega = 2 * np.pi * 400e3
    duration = 2e-6
    distance = 30e-6
    times = np.linspace(0.0, duration, 801)
    s = times / duration
    shape = 10*s**3 - 15*s**4 + 6*s**5
    centers = np.zeros((len(times), 3))
    centers[:, 0] = distance * shape
    hessian = mass * omega**2 * np.eye(3)
    hessians = np.repeat(hessian[None, :, :], len(times), axis=0)
    modes = np.eye(3)
    initial = Local(centers[0].copy(), hessian, np.full(3, omega), modes)
    final = Local(centers[-1].copy(), hessian, np.full(3, omega), modes)
    result = propagate(times, centers, hessians, initial, final, mass, centers[-1],
                       method="exact", hold="linear", return_trajectory=True,
                       trajectory_decimate=4000, path_samples_per_interval=8)
    trajectory = result["trajectory"]
    time_us = np.asarray(trajectory["times_s"]) * 1e6
    well_um = np.asarray(trajectory["well_centers_m"])[:, 0] * 1e6
    ion_um = np.asarray(trajectory["mean_positions_m"])[:, 0] * 1e6
    signed_lag_um = ion_um - well_um

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.1, 1]})
    axes[0].plot(time_us, well_um, color="#2563eb", linewidth=2.2,
                 label="центр мгновенной ямы")
    axes[0].plot(time_us, ion_um, color="#dc2626", linewidth=1.8, linestyle="--",
                 label="среднее положение иона")
    axes[0].set_ylabel("положение вдоль переноса, мкм")
    axes[0].set_title("Ускоряемая яма и инерционное отставание иона")
    axes[0].legend(loc="upper left", frameon=False)
    axes[0].grid(alpha=0.25)
    axes[1].plot(time_us, signed_lag_um, color="#7c3aed", linewidth=1.6)
    axes[1].axhline(0, color="#64748b", linewidth=0.8)
    axes[1].set_xlabel("время, мкс")
    axes[1].set_ylabel("ион − центр, мкм")
    axes[1].grid(alpha=0.25)
    axes[1].text(0.99, 0.92,
                 f"максимум модуля: {result['max_ion_well_lag_m']*1e6:.2f} мкм",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=9,
                 bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cbd5e1"})
    fig.text(0.5, 0.01,
             "Иллюстрационный одномерный профиль в трёх одинаковых гармонических направлениях; не опорный прогон ловушки.",
             ha="center", fontsize=8, color="#475569")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "dvizhenie_i_otslavanie.png", bbox_inches="tight")
    plt.close(fig)
    return result


def draw_dimensionless_loss():
    u = np.linspace(0, 3, 500)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    panels = [
        (np.log1p(u), u, "конечное возбуждение", "$n/n_0$"),
        (np.log1p(u**2), u, "конечная ошибка положения", "$\\epsilon/\\epsilon_0$"),
        (np.log1p(u**2), u, "максимальное отставание", "$e_{\\max}/d$"),
    ]
    for ax, (score, x, title, xlabel) in zip(axes, panels):
        ax.plot(x, score, color="#2563eb", linewidth=2.2)
        ax.axvline(1, color="#dc2626", linestyle="--", linewidth=1.1)
        ax.axhline(np.log(2), color="#64748b", linestyle=":", linewidth=1.0)
        ax.scatter([1], [np.log(2)], color="#dc2626", zorder=3)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("безразмерный вклад в потерю")
    fig.suptitle("Логарифмическая цена нормированных отклонений", y=1.03, fontsize=14)
    fig.text(0.5, -0.02,
             "Красная отметка: соответствующее относительное отклонение равно единице; вклад тогда равен ln 2.",
             ha="center", fontsize=9, color="#475569")
    fig.tight_layout()
    fig.savefig(FIGURES / "obezrazmerivanie_i_poteri.png", bbox_inches="tight")
    plt.close(fig)


def evaluate_reference():
    from shuttle_sim.pipeline import Runner

    cfg = json.loads((ROOT / "configs" / "shuttle_reference.json").read_text(encoding="utf-8"))
    runner = Runner(cfg)
    runner.prepare()
    validation_path = ROOT / "output" / "shuttle_reference" / "validation.json"
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        theta = np.asarray(validation["selected_theta"], float)
        baseline = validation.get("selected_from", "sta")
    else:
        theta = np.zeros(8)
        baseline = "sta"
    if baseline not in ("sta", "smooth"):
        baseline = "sta"
    metrics, trace = runner.evaluate(
        theta, baseline=baseline, exact_nodes=321, subdivisions=8, keep=True,
        return_trajectory=True, trajectory_decimate=50000,
        path_samples_per_interval=int(cfg.get("path_samples_per_interval", 8)),
    )
    return cfg["task"], metrics, runner, trace


def draw_reference_breakdown(task, metrics):
    n0 = float(task["max_n_total"])
    tol = float(task["position_tolerance_m"])
    distance = float(task["distance_m"])
    components = {
        "конечное число квантов": np.log1p(metrics["n_total"] / n0),
        "ошибка иона в конце": np.log1p((metrics["position_error_m"] / tol)**2),
        "ошибка ямы в конце": np.log1p((metrics["well_target_error_m"] / tol)**2),
        "пиковое отставание по пути": np.log1p((metrics["max_ion_well_lag_m"] / distance)**2),
    }
    names = list(components)
    values = np.array([components[k] for k in names])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(names))
    ax.barh(y, values, color=["#2563eb", "#16a34a", "#0f766e", "#7c3aed"])
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("безразмерный вклад, логарифмическая шкала")
    ax.set_title("Разложение текущей функции потерь для опорной геометрии")
    ax.grid(axis="x", alpha=0.25, which="both")
    for yi, value in enumerate(values):
        ax.text(value * 1.12, yi, f"{value:.2e}", va="center", fontsize=9)
    fig.text(0.5, 0.01,
             f"Проверочный расчёт: n={metrics['n_total']:.3g}; максимум отставания={metrics['max_ion_well_lag_m']*1e9:.2f} нм. Регуляризация управления показана отдельно в тексте.",
             ha="center", fontsize=8, color="#475569")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGURES / "razlozhenie_poteri_opornyi.png", bbox_inches="tight")
    plt.close(fig)
    return components


def draw_reference_trajectory(metrics):
    trajectory = metrics["trajectory"]
    time_us = np.asarray(trajectory["times_s"]) * 1e6
    mean = np.asarray(trajectory["mean_positions_m"])
    well = np.asarray(trajectory["well_centers_m"])
    delta_nm = np.linalg.norm(mean - well, axis=1) * 1e9
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.0), sharex=True,
                             gridspec_kw={"height_ratios": [2.0, 1]})
    axes[0].plot(time_us, well[:, 0]*1e6, label="центр ямы", color="#2563eb", linewidth=2)
    axes[0].plot(time_us, mean[:, 0]*1e6, label="среднее положение иона", color="#dc2626",
                 linewidth=1.5, linestyle="--")
    axes[0].set_ylabel("координата вдоль переноса, мкм")
    axes[0].set_title("Опорный расчёт: центр ямы и динамическое положение иона")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)
    axes[1].plot(time_us, delta_nm, color="#7c3aed", linewidth=1.5)
    axes[1].set_xlabel("время, мкс")
    axes[1].set_ylabel("отставание, нм")
    axes[1].grid(alpha=0.25)
    axes[1].text(0.99, 0.92,
                 f"пиковое отставание: {metrics['max_ion_well_lag_m']*1e9:.2f} нм",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=9,
                 bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cbd5e1"})
    fig.tight_layout()
    fig.savefig(FIGURES / "opornyi_perevoz_iona.png", bbox_inches="tight")
    plt.close(fig)


def draw_full_field_comparison(runner, trace, harmonic_metrics):
    from shuttle_sim.dynamics import validate_full_field_centroid

    target = runner.target if runner.task["protocol"] == "one_way" else runner.start
    full = validate_full_field_centroid(
        runner.trap, trace["t"], trace["v"], trace["initial"], target,
        return_trajectory=True, trajectory_decimate=5000,
    )
    full_t = np.asarray(full["trajectory"]["times_s"])
    full_mean = np.asarray(full["trajectory"]["mean_positions_m"])
    harmonic = harmonic_metrics["trajectory"]
    harmonic_t = np.asarray(harmonic["times_s"])
    harmonic_mean = np.asarray(harmonic["mean_positions_m"])
    centers = np.asarray(trace["r"])
    command_t = np.asarray(trace["t"])
    harmonic_at_full = np.column_stack([
        np.interp(full_t, harmonic_t, harmonic_mean[:, axis]) for axis in range(3)
    ])
    center_at_full = np.column_stack([
        np.interp(full_t, command_t, centers[:, axis]) for axis in range(3)
    ])
    harmonic_lag_nm = np.linalg.norm(harmonic_at_full - center_at_full, axis=1) * 1e9
    full_lag_nm = np.linalg.norm(full_mean - center_at_full, axis=1) * 1e9
    model_difference_nm = np.linalg.norm(full_mean - harmonic_at_full, axis=1) * 1e9
    time_us = full_t * 1e6

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.0, 1]})
    axes[0].plot(time_us, center_at_full[:, 0]*1e6, color="#2563eb", linewidth=2,
                 label="центр ямы")
    axes[0].plot(time_us, harmonic_at_full[:, 0]*1e6, color="#dc2626", linestyle="--",
                 linewidth=1.4, label="локальная гармоническая модель")
    axes[0].plot(time_us, full_mean[:, 0]*1e6, color="#059669", linestyle=":",
                 linewidth=1.6, label="полное псевдополе, движение среднего")
    axes[0].set_ylabel("координата вдоль переноса, мкм")
    axes[0].set_title("Проверка локально-гармонического движения полем ловушки")
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[0].grid(alpha=0.25)
    axes[1].plot(time_us, harmonic_lag_nm, color="#dc2626", linestyle="--",
                 linewidth=1.2, label="отставание в гармонической модели")
    axes[1].plot(time_us, full_lag_nm, color="#059669", linewidth=1.4,
                 label="отставание в полном поле")
    axes[1].set_xlabel("время, мкс")
    axes[1].set_ylabel("отставание, нм")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8.5)
    fig.text(0.5, 0.01,
             f"Максимальное расхождение средних траекторий: {model_difference_nm.max():.2f} нм; полный расчёт не включает квантовую ковариацию.",
             ha="center", fontsize=8, color="#475569")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "sravnenie_lokalnoy_i_polnoy_poley.png", bbox_inches="tight")
    plt.close(fig)
    return full, float(model_difference_nm.max()), float(np.max(full_lag_nm))


def main():
    draw_physical_flow()
    toy = draw_moving_well_example()
    draw_dimensionless_loss()
    task, metrics, runner, trace = evaluate_reference()
    components = draw_reference_breakdown(task, metrics)
    draw_reference_trajectory(metrics)
    print("Created figures in", FIGURES)
    print("Toy moving-well peak lag [m]:", toy["max_ion_well_lag_m"])
    print("Reference metrics:", {k: metrics[k] for k in (
        "n_total", "position_error_m", "well_target_error_m", "max_ion_well_lag_m")})
    print("Reference loss components:", components)
    if "--full-field" in sys.argv:
        full, model_difference, full_lag = draw_full_field_comparison(runner, trace, metrics)
        print("Full-field centroid check:", {
            "coherent_n_total": full["coherent_n_total"],
            "position_error_m": full["position_error_m"],
            "max_lag_nm_sampled": full_lag,
            "max_difference_from_harmonic_nm": model_difference,
        })


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
