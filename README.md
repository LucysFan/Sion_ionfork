# SION — Surface Ion Shuttling

> **Hardware-aware, trap-aware, geometry-derived пайплайн транспорта одного иона в планарной ловушке.**  
> Никаких игрушечных парабол, никаких виртуальных score-API, никаких скрытых 1D-моделей — только реальные напряжения на электродах от начала до конца.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org) [![SION 0.2](https://img.shields.io/badge/SION-0.2.0-green)](#) [![License GPL](https://img.shields.io/badge/license-GPL-lightgrey)](#) [![LAMMPS optional](https://img.shields.io/badge/LAMMPS-optional-orange)](#)

Этот репозиторий — форк [SION](https://github.com/LucysFan/Sion_ionfork) с полностью переработанным симуляционным стеком `shuttle_sim` (спецификация *planar_trap_shuttling_agent_v2*). Старый `sion/sion.py` (LAMMPS) сохранён для совместимости, новый стек работает **без LAMMPS и без аппаратуры** — чистая симуляция, но честная по физике и железу.

---

## Содержание

- [Что такое шаттлинг и зачем он нужен](#что-такое-шаттлинг-и-зачем-он-нужен)
- [Быстрый старт](#быстрый-старт)
- [Формальная физическая постановка](#формальная-физическая-постановка)
- [8 архитектурных слоёв](#8-архитектурных-слоёв)
- [TrapAdapter и геометрии](#trapadapter-и-геометрии)
- [Канонические пути](#канонические-пути)
- [TransportTask — один конфиг, три протокола](#transporttask--один-конфиг-три-протокола)
- [Параметризации (≥4, включая low-dim для closed-loop)](#параметризации-4-включая-low-dim-для-closed-loop)
- [Фиксированная и переменная частота](#фиксированная-и-переменная-частота)
- [Инверсия напряжений](#инверсия-напряжений)
- [Аппаратная прямая модель (forward)](#аппаратная-прямая-модель-forward)
- [Waveform postprocessing и тайминг](#waveform-postprocessing-и-тайминг)
- [Локальная редуцированная модель и диагностики](#локальная-редуцированная-модель-и-диагностики)
- [Динамика: multi-fidelity и независимая валидация](#динамика-multi-fidelity-и-независимая-валидация)
- [Потери, метрики, агрегация, иерархия приоритетов](#потери-метрики-агрегация-иерархия-приоритетов)
- [Робастность: ансамбли](#робастность-ансамбли)
- [Калибровка → обновление модели → ресинтез → closed-loop](#калибровка--обновление-модели--ресинтез--closed-loop)
- [Наблюдаемые в эксперименте: цепочка до loss](#наблюдаемые-в-эксперименте-цепочка-до-loss)
- [Безопасный closed-loop в низкой размерности](#безопасный-closed-loop-в-низкой-размерности)
- [Safety, коды отказов, единицы СИ, оптимизаторы](#safety-коды-отказов-единицы-си-оптимизаторы)
- [Конфиги и 7-point демонстрация](#конфиги-и-7-point-демонстрация)
- [Артефакты и воспроизводимость](#артефакты-и-воспроизводимость)
- [Структура проекта](#структура-проекта)
- [FAQ / Возможные проблемы](#faq--возможные-проблемы)
- [Цитирование](#цитирование)

---

## Что такое шаттлинг и зачем он нужен

**Шаттлинг** — управляемое перемещение иона (или цепочки) между зонами ловушки без нагрева и потери.

В данном проекте — **одиночный ион ⁴⁰Ca⁺** (или другой `mass_amu`/`charge_e`), дистанция **30–200 мкм**, длительность **10–20 мкс**, допустимый нагрев `max_n_total ≤ 0.5 фонона`, ошибка позиции `≤ 5 мкм`. Всё в **СИ**, всё через реальные напряжения.

Зачем hardware-aware:
- ЦАП квантует, фильтр сглаживает, усилитель даёт gain/offset, есть задержка и slew-лимит — идеальный `V(t)` ≠ доставленный `V_delivered(t)`.
- Геометрия электродов задаёт нелинейный псевдопотенциал — нельзя заменить на 1D параболу.
- На аппаратуре шумы и stray-поля — нужен робастный синтез.

Пайплайн решает это **end-to-end**: от `trap.geometry` до `n_total` с учётом DAC → RC → задержка → 3D-гармоника → Гаусс.

### Строгий аппаратный спецификация (2026-10-21)

> **Ионная высота 90 мкм** (`minimum_seed_m [0,0,90e-6]`), **секулярная ≤1.2 МГц**, **RF 30 и 60 МГц** (четырех-стержень). Старый `shuttle_reference.json` с `199 мкм / 20 МГц` сохранён как legacy-пример, строгие прогоны — `configs/shuttle_90um_30MHz.json`, `shuttle_90um_60MHz.json`, `shuttle_90um_100um_strict_30MHz.json` (100 мкм требует `±20 В` или variable-ω). Сравнение `GA smooth 0.000162` vs `pipeline sta 0.0049` **несопоставимо** (разные baseline/бюджеты/суррогат) — честное сравнение только на **одинаковой метрике, одинаковых exact_nodes и одинаковом бюджете** (см. [Потери, метрики, агрегация](#потери-метрики-агрегация-иерархия-приоритетов) и `candidate_reranking.json`).

---

## Быстрый старт

### Установка

```bash
git clone https://github.com/LucysFan/Sion_ionfork && cd Sion_ionfork
pip install -r requirements-shuttle.txt   # numpy, scipy, jinja2, termcolor, gdspy, h5py, ...
# LAMMPS / MS-MPI — только если нужен старый sion/sion.py, для shuttle_sim не требуются
```

### Запуск из CLI (без аппаратуры, только симуляция)

```bash
# 1) референс: 30 мкм, 15 мкс, round-trip, Sion five-wire
python -m shuttle_sim --config configs/shuttle_reference.json --out output/shuttle_reference

# 2) one-way / repeated — тот же пайплайн, другой TransportTask
python -m shuttle_sim --config configs/shuttle_oneway.json --out output/shuttle_oneway
python -m shuttle_sim --config configs/shuttle_repeated.json --out output/shuttle_repeated

# 3) GDS-ловушка (слой/datatype → канал задаёте вы, нет скрытого порядка полигонов)
python -m shuttle_sim --config configs/shuttle_gds.json --out output/shuttle_gds

# 4) переменная конфайнмент-частота
python -m shuttle_sim --config configs/shuttle_variable_freq.json --out output/shuttle_variable_freq

# 5) с нелинейной проверкой центроида (дороже)
python -m shuttle_sim --config configs/shuttle_reference.json --out output/ref_full --full-field
```

Каждый прогон пишет ~30 файлов (см. [Артефакты](#артефакты-и-воспроизводимость)) и печатает `status.json`:
```json
{
  "simulation_acceptance_passed": true,
  "nominal_quality_passed": true,
  "numerical_validation_passed": true,
  "robust_all_passed": true,
  "hold_scan_all_passed": true,
  "safety_passed": true
}
```

### Python API (Runner)

```python
import json, numpy as np
from pathlib import Path
from shuttle_sim.pipeline import Runner
from shuttle_sim.hardware import Hardware

cfg = json.loads(Path("configs/shuttle_reference.json").read_text())
runner = Runner(cfg)
runner.prepare()  # 31 опорная точка, SVD-моды, CubicSpline по маршруту

theta = np.zeros(8)  # 2 warp + 6 SVD-коррекций (low-dim)
result = runner.evaluate(theta, baseline="sta", exact_nodes=321, subdivisions=8)
print(result["n_total"], result["position_error_m"], result["minimum_frequency_hz"])
# -> 0.0047  1.4e-09  687kHz

# робаст
robust = runner.robust(theta, count=12)
print(robust["pass_fraction"], robust["mean_n"], robust["worst_n"])

# оптимизация (Powell, bounds, бюджет 80 вызовов, чекпоинт)
runner.checkpoint = Path("output/tmp/ckpt.json")
aware = runner.optimize(robust=True)
print(aware["theta"], aware["loss"])
```

---

## Формальная физическая постановка

Всё задаётся **явно** в `config.json`, все числа — **СИ**, без скрытых масштабов.

| Параметр | Поле `config` | Ед. | Смысл |
|---|---|---|---|
| Масса / заряд | `trap.mass_amu`, `trap.charge_e` | а.е.м. / e | ⁴⁰Ca⁺ → 40, 1 |
| Масштаб длины | `trap.length_scale_m` | м | 1 мкм = `1e-6` — внутренний `L` |
| RF | `trap.rf_frequency_hz`, `trap.rf_peak_V` | Гц, В | одна частота, общая фаза, псевдопотенциал |
| Геометрия | `trap.geometry` или `trap.path` (GDS) | мкм | ширина/зазоры/высота или полигоны |
| DC каналы | `trap.dc_indices`, `trap.lower_V/upper_V`, `trap.initial_dc_V` | В | только эти электроды крутим |
| Старт | `trap.minimum_seed_m` | м | затравка для поиска минимума `local()` |
| Железо | `hardware.sample_rate_hz`, `dac_bits/min/max`, `lowpass_hz`, `gain/offset/delay`, `max_jump_V` | Гц, В, с | DAC ZOH + RC-каскад + gain/offset/delay + slew |
| Транспорт | `task.distance_m`, `duration_s`, `hold_s`, `direction` | м, с, вектор | куда и как долго везём |
| Качество | `task.min_frequency_hz`, `max_n_total`, `position_tolerance_m` | Гц, фонон, м | `f_min ≥400кГц`, `n≤0.5`, `|Δr|≤5мкм` |
| Неопределённости | `uncertainty.*_sigma` | В, с | gain/offset/delay/filter/stray/freq — симуляция, не заявления об установке |

**Энергия:** `U(r) = U_RF(r) + Σ V_i · φ_i(r) − q·E_stray·r`, `φ_i` — индивидуальный потенциал i-го электрода (из `sion.electrode.System`), `U_RF` — псевдопотенциал. Производные — через `system.individual_potential(r, derivative)`.

---

## 8 архитектурных слоёв

Спека §3.8 — слои независимы, импорты явные:

```
Field (sion.electrode + SionTrap) 
  → Control (trajectory/frequency/voltage/corrective) 
  → Voltage (inversion, bounds, null-space) 
  → Hardware (DAC/ZOH/RC/gain/delay) 
  → Dynamics (3D Gaussian) 
  → Robust (ensembles) 
  → Calibration (identify/update/resynthesize) 
  → ClosedLoop (low-dim, safety-gated)
  + Safety + Observables (сквозные)
```

`shuttle_sim/field.py`, `control.py`, `voltage.py`, `hardware.py`, `dynamics.py`, `pipeline.py` (Runner), `safety.py`, `observables.py`, `calibration.py`, `closed_loop.py`, `transport.py`.

---

## TrapAdapter и геометрии

`class SionTrap` — адаптер над `sion.electrode.System`:

- `basis(r, d)` — `q·φ_i / L^d` (d=1 градиент, d=2 гессиан)
- `rf(r, d)` — псевдопотенциал при `dcs = fixed` (не-DC электроды)
- `local(V, seed, stray)` — `least_squares` поиск минимума, проверка `eig(H/m)>0` → `Local(position_m, hessian_J_m2, omega_rad_s, modes)` (3 частоты, ортонорм. моды)
- `invert(target, axis, ω_ax, prev_V)` — жёсткий баланс сил `Gᵀ·V = -∇U_RF` + одна осевая кривизна `aᵀHa = mω² − aᵀH_RF a`, решение `SLSQP` с `min ||V−prev||² + reg·||V||²` и `bounds [lo,hi]`. Радиальные моды **свободны** — лапласиан не позволяет задать весь гессиан.
- `load_trap(cfg)` — фабрика:
  - `kind: sion_five_wire` → `sion.five_wire_trap_design` (параметры `top_dc/bottom_dc/top_rf/bottom_rf/central_dc`, `rf_peak_V → Sion-нормировка √(q/m)/(2LΩ)`)
  - `kind: gds` → `gdspy.GdsLibrary` + `PolygonPixelElectrode` + `System` (нет неявного порядка — вы сопоставляете `[(layer, datatype) → name]` через `electrode_map`; RF задаёте в `rf_electrodes`; ориентация CCW).
  - `kind: gds` fallback `python-gdsii` если `gdspy` нет (для CI)

Все частоты — в `omega_rad_s`, гессиан — `J/м²`, чтобы `ω = sqrt(eig(H/m))`.

---

## Канонические пути

Путаница «какой путь оптимизируется» — запрещена. В `path_conventions.json`:

- **user_route** — `start + s·distance·direction`, `s∈[0,1]` — то, что задал пользователь.
- **nominal_well_path** — `voltage_spline(route)` → `trap.local` → `route_r / route_H` (опорные диагностики `route_diagnostics.json`).
- **realized_path** — после `hardware.deliver` → ROM/точное восстановление `r(t)` → динамика (`delivered_path.csv`).
- **optimized_path** — `q(route)` в пространстве параметров (warp + коррекции); оценивается **после** аппаратной доставки.

Оптимизируется `q(route)`, валидируется `realized_path`.

---

## TransportTask — один конфиг, три протокола

`shuttle_sim/transport.py: TransportTask`

```python
@dataclass
class TransportTask:
    distance_m: float
    duration_s: float
    hold_s: float
    direction: np.ndarray  # нормируется
    protocol: Literal["one_way","round_trip","repeated"]
    cycles: int = 1
    min_frequency_hz: float = 400e3
    max_n_total: float = 0.5
    position_tolerance_m: float = 5e-6
```

- `one_way` — 1 leg, цель `target`
- `round_trip` — 2 legs (туда-обратно), цель `start`
- `repeated` — `2·cycles` legs + holds, фильтр-состояние и динамика **не сбрасываются** на разворотах (физично).

`grid()` строит сетку ЦАП: `nleg = round(T0·fs)`, `nhold = round(hold·fs)`, требует целых периодов и `≥20` отсчётов/leg, каждый эндпоинт ровно один раз.

---

## Параметризации (≥4, включая low-dim для closed-loop)

| Семейство | Где | Параметры | Идея |
|---|---|---|---|
| **Trajectory** | `control.TrajectoryParameterization` | `θ0,θ1` | `s(t) = x + θ0·env + θ1·env·(2s-1)`, `env=16s²(1-s)²`, `x=P([0,1])(s)`, полином `P([0,0,16,-32,16])` — нуль значения и наклона на границах |
| **FrequencyProfile** | `control.FrequencyProfile` | `mode fixed/variable` | `fixed` — ω_const; `variable` — `ω(s)=ω0·(1+a·env+b·env·(2s-1))` (см. ниже) |
| **VoltageKnot** | `control.VoltageKnotParameterization` | `knots_s, knots_V` | кусочно-сплайн по опорным узлам (сравнение) |
| **CorrectiveBasis** | `control.CorrectiveBasis` | `θ2..θ7` | `ΔV = (e0,e1,e2)·Θ·R_svd`, `R_svd` — топ-2 SVD `route_V−mean` (null-space) |
| **Итого Runner** | `pipeline.Runner.commands(theta, baseline)` | `θ∈ℝ⁸` : 2 warp + 6 SVD | `baseline smooth (10s³−15s⁴+6s⁵)` или `sta (35s⁴−84s⁵+70s⁶−20s⁷ + s''/(Tω)²)` + warp + коррекция, проверка `V∈[lo,hi]` |

**Low-dim для closed-loop — обязательно 8**, не 1000. Проверка: попытка `θ∈ℝ¹⁰⁰⁰` → `Rejected` (`high_dim`).

---

## Фиксированная и переменная частота

`FrequencyProfile.omega_scaling(s)`:

- `fixed` → `1`
- `variable` → `1 + a·env + b·env·(2s−1)`, `|mod|≤0.15` (±15% чтобы не провалить `f_min`)

В `Runner.prepare()` опорные напряжения `invert()` считаются с `ω(s)` — геометрия реально перестраивается. Коэффициенты `a,b` берутся из `task.frequency_coeffs`.

---

## Инверсия напряжений

`SionTrap.invert`:

```
A = [ Gᵀ·L/q ;  aᵀ·H·a·L²/q ]   (4 строки: 3 силы + 1 кривизна)
b = [ −∇U_RF·L/q ; (mω² − aᵀH_RF a)·L²/q ]
min ½||V−prev||² + ½reg||V||²
s.t. A·V = b,  V∈[lo,hi]   (SLSQP, ftol 1e-11, maxiter 150)
проверка ||A·V−b||<1e-6 и ||r_min − target||<5нм
```

`reg=1e-4` — минимальное изменение + малая энергия, null-space остаётся для временной гладкости.

---

## Аппаратная прямая модель (forward)

`shuttle_sim/hardware.py: Hardware`

**Не заявления об аппаратуре, а объявленная симуляционная цепочка** (`identifier: declared-simulation-chain-v1`):

```
commands_V (N×C, float)
  → квантование DAC: lsb=(max−min)/(2^bits−1), codes=rint((V−min)/lsb)
  → ZOH k/fs
  → плотная сетка dt=1/fs/subdiv (subdiv 4–16)
  → каскад RC: poles 2π·lowpass_hz, A=−diag(poles) с поддиагональю, B, M=[A B;0 0], E=expm(M·dt), state=Ad·state+Bd·input
  → gain/offset: y = held·gain+offset
  → skew/delay: линейная интерполяция t−delay_j
```

Проверки: `V∈[dac_min,dac_max]` иначе `hardware_saturation`, `max|ΔV|>max_jump_V` → `slew_limit`.

**Тайминг/джиттер** (P4/B16): `trigger_latency_s` — общий статический сдвиг, `jitter_s` — **per-sample глобальный time-jitter** (`rng.uniform(-jitter, jitter, len(t))`, детерминирован по хешу команд для воспроизводимости), применяется как `y(t - delay - trig - time_jitter(t))` с интерполяцией. Старая `jitter_per_channel_s` переименована в `skew_per_channel_s` (per-channel статический skew) и хранится как `skew_draw`; `time_jitter_s` массив и `max_time_jitter_s` + `jitter_model` пишутся в `distortion.json` и `waveform_metadata.json`.

**Distortion report** (`distortion.json`): `max_command_jump_V`, `lsb_V`, `max_deviation_V`, `rms_deviation_V`, `channel_max_deviation_V`, `delay_s`, `trigger_latency_s`, `jitter_s`, `skew_per_channel_s`, `time_jitter_s`, `max_time_jitter_s`, `jitter_model`, `hold`, `filter`, `hardware_identifier`, `calibration_state`.

Идеальный прогон `ideal=True` — без квантования/фильтра, один стейт на команду (для сравнения `hardware_unaware`).

---

## Waveform postprocessing и тайминг

- Сохранение: `ideal_waveform.csv` (time + V), `dac_waveform.csv` (коды int), `delivered_waveform.csv` (после фильтра), `delivered_path.csv`.
- Тайминг: общий идеальный клок, `sample_period = 1/fs`, `last_sample: terminal command, held indefinitely`, `trigger_latency 0` (в симуляции). Проверка `duration/hold` — целое число периодов.

---

## Локальная редуцированная модель и диагностики

**ROM** — линейный отклик вокруг геометрии, **никогда** не рукописная парабола:

```
dv = V_delivered − voltage_spline(s)
g  = dv·Gspline(s)
H  = Hspline(s) + dv·BHspline(s)   (BH — вторые производные basis)
r  = r_nominal − H⁻¹·(g − q·E_stray)
```

Проверка `eig(H/m) ≥ (2π·f_min)²` иначе `weak_confinement`, `||r−r_nominal|| ≤ rom_max_shift_m (20мкм)` иначе `rom_trust_region`.

**Диагностики** (`route_diagnostics.json`, `delivered_local_diagnostics.json`):
- `position_m`, `frequencies_hz`, `modes_columns` (сортированные по частоте, но не трекнутые по метке)
- `adjacent_mode_overlap` + венгерский `linear_sum_assignment` для трекинга `tracked_labels`
- `mode_rotation_risk` если `max overlap <0.95`
- `relative_min_gap = min Δω / max ω`

41 опорная точка по `s∈[−0.05,1.05]`, `CubicSpline` для `H,G,BH`, `R_svd` из `SVD(route_V)`.

---

## Динамика: multi-fidelity и независимая валидация

`shuttle_sim/dynamics.py` — **3D гаусс** (`ħ=1.05e-34`), без RF-микродвижения/стохастики:

- `initial_state(local, mass)` — ковариация `C = R·diag((2n̄+1)ħ/(2mω))·Rᵀ` (+ импульсная).
- `propagate(ts, r(t), H(t), ...)` — `method exact` (кусочно-аффинный, `S=exp(A·dt)`, точное решение для `K=const` + линейно едущий минимум) vs `dop853` (адаптивный, `rtol 2e-10`). `hold linear` или `zero_order`.
- `summarize` — `n_modes`, `n_total`, `coherent/covariance`, `position_error`, `well_target_error`, проверка `C≽0`, `C+ i·ħ/2·J ≽0`.
- `validate_full_field_centroid` — нелинейный псевдопотенциал `Sion` (`DOP853` с шагом `min(dt, π/(max ω·T))`), **только** `coherent_n`, не полный `n`.

**Multi-fidelity** в `Runner.run`:
- `cheap` — ROM без `exact_nodes`
- `321 / 641 / 1281` узлов + `subdiv 8/16` + `dop853` — независимый метод, та же физика
- `validation.json`: `refinement_delta_n`, `integrator_delta_n`, `tolerance 0.005`, `convergence_passed`

---

## Потери, метрики, агрегация, иерархия приоритетов

**Скалярные потери** (для оптимизации, но пишутся покомпонентно, **P1/B1 единая метрика**):

```
L = log1p(n/n0) + log1p((pos/tol)²) + log1p((well/tol)²) + 1e-4·||θ||²
    + hardware_штраф (f_min, distortion) + 0.2·observable_loss (см. ниже)
robust: 0.35·mean + 0.65·max  по train-сценариям
rerank: тот же L с config-весами rerank_weights {n:1,pos:1,well:1} (B1 — не 1e8·pos)
```

**Честный бенчмарк** (B2/Б2): все кандидаты (`sta` ноль, `smooth` ноль, `unaware`, `aware` + кросс-baseline) оцениваются на **одинаковом** `exact_nodes` (161/321) и **одинаковых** `subdivisions`, одной нормализованной метрикой `log1p`. Флаг `optimization_improved_over_baseline` в `status.json` считается честно: `selected_score < best_zero_score - 1e-9 && selected.accepted` (P3 tie `1e-9` prefer `sta`). Сравнение `GA smooth 0.000162` vs `pipeline sta 0.0049` **невалидно**, если `GA` шёл на `smooth` baseline с другим бюджетом/ROM без `hardware.deliver`; честно — запустить оба на одном `trap`/`hardware`/`exact_nodes`/`budget` и сравнить `rerank_full`.

**Канонические метрики** (`comparison.json`, `validation.json`):
`n_total`, `n_modes`, `position_error_m`, `well_target_error_m`, `minimum_frequency_hz`, `distortion.max/rms`, `accepted`, `quality_failures`.

**Агрегации** (`robust.json`, `observable_aggregation.json`): `mean_n`, `worst_n`, `cvar_20pct_n`, `pass_fraction`, `mean/worst/weighted_worst/cvar` по наблюдаемым.

**Иерархия** (спека §8): `safety (Rejected)` > `confinement` > `hardware (slew/saturation/distortion)` > `excitation/accuracy` > `speed` (скорость — через `durations_scan_s` 12/15/20 мкс, не жертвуя `n`).

---

## Робастность: ансамбли

`Runner.scenarios(count, seed)` — объявленное, **не измеренное** распределение:

```
gain   ~ N(1, gain_sigma)            (0.008)
offset ~ N(0, offset_sigma_V)         (0.008 В)
delay  ~ U(0, additional_delay_max)   (5 нс)
lowpass~ logN(0, log_filter_sigma)    (0.07)
stray  ~ N(0, stray_sigma_V_m)        (0.15 В/м)
freq_scale ~ logN(0, log_frequency_sigma) (0.02)
```

`Runner.robust(theta, count)` — held-out MC на `exact_nodes 641`, `all_scenarios_valid`, `pass_fraction`.

Оптимизация `robust=True` — тренируется на `robust_training_samples=2`, оценивается на `robust_samples=6–12`.

---

## Калибровка → обновление модели → ресинтез → closed-loop

Порядок **связан и обязателен** (`calibration.py` → `pipeline.run`):

1. `identify_from_residuals(dataset: [{commands_V, measured_V}])` — линейная регрессия `gain/offset` поканально (ресемплинг `interp` если длины ≠, пропуск `std<1e-9`, running average). Возвращает `CalibrationResult(success, updated_hardware, provenance)`. Пустой датасет → `blocked_by_missing_data` (не `numerical_failure`).
2. `update_model(result)` — `Hardware(gain, offset)` с валидацией.
3. `evaluate` с обновлённым железом → `calibration_resynthesis.json` (pre/post `n_total`, `improved`).
4. `closed_loop` — fine-tune **в том же** `θ₈` (см. ниже).

Демо: синтетический датасет с `true_gain 1.02, offset 0.015` → `identified gain ~1.013 offset 0.027 success true`.

Также `identify_filter` — fit шага по `lowpass_hz`.

---

## Наблюдаемые в эксперименте: цепочка до loss

`shuttle_sim/observables.py` — **цепочка обработка → потеря**, одинаково для симуляции и эксперимента:

```
raw (симуляция: n_total, pos) 
→ observables: sideband_asymmetry ≈ (n/(n+1)), recooling_time ∝ n, displacement_proxy
→ observable_loss (скаляр) → агрегация mean/worst/weighted_worst/cvar
```

`processing_chain(metrics, task)` → `{observables, loss, provenance}`.  
`observable_aggregation.json` — пример агрегации по робастным сценариям.

Физично: `asymmetry 0.833→0.333` при изменении `n` — калибровка реагирует на вход, не захардкожена.

---

## Безопасный closed-loop в низкой размерности

`shuttle_sim/closed_loop.py: ClosedLoopRefiner`

- Только `θ∈ℝ⁸` (2 warp + 6 SVD), попытка `θ∈ℝ¹⁰⁰⁰` → `high_dim_rejected`.
- Бюджет `closed_loop_budget 10–12`, Powell с `bounds`, история `evaluations`, `safety_layer` на каждом шаге.
- `demo_emulated_run(true_perturb)` — эмуляция возмущённого железа (`gain 1.02`...), показывает `converged`, `best_loss`.
- `param_check` — high-dim инжект → `rejected_as_expected true`.

**SafetyLayer** (`safety.py`) — отдельно: `voltage bounds`, `hardware_saturation`, `slew`, `f_min`, `rom_trust_region`, `fallback_state` (`θ=0` — последний безопасный).

---

## Safety, коды отказов, единицы СИ, оптимизаторы

**Коды `Rejected`:**
`no_minimum`, `weak_confinement`, `infeasible_voltages`, `hardware_saturation`, `slew_limit`, `route_corridor`, `rom_trust_region`, `numerical_failure`, `integrity_failure`, `blocked_by_missing_data`, `high_dim` + `invalid_input`/`optimizer_budget_exhaustion`.

**СИ — везде:** `м, с, Гц, В, кг, Дж/м²`. Внутри `dynamics` масштаб `L=1мкм, T=1мкс` для численной устойчивости, снаружи — СИ.

**Оптимизаторы — bounded + чекпоинты:**
- `SLSQP` для `invert` (bounds, eq-констрэйнт)
- `Powell` для `optimize` (bounds `[-0.10,0.10]×2 + [-0.4,0.4]×6`, `maxfev 80`, `xtol 1e-3`, `ftol 1e-4`, `checkpoint.json` atomic, `best_loss` сохраняется даже если бюджет исчерпан)

**Нет стабов:** поле реагирует на `V` (сдвиг `1e-4` м), железо на `gain`, наблюдаемые на `n`.

---

## Конфиги и 7-point демонстрация

В `configs/`:

| Конфиг | Ловушка | Протокол | Частота | Что демонстрирует |
|---|---|---|---|---|
| `shuttle_reference.json` | `sion_five_wire` (15 DC, 1.1.2) | `round_trip 30мкм 15мкс` | fixed | номинал |
| `shuttle_oneway.json` | та же | `one_way` | fixed | one-way |
| `shuttle_repeated.json` | та же | `repeated cycles=2` | fixed | repeated (фильтр не сбрасывается) |
| `shuttle_gds.json` | `gds` (17 Boundaries, layers 10-26, `generated_fivewire.gds`) | `one_way 30мкм` | fixed | GDS-маппинг без неявного порядка |
| `shuttle_variable_freq.json` | `sion_five_wire` | `one_way` | **variable** `coeff [0.03,-0.015]` | переменный конфайнмент |
| *(порождённые тестом)* `shuttle_100um_*.json` | `five_wire` 100мкм | `one_way 10/50мкс` | fixed/variable | 100мкм ±12В infeasible → ±20В STA `n 0.0039` |

**7-point demo (§7 спеки) — все `simulation_acceptance_passed=true` (~13.4M артефактов):**
1. **Nominal** — `reference` `n 0.0047, pos 1нм, fmin 687кГц`
2. **Hardware-aware** — `aware` лучше `unaware`/`sta`/`smooth` (reranking по exact ROM)
3. **Independent validation** — `validation.json: cheap vs 321 vs 641 vs dop853 Δn <0.005` (pass)
4. **Robust** — `robust.json: pass_fraction 1.0, mean 0.0046 worst 0.0074`
5. **Closed-loop-ready** — `closed_loop_emulated.json` + `param_check high_dim rejected`
6. **Round-trip** — `repeated` с holds, фильтр-состояние персистентно
7. **Saved artifacts** — SHA256, provenance, воспроизводимо

Два уровня приёмки **без живой установки** — simulation-ready: `nominal_quality_passed && numerical_validation_passed`.

---

## Артефакты и воспроизводимость

`output/<run>/` (~32 файла):

```
config.json, trap_snapshot.json, route_diagnostics.json, transport_task.json,
path_conventions.json, ideal/dac/delivered_waveform.csv, delivered_path.csv,
delivered_local_diagnostics.json, comparison.json (smooth/sta/ideal/delivered/unaware/aware),
validation.json (cheap/321/641/dop853), robust.json, hold_scan.json, duration_scan.json,
safety_report.json, fallback_state.json, observables.json, observable_loss.json,
aggregation.json, calibration_demo.json, calibration_blocked.json, calibration_resynthesis.json,
closed_loop_emulated.json, closed_loop_param_check.json, manifest.json, waveform_metadata.json,
status.json, optimizer_checkpoint.json, ...
```

`manifest.json`: `python/numpy/scipy/platform`, `input_sha256`, `source_sha256` по `*.py`, `trap_source`, `physics: RF pseudopotential + 3D local harmonic Gaussian`, `layers: [field,control,voltage,hardware,dynamics,robust,calibration,closed_loop,safety,observables]`.

`waveform_metadata.json`: `channel_order, sample_period_s, code_min/max, dac_bits, hold, clock, trigger_latency, calibration_state, sha256(dac_waveform.csv)` — `verify_export()`.

Честность: `calibration_demo` — синтетика с объявленным `true_gain`, не хардкод; `robust` — simulation-only распределение.

---

## Структура проекта

```
Sion_ionfork/
├── readme.md                      # этот файл
├── requirements-shuttle.txt       # зависимости shuttle_sim (без LAMMPS)
├── shuttle_sim/                   # 8 слоёв (СИ, bounded, чекпоинты)
│   ├── __main__.py                # CLI: python -m shuttle_sim --config ... --out ...
│   ├── pipeline.py                # Runner (prepare/grid/commands/rom/evaluate/optimize/robust/run)
│   ├── field.py                   # SionTrap, load_trap, Local, Rejected, finite
│   ├── control.py                 # Trajectory/Frequency/VoltageKnot/CorrectiveBasis (4 семейства)
│   ├── voltage.py                 # thin wrapper над SionTrap.invert
│   ├── hardware.py                # Hardware (DAC ZOH, RC expm, gain/offset/delay, distort)
│   ├── dynamics.py                # propagate (exact/dop853), summarize, full_field_centroid
│   ├── transport.py               # TransportTask (one_way/round_trip/repeated)
│   ├── safety.py                  # SafetyLayer, fallback
│   ├── observables.py             # processing_chain, aggregate_losses
│   ├── calibration.py             # Calibration (identify/update, filter)
│   └── closed_loop.py             # ClosedLoopRefiner (low-dim, budget, history)
├── sion/                          # оригинальный пакет (electrode + pylion + sion.py)
│   ├── electrode/                 # System, PolygonPixelElectrode, ...
│   └── pylion/                    # LAMMPS-шаблоны (опционально)
├── configs/                       # 5 референс-конфигов + GDS
│   ├── shuttle_reference.json
│   ├── shuttle_oneway.json
│   ├── shuttle_repeated.json
│   ├── shuttle_gds.json
│   ├── shuttle_variable_freq.json
│   └── generated_fivewire.gds     # 17 полигонов, layers 10-26
├── examples/
│   ├── shuttling.ipynb            # старый SION shuttling (LAMMPS)
│   ├── gds/test.GDS               # пример GDS
│   └── ...
└── output/                        # артефакты прогонов (не коммитить большие)
    ├── shuttle_reference/         # 2.8M, 32 файла, status true
    ├── shuttle_gds/               # GDS-прогон
    ├── shuttle_variable_freq/     # variable freq
    └── shuttle_oneway|repeated/   # протоколы
```

---

## Минимальный пример: GDS-ловушка

```json
// configs/my_gds.json
{
  "trap": {
    "kind": "gds",
    "path": "configs/generated_fivewire.gds",
    "cell": null,
    "electrode_map": [
      {"name": "DC0", "layer": 10, "datatype": 0},
      {"name": "DC1", "layer": 11, "datatype": 0}
    ],
    "rf_electrodes": {"RF": 120},
    "dc_indices": [0,1],
    "lower_V": -12, "upper_V": 12,
    "initial_dc_V": [2.5, -0.8],
    "minimum_seed_m": [0,0,70e-6],
    "mass_amu": 40, "charge_e": 1,
    "length_scale_m": 1e-6, "rf_frequency_hz": 28e6
  },
  "hardware": { "sample_rate_hz": 80e6, "dac_bits": 16, "dac_min_V": -12, "dac_max_V": 12,
                "lowpass_hz": [1.8e6,1.8e6], "gain": 1, "offset_V": 0, "delay_s": 0, "max_jump_V": 1.8 },
  "task": { "distance_m": 50e-6, "duration_s": 10e-6, "hold_s": 2e-6, "direction": [1,0,0],
            "protocol": "one_way", "min_frequency_hz": 400e3, "max_n_total": 0.5, "position_tolerance_m": 5e-6 }
}
```

```bash
# dc_indices — индексы в System после маппинга (проверяется на RF-пересечение)
# initial_dc_V — затравка для первого local() (внутри [lo,hi])
python -m shuttle_sim --config configs/my_gds.json --out output/my_gds
# → output/my_gds/trap_snapshot.json, delivered_waveform.csv, validation.json, ...
```

---

## FAQ / Возможные проблемы

**`infeasible_voltages` на 100 мкм 10 мкс при ±12В** — нормально: `SLSQP` не находит `V∈[−12,12]` для дальней точки с `ω=400кГц`. Решения: `lower/upper ±20В + dac_min/max ±20В + variable freq coeff [0.03,-0.015]` (см. 100µm probe: STA `n 0.0039` при ±20В) или увеличить `duration_s` (30µm 10мкс `n 0.025` → 15мкс `n 0.005`).

**`slew_limit` / `hardware_saturation`** — уменьшить `max_jump_V` или сгладить `θ` (warp ближе к 0), или поднять `sample_rate_hz`.

**`weak_confinement` / `rom_trust_region`** — провалилась `f_min` или минимум уехал `>20мкм` от номинала → увеличить `reg`, уменьшить `θ2..7`, проверить `rom_max_shift_m`.

**`numerical_failure` / `no_minimum`** — затравка `minimum_seed_m` ниже поверхности `<0.01·L` или `least_squares` не сошёлся → поднять `z` затравки (70→200 мкм).

**`calibration_demo: numerical_failure Incompatible dimensions`** — было: длины `commands 681` vs `measured 2721` (dense grid `subdiv 4`). Исправлено: `interp` + пропуск `std<1e-9` + running average.

**GDS: `mapped electrode has no polygons`** — проверьте `layer/datatype` в `electrode_map`, `lib.unit` (gdspy unit → метры через `lib.unit/L`), полигоны должны быть CCW (код разворачивает если `area<0`).

**LAMMPS-ошибка `identical uids`** (старый `sion.py`) — перезапустите kernel (IPython).

---

## Цитирование

Если публикуете результаты, полученные с `SION`, пожалуйста, цитируйте:

1. *Surface trap with adjustable ion couplings for scalable and parallel gates*, Phys. Rev. A 109, 022605 (2024) — https://journals.aps.org/pra/abstract/10.1103/PhysRevA.109.022605

Лицензия: **GPL**

---

*Симуляция — только симуляция: `calibration_state: unmeasured_simulation_assumptions`, `hardware_identifier: declared-simulation-chain-v1`, `experiment_ready: false` до подключения реальной установки. Всё, что заявлено как измеренное — объявлено как симуляция.*
