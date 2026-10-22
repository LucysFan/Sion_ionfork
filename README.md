# SION — Surface Ion Shuttling

### Перенос одиночного иона в планарной ловушке: от геометрии электродов до числа фононов

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square">
  <img alt="shuttle_sim" src="https://img.shields.io/badge/shuttle__sim-0.3.0-2ea043?style=flat-square">
  <img alt="surface-ion" src="https://img.shields.io/badge/surface--ion-1.1.2-6f42c1?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/license-GPL--3.0-lightgrey?style=flat-square">
  <img alt="LAMMPS" src="https://img.shields.io/badge/LAMMPS-optional-orange?style=flat-square">
  <img alt="Scope" src="https://img.shields.io/badge/scope-simulation--only-blue?style=flat-square">
</p>

<a id="top"></a>

![Пять прогонов пайплайна](assets/fig-scorecard.png)

*Рис. 1.* Пять конфигураций из репозитория.

Пайплайн считает перенос иона так, как он идёт на стенде. Сначала потенциал по геометрии
электродов, а не по одномерной параболе. Потом инверсия: какие напряжения поставить, чтобы яма
оказалась в нужной точке. Дальше эти напряжения проходят через ЦАП, ZOH, RC-фильтр, усиление,
задержку и slew-лимит — и только после этого попадают в трёхмерную динамику гауссова пакета.
На выходе величины, которые измеряют в лаборатории: нагрев в фононах, ошибка позиции, минимальная
секулярная частота.

![Ион в ловушке](assets/anim-shuttle.gif)

*Рис. 2.* Номинальный прогон: ион проходит 30 мкм и возвращается; справа — доставленные напряжения
и координата `x(t)`.

| Что внутри | Как сделано |
|---|---|
| Ловушка | `sion_five_wire` (аналитическая геометрия) или `gds` — канал задаётся парой `(layer, datatype)` |
| Управление | `θ ∈ ℝ⁸`: 2 параметра time-warp и 6 параметров SVD-коррекции |
| Аппаратный тракт | ЦАП (разрядность, LSB, клиппинг), ZOH, RC-каскад, усиление, смещение, задержка, джиттер, slew |
| Динамика | трёхмерная локальная гармоника, точный пропагатор или DOP853, ковариация 6×6 |
| Приёмка | номинал, multi-fidelity валидация, робастный ансамбль, скан паузы, коды отказа |
| Оптимизация | Powell по 8 параметрам, островной GA (5 стратегий), closed-loop с бюджетом |
| Артефакты | ~30 файлов на прогон: конфиг, волновые формы, траектория, диагностики, `status.json` |

| Метрика | Норма | Что значит |
|---|---|---|
| `n_total` | ≤ 0.5 фонона | нагрев, который потом снимают охлаждением |
| `position_error_m` | ≤ 5 мкм | промах по позиции |
| `minimum_frequency_hz` | ≥ 400 кГц | слабое удержание, ион можно потерять |
| `max_deviation_V`, `rms_deviation_V` | десятки мВ, без клиппинга | железо не воспроизвело команду |

<a id="toc"></a>

## Содержание

[Запуск](#quickstart) · [API](#api) · [Артефакты](#output) · [Архитектура](#architecture) ·
[Аппаратная модель](#hardware) · [Метрики](#metrics) · [Номинальный прогон](#results-nominal) ·
[Валидация](#results-validation) · [Робастность](#results-robust) · [GA](#results-ga) ·
[Дашборд](#dashboard) · [GDS](#gds) · [Конфиги](#configs) · [Ограничения](#honesty) ·
[Ошибки](#faq) · [Структура](#structure) · [Цитирование](#cite)

---

<a id="quickstart"></a>

## Запуск

```bash
git clone https://github.com/LucysFan/Sion_ionfork && cd Sion_ionfork
pip install -r requirements-shuttle.txt      # numpy, scipy, jinja2, matplotlib, python-gdsii, …
```

LAMMPS и MS-MPI нужны только для старого `sion/sion.py`; для `shuttle_sim` они не требуются.

Номинальный прогон — 30 мкм, 15 мкс на плечо, туда и обратно, геометрия five-wire:

```bash
python -m shuttle_sim --config configs/shuttle_reference.json --out output/shuttle_reference
```

Минуты через две в `output/shuttle_reference/` появятся файлы прогона и такая сводка:

```json
{
  "simulation_acceptance_passed": true,
  "nominal_quality_passed": true,
  "numerical_validation_passed": true,
  "robust_all_passed": true,
  "hold_scan_all_passed": true,
  "safety_passed": true,
  "experiment_ready": false
}
```

Остальные сценарии запускаются так же:

```bash
python -m shuttle_sim --config configs/shuttle_oneway.json        --out output/shuttle_oneway
python -m shuttle_sim --config configs/shuttle_repeated.json      --out output/shuttle_repeated
python -m shuttle_sim --config configs/shuttle_gds.json           --out output/shuttle_gds
python -m shuttle_sim --config configs/shuttle_variable_freq.json --out output/shuttle_variable_freq
python -m shuttle_sim --config configs/shuttle_reference.json     --out output/ref_full --full-field
```

<a id="api"></a>

### API

```python
import json, numpy as np
from pathlib import Path
from shuttle_sim.pipeline import Runner

cfg = json.loads(Path("configs/shuttle_reference.json").read_text())
runner = Runner(cfg)
runner.prepare()                     # 31 опорная точка, SVD-моды, сплайны по маршруту

theta = np.zeros(8)                  # 2 параметра warp + 6 параметров SVD-коррекции
res = runner.evaluate(theta, baseline="sta", exact_nodes=321, subdivisions=8)
print(res["n_total"], res["position_error_m"], res["minimum_frequency_hz"])
# 0.0049  1.4e-09  687910.0

robust = runner.robust(theta, count=12)      # ансамбль по объявленным неопределённостям
print(robust["pass_fraction"], robust["mean_n"], robust["worst_n"])

aware = runner.optimize(robust=True)         # Powell, ограничения, чекпоинт, бюджет вызовов
print(aware["theta"], aware["loss"])
```

<a id="output"></a>

### Что пишется в `output/`

| Файл | Содержание |
|---|---|
| `status.json` | сводка приёмки |
| `ideal_waveform.csv`, `dac_waveform.csv`, `delivered_waveform.csv` | команда, коды ЦАП, доставленный сигнал по 15 каналам |
| `delivered_path.csv` | траектория иона `x, y, z (t)` после железа |
| `route_diagnostics.json`, `delivered_local_diagnostics.json` | позиция, частоты, моды и их перекрытие вдоль маршрута |
| `comparison.json` | `smooth`, `sta`, `unaware`, `aware` на одной метрике |
| `validation.json` | ROM, 321 и 1281 узел, DOP853; дрейфы `Δn` |
| `robust.json`, `hold_scan.json`, `duration_scan.json` | сценарии возмущений, `mean`/`worst`/`CVaR`, зависимость от паузы и длительности |
| `safety_report.json`, `fallback_state.json` | проверки границ и последнее безопасное состояние |
| `calibration_*.json`, `closed_loop_*.json` | идентификация `gain`/`offset`, ресинтез, low-dim closed-loop |
| `manifest.json`, `waveform_metadata.json` | версии, SHA-256 входов и исходников, порядок каналов, единицы |

<a id="architecture"></a>

## Архитектура

![Архитектура](assets/fig-architecture.png)

*Рис. 3.* Слои, модули и артефакты.

Восемь слоёв, каждый — отдельный модуль со своим артефактом. Подробнее о соответствии спецификации:
[`docs/architecture.md`](docs/architecture.md); об островном алгоритме —
[`docs/ga_architecture.md`](docs/ga_architecture.md).

| № | Слой | Модуль | За что отвечает |
|---|---|---|---|
| 1 | Поле | `field.py` | `SionTrap`: псевдопотенциал RF и вклад DC, поиск минимума, гессиан, моды |
| 2 | Управление | `control.py` | `θ ∈ ℝ⁸`: time-warp, огибающие, SVD-моды; фиксированная и переменная частота |
| 3 | Инверсия | `voltage.py` | баланс сил и одно условие кривизны, `SLSQP`, ограничения, регуляризация |
| 4 | Аппаратный тракт | `hardware.py` | ЦАП и ZOH, RC-каскад через `expm`, усиление, смещение, задержка, slew |
| 5 | Динамика | `dynamics.py` | трёхмерный гауссов пакет, `exact` и `DOP853`, ковариация и `n_total` |
| 6 | Робастность | `pipeline.robust` | ансамбль по `gain`, `offset`, задержке, фильтру, stray-полю и масштабу частоты |
| 7 | Калибровка | `calibration.py` | `identify_from_residuals` → `update_model` → ресинтез |
| 8 | Closed-loop | `closed_loop.py` | 8 параметров, `SafetyLayer`, бюджет, журнал итераций |

<details>
<summary><b>Постановка и что именно оптимизируется</b></summary>

Энергия: `U(r) = U_RF(r) + Σ Vᵢ·φᵢ(r) − q·E_stray·r`, где `φᵢ` — потенциал i-го электрода из
`sion.electrode.System`, `U_RF` — псевдопотенциал. Производные берутся через
`system.individual_potential(r, derivative)`.

Инверсия напряжений:

```
A = [ Gᵀ·L/q ; aᵀ·H·a·L²/q ]          (3 условия баланса сил + 1 условие кривизны)
b = [ −∇U_RF·L/q ; (mω² − aᵀH_RF a)·L²/q ]
min ½‖V−prev‖² + ½·reg·‖V‖²   при  A·V = b,  V ∈ [lo, hi]
```

Дальше проверка `‖local − target‖ < 5 нм`. Радиальные моды остаются свободными: лапласиан не даёт
задать весь гессиан.

Маршрут в пайплайне — не одно понятие, а четыре, и они не смешиваются (`path_conventions.json`):

| Маршрут | Откуда |
|---|---|
| `user_route` | `start + s·distance·direction`, `s ∈ [0,1]`, из конфига |
| `nominal_well_path` | `voltage_spline(s)` → `trap.local()` → `route_r`, `route_H` |
| `realized_path` | доставленные напряжения → точная ROM-реконструкция → динамика |
| `optimized_path` | `q(route)` в пространстве параметров, оценивается после железа |

Оптимизируется `q(route)`, проверяется `realized_path`.

</details>

<details>
<summary><b>Почему 8 параметров, а не 200 отсчётов</b></summary>

| Семейство | Модуль | Параметры | Определение |
|---|---|---|---|
| `TrajectoryParameterization` | `control.py` | `θ₀, θ₁` | `s(t) = x + θ₀·env + θ₁·env·(2s−1)`, `env = 16s²(1−s)²`; значение и наклон на границах нулевые |
| `FrequencyProfile` | `control.py` | `a, b` | `ω(s) = ω₀·(1 + a·env + b·env·(2s−1))`, отклонение до 15 % |
| `VoltageKnotParameterization` | `control.py` | узлы сплайна | кусочно-сплайновая аппроксимация опорных напряжений, для сравнения |
| `CorrectiveBasis` | `control.py` | `θ₂…θ₇` | `ΔV = (e₀,e₁,e₂)·Θ·R_svd`, `R_svd` — две старшие сингулярные моды `route_V` |

`θ ∈ ℝ⁸` — это 2 параметра временной деформации и 6 в подпространстве, которое реально сдвигает
минимум. Запрос размерности 1000 отклоняется с кодом `Rejected('high_dim')`.

</details>

<a id="hardware"></a>

## Аппаратная модель

![Волновые формы](assets/fig-waveforms.png)

*Рис. 4.* Сверху — как меняется команда `ΔV(t)`; снизу — насколько доставленный сигнал отличается
от команды по каналам.

```
commands_V (N × C)
  → квантование ЦАП:  lsb = (max − min)/(2^bits − 1)
  → ZOH на сетке k/fs
  → плотная сетка dt = 1/(fs·subdiv), subdiv 4…16
  → RC-каскад: точное expm(M·dt) на ZOH-сетке, состояние с предысторией
  → gain·held + offset (поканально)
  → skew, задержка и per-sample time-jitter (детерминирован по хешу команд)
  → проверки: клиппинг → hardware_saturation;  max|ΔV| → slew_limit
```

В номинальном прогоне `max|ΔV| = 96.7 мВ`, `rms = 27.2 мВ`, `LSB = 366 мкВ`, клиппинга нет.
Идеальный и доставленный сигналы различаются, и оптимизатор видит именно второй.

<details>
<summary><b>Тайминг, джиттер, отчёт об искажениях</b></summary>

- `trigger_latency_s` — общий статический сдвиг.
- `jitter_s` — per-sample джиттер, `rng.uniform(-jitter, jitter, len(t))`, учитывается как
  `y(t − delay − trig − time_jitter(t))` с интерполяцией.
- `skew_per_channel_s` — статический поканальный сдвиг (бывший `jitter_per_channel_s`).
- `distortion.json`: `max_command_jump_V`, `lsb_V`, `max_deviation_V`, `rms_deviation_V`,
  `channel_max_deviation_V`, `delay_s`, `trigger_latency_s`, `jitter_s`, `skew_per_channel_s`,
  `time_jitter_s`, `max_time_jitter_s`, `jitter_model`, `hold`, `filter`, `hardware_identifier`,
  `calibration_state`.

Режим `ideal=True` выключает квантование и фильтр (одно состояние на команду) — для сравнения
решений с учётом железа и без него.

</details>

<a id="metrics"></a>

## Приоритеты и потери

Приоритеты из спецификации (§8): нарушение верхнего уровня отменяет результат при любых остальных
показателях.

```
safety  →  confinement  →  hardware realizability  →  excitation / accuracy  →  speed
```

```
L = log1p(n/n₀) + log1p((pos/tol)²) + log1p((well/tol)²) + 1e-4·‖θ‖²  + штрафы за железо
robust: 0.35·mean + 0.65·max по обучающим сценариям
```

| Метрика | Где смотреть | Смысл |
|---|---|---|
| `n_total`, `n_modes` | везде | нагрев, бюджет ≤ 0.5 |
| `position_error_m` | везде | ошибка позиции, допуск 5 мкм |
| `minimum_frequency_hz` | везде | удержание, порог 400 кГц |
| `max_deviation_V`, `rms_deviation_V` | `distortion.json` | насколько железо ушло от команды |
| `mean_n`, `worst_n`, `cvar_20pct_n`, `pass_fraction` | `robust.json` | хвост распределения, а не только среднее |
| `refinement_delta_n`, `integrator_delta_n` | `validation.json` | насколько ответ зависит от выбора модели |
| `optimization_improved_over_baseline` | `status.json` | оптимизация реально лучше нулевого `θ` или нет |

Коды отказа: `no_minimum`, `weak_confinement`, `infeasible_voltages`, `hardware_saturation`,
`slew_limit`, `route_corridor`, `rom_trust_region`, `numerical_failure`, `integrity_failure`,
`blocked_by_missing_data`, `high_dim`, `invalid_input`, `optimizer_budget_exhaustion`.

<a id="results-nominal"></a>

## Номинальный прогон

![Траектория](assets/fig-trajectory.png)

*Рис. 5.* Траектория, скорость и минимальная секулярная частота.

30 мкм туда и обратно, 15 мкс на плечо, пауза 2 мкс. Вышло `n_total = 0.0071` фонона — 1.4 %
бюджета, ошибка позиции `1.38 нм` при допуске 5 мкм, минимальная частота `688 кГц` при пороге
400 кГц.

![Частоты вдоль маршрута](assets/fig-frequencies.png)

*Рис. 6.* Секулярные частоты по 41 опорной точке маршрута (ROM-диагностика). Старшая мода задаёт
запас по удержанию, две другие — радиальные колебания.

<a id="results-validation"></a>

## Валидация

![Валидация](assets/fig-validation.png)

*Рис. 7.* Четыре уровня точности на одном прогоне и дрейф при смене модели.

Тот же прогон, четыре способа счёта. Ответ почти не двигается:

| Сравнение | `Δn` по модулю |
|---|---|
| дешёвая ROM → точная модель (321 узел) | `1.7e-4` |
| уточнение сетки до 1281 узла | `1.4e-4` |
| грубая модель → тонкая | `2.2e-3` |
| `exact` → `DOP853` | `3.0e-7` |

Все дрейфы меньше допуска `0.01` фонона, `convergence_passed = true`. Искать можно на дешёвой модели,
а решение всё равно проверяется на точной.

<a id="results-robust"></a>

## Робастность

![Робастность](assets/fig-robust.png)

*Рис. 8.* Распределение `n_total` по сценариям и запас до бюджета.

Шесть сценариев с объявленными (не измеренными) возмущениями: `gain`, `offset`, задержка, фильтр,
stray-поле, масштаб частоты. Приняты все (`pass_fraction = 1.0`), распределение показано целиком.

| Статистика | `n_total` | Доля бюджета 0.5 |
|---|---|---|
| `mean` | `0.0269` | 5 % |
| `CVaR 20 %` | `0.0744` | 15 % |
| `worst` | `0.0764` | 15 % |

Среднее в шесть раз меньше худшего — поэтому агрегация `0.35·mean + 0.65·max`, а не среднее.

<a id="results-ga"></a>

## Островной GA

![Island GA](assets/fig-ga.png)

*Рис. 9.* Фитнес по островам, разнообразие популяции, зал славы, лучший геном.

Пять островов работают одновременно, каждый со своим уклоном:

| Остров | Базовая траектория | Стратегия | Геном |
|---|---|---|---|
| **S** | `smooth` (10s³ − 15s⁴ + 6s⁵) | гладкая адиабатика, малая мутация | 8 |
| **A** | `sta` (слагаемое `s''/(Tω)²`) | лучше работает на малых `T`, чувствителен к фильтру | 8 |
| **V** | `sta` | переменная частота; делает допустимыми 100 мкм при ±12 В | 10 |
| **R** | `sta` | отбор по `worst`, а не по `mean` | 8 |
| **H** | `sta` | штраф за искажение фильтром (`rms`, `max_deviation`) | 8 |

Миграция по кольцу `S→A→V→R→H→S` каждые 5 поколений, зал славы с фильтром разнообразия
(нормализованное расстояние по `θ` не меньше 0.12), чекпоинт после каждой генерации.

Демо-прогон `output/ga_demo`, 8 поколений за 106 с: фитнес `0.00236`, `n_total = 0.00125` фонона,
ошибка позиции `1.25 нм`. Лучшее решение — 10 чисел вместо 200 отсчётов напряжения.

```bash
python -m shuttle_sim.ga --config configs/shuttle_ga_island.json --out output/ga_run \
       --dashboard --port 8000 --generations 40
```

<a id="dashboard"></a>

## Дашборд

![Дашборд](assets/fig-dashboard.png)

*Рис. 10.* Панели дашборда на данных демо-прогона.

Эволюция не ждёт дашборд: `state.json` перезаписывается атомарно (`state.json.tmp → state.json`)
после каждой генерации, сервер живёт отдельным процессом и только читает файл.

```bash
python -m shuttle_sim.dashboard.server --out output/ga_demo --port 8000
```

Панели: фитнес по поколениям и островам, разнообразие, гистограмма `n_total`, зал славы,
Парето-фронт `n`–`worst`, 15 каналов `V(t)`, траектория `r(t)`, ловушка сверху с анимацией
кандидатов, граф миграций, состояние CUDA (без `cupy` работает numpy-ветка, результат тот же).

<a id="gds"></a>

## Геометрия из GDS

![GDS](assets/fig-gds.png)

*Рис. 11.* Топология примера GDS и соответствие слоёв каналам.

Канал — это явная пара `(layer, datatype)` в `electrode_map`; порядок полигонов в файле ни на что
не влияет. RF-электроды задаются отдельно, пересечение RF и DC проверяется при загрузке.

```jsonc
{
  "trap": {
    "kind": "gds",
    "path": "configs/generated_fivewire.gds",
    "electrode_map": [
      {"name": "RF0", "layer": 10, "datatype": 0},
      {"name": "RF1", "layer": 11, "datatype": 0},
      {"name": "DC0", "layer": 12, "datatype": 0},
      {"name": "DC1", "layer": 13, "datatype": 0}
    ],
    "rf_electrodes": {"RF0": 160, "RF1": 160},
    "dc_indices": [2, 3],
    "lower_V": -12, "upper_V": 12,
    "initial_dc_V": [2.5, -0.8],
    "minimum_seed_m": [0, 0, 90e-6],
    "mass_amu": 40, "charge_e": 1, "length_scale_m": 1e-6, "rf_frequency_hz": 28e6
  }
}
```

<details>
<summary><b>О масштабе в файле примера</b></summary>

`configs/generated_fivewire.gds` — демонстрационный файл для проверки соответствия слоёв, а не
описание реального устройства: запись UNITS в нём условная, поэтому координаты на рисунке даны в
единицах GDS. Для реальной геометрии достаточно указать корректный UNITS: `load_trap` переводит
единицы файла в метры по коэффициенту `lib_unit / L`.

</details>

<a id="configs"></a>

## Конфиги

| Конфигурация | Ловушка | Протокол | Для чего |
|---|---|---|---|
| `shuttle_reference.json` | five-wire, RF 20 МГц | round-trip 30 мкм, 15 мкс | номинал, базовая приёмка |
| `shuttle_oneway.json` | five-wire | one-way 30 мкм | другой протокол на том же пайплайне |
| `shuttle_repeated.json` | five-wire | repeated ×2, 20 мкм | состояние фильтра не сбрасывается на разворотах |
| `shuttle_gds.json` | GDS, слои 10–26 | round-trip 30 мкм | соответствие слоёв каналам |
| `shuttle_variable_freq.json` | five-wire | round-trip, `coeff [0.03, −0.015]` | переменная частота удержания |
| `shuttle_90um_30MHz.json` | five-wire, h = 90 мкм, RF 30 МГц, ±12 В | one-way 30 мкм | строгая аппаратная постановка |
| `shuttle_90um_60MHz.json` | five-wire, h = 90 мкм, RF 60 МГц, ±12 В | one-way 30 мкм | та же геометрия при удвоенной RF |
| `shuttle_90um_100um_strict_30MHz.json` | h = 90 мкм | round-trip 100 мкм, 10 мкс | предел по напряжениям |
| `shuttle_100um_10us.json`, `shuttle_100um_50us.json` | ±20 В | one-way 100 мкм, 10 и 50 мкс | что даёт длительность на дальнем переносе |
| `shuttle_100um_strict.json` | ±20 В, GA | round-trip 100 мкм | конфиг генетического алгоритма |
| `shuttle_ga_island.json` | five-wire | round-trip 30 мкм | пять островов, миграция, зал славы |
| `shuttle_ga_isolated.json` | h = 90 мкм, переменная `ω` | round-trip 30 мкм | остров с геномом размерности 10 |

<details>
<summary><b>Поля конфига</b></summary>

| Параметр | Поле | Единица | Смысл |
|---|---|---|---|
| Масса и заряд | `trap.mass_amu`, `trap.charge_e` | а.е.м., e | ⁴⁰Ca⁺: 40 и 1 |
| Масштаб длины | `trap.length_scale_m` | м | внутренний `L`, обычно `1e-6` |
| RF | `trap.rf_frequency_hz`, `trap.rf_peak_V` | Гц, В | одна частота, общая фаза, псевдопотенциал |
| Геометрия | `trap.geometry` либо `trap.path` с `electrode_map` | мкм, м | аналитика или GDS |
| DC-каналы | `trap.dc_indices`, `lower_V`, `upper_V`, `initial_dc_V` | В | какие электроды крутим и в каких границах |
| Затравка | `trap.minimum_seed_m` | м | стартовая точка поиска минимума |
| Железо | `hardware.*` | Гц, В, с | дискретизация, фильтр, усиление, задержка, slew |
| Протокол | `task.distance_m`, `duration_s`, `hold_s`, `direction`, `protocol`, `cycles` | м, с, — | параметры переноса |
| Критерии | `task.min_frequency_hz`, `max_n_total`, `position_tolerance_m` | Гц, фонон, м | условия приёмки |
| Неопределённости | `uncertainty.*_sigma` | В, с, — | объявленные распределения |

</details>

<a id="honesty"></a>

## Ограничения

Результаты получены расчётом, а не измерением на стенде, и это видно в артефактах:

- `calibration_state: unmeasured_simulation_assumptions`, `hardware_identifier:
  declared-simulation-chain-v1`, `experiment_ready: false` — до подключения установки;
- поле — трёхмерная локальная гармоника; полное RF-микродвижение и негауссова квантовая эволюция
  не моделируются (записано в `manifest.json` и `validation.json`);
- распределения в `robust` объявлены, а не измерены;
- калибровочный датасет синтетический: `calibration_resynthesis.json` содержит `improved: false`
  (0.0049 → 0.0236) — идентификация по трём точкам модель ухудшила, и файл оставлен как есть;
- демо closed-loop упирается в бюджет: `evaluations 11`, `optimizer_success: false`,
  `termination: Maximum number of function evaluations…`; потери всё равно упали с `0.0472`
  до `0.0158`;
- запрос размерности 1000 для closed-loop отклоняется: `closed_loop_param_check.json`,
  `rejected_as_expected: true`;
- без внешних данных калибровка не подменяется расчётом: `calibration_blocked.json`,
  `blocked_by_missing_external_data`;
- сравнивать GA и пайплайн можно только при одинаковых `trap`, `hardware`, `exact_nodes` и бюджете.

<a id="faq"></a>

## Ошибки и что делать

<details>
<summary><b><code>infeasible_voltages</code> на 100 мкм за 10 мкс при ±12 В</b></summary>

Это ожидаемо: `SLSQP` не находит `V ∈ [−12, 12]` для дальней точки при `ω = 400 кГц`.
По возрастанию затрат: увеличьте `duration_s`; расширьте границы до ±20 В и согласуйте их с
`hardware.dac_min/max`; включите переменную частоту (`coeff [0.03, −0.015]`), снижая её в середине
маршрута. Если задан `task.durations_scan_s`, прогон положит `duration_scan.json` —
по нему видно, где кончается длительность.

</details>

<details>
<summary><b><code>slew_limit</code> и <code>hardware_saturation</code></b></summary>

Уменьшите `max_jump_V`, поднимите `sample_rate_hz` или сведите `θ` (деформацию времени) к нулю.
`max_command_jump_V` в `distortion.json` покажет канал и участок, где сработало ограничение.

</details>

<details>
<summary><b><code>weak_confinement</code> и <code>rom_trust_region</code></b></summary>

Провалилась `f_min` или минимум уехал дальше `rom_max_shift_m` (20 мкм) от номинала. Увеличьте
`reg`, уменьшите `θ₂…θ₇` и посмотрите `delivered_local_diagnostics.json`: поля `mode_rotation_risk`
и `relative_min_gap` показывают участок, где ломается структура мод.

</details>

<details>
<summary><b><code>numerical_failure</code> и <code>no_minimum</code></b></summary>

Затравка `minimum_seed_m` слишком близко к поверхности или `least_squares` не сошёлся. Поднимите
`z` затравки (например, с 70 до 200 мкм) и проверьте, что `initial_dc_V` внутри `[lo, hi]`.

</details>

<details>
<summary><b>GDS: «mapped electrode has no polygons»</b></summary>

Сверьте `layer` и `datatype` с тем, что лежит в файле. Список полигонов печатает
`python tools/gds_reader.py configs/generated_fivewire.gds --scale 1e-6`; `--scale` нужен, если
запись UNITS условная. Полигоны обходятся против часовой стрелки — при отрицательной площади
направление разворачивается само.

</details>

<details>
<summary><b>LAMMPS: «identical uids» в старом <code>sion/sion.py</code></b></summary>

Перезапустите ядро (IPython): старый стек LAMMPS держит состояние между вызовами. На `shuttle_sim`
это не влияет — LAMMPS там не используется.

</details>

<a id="structure"></a>

## Структура

```
Sion_ionfork/
├── README.md
├── assets/                       # иллюстрации, собираются из output/
├── tools/
│   ├── make_docs_assets.py       # рисунки и анимация по артефактам прогонов
│   ├── gds_reader.py             # чтение GDSII без внешних зависимостей
│   └── preview_readme.py         # рендер README в HTML, чтобы проверить вёрстку
├── shuttle_sim/
│   ├── __main__.py               # python -m shuttle_sim --config … --out …
│   ├── pipeline.py               # Runner: prepare/grid/commands/rom/evaluate/optimize/robust/run
│   ├── field.py                  # SionTrap, load_trap, Local, Rejected
│   ├── control.py                # четыре семейства параметризаций
│   ├── voltage.py hardware.py    # инверсия напряжений и аппаратная модель
│   ├── dynamics.py               # трёхмерный гауссов пакет, exact и dop853
│   ├── safety.py observables.py  # сквозные проверки и наблюдаемые
│   ├── calibration.py closed_loop.py transport.py
│   ├── ga/                       # островной алгоритм: engine, island, migration, fitness, cuda
│   └── dashboard/                # сервер и статика дашборда
├── sion/                         # исходный пакет: electrode, pylion, sion.py (LAMMPS)
├── configs/                      # 13 конфигураций и пример GDS
├── docs/                         # architecture.md, ga_architecture.md
├── examples/                     # ноутбуки: исходный SION, GDS, моды
├── output/                       # артефакты прогонов
└── tests/                        # test_regression.py
```

`pytest tests/test_regression.py -q`

Рисунки в `assets/` пересобираются одной командой; данные берутся из `output/`, поэтому картинки
не расходятся с числами в тексте:

```bash
python tools/make_docs_assets.py                 # всё
python tools/make_docs_assets.py --only wave traj
```

<a id="cite"></a>

## Цитирование

> *Surface trap with adjustable ion couplings for scalable and parallel gates*,
> Phys. Rev. A **109**, 022605 (2024) — https://journals.aps.org/pra/abstract/10.1103/PhysRevA.109.022605

Лицензия: **GPL-3.0-or-later**.

---

<sub>Состояние калибровки `unmeasured_simulation_assumptions`, идентификатор аппаратной модели
`declared-simulation-chain-v1`, признак `experiment_ready: false` — до подключения реальной установки.</sub>
