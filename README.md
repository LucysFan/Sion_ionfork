# SION — Surface Ion Shuttling

### Один ион, планарная ловушка, реальные напряжения на электродах — от геометрии до числа фононов

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

<sub>↑ Пять прогонов этого репозитория. Числа взяты из их `status.json` и `comparison.json` — не набраны руками.</sub>

**Транспорт иона — не «парабола в вакууме».** Здесь нет ни виртуальных score-API, ни скрытых 1D-моделей:
геометрия электродов → инверсия напряжений → ЦАП с квантованием → RC-фильтр → задержки → 3D-динамика гауссова пакета →
`n_total` в фононах. Всё в СИ, всё воспроизводимо, всё лежит в `output/<run>/`.

<br>

![Ион едет по ловушке](assets/anim-shuttle.gif)

<sub>↑ Реальный прогон `output/shuttle_reference`: ион проходит 30 мкм туда и обратно, справа — доставленные напряжения и `x(t)`. GIF собран из тех же CSV, что и графики (см. [как это сделано](#assets)).</sub>

<br>

| Что внутри | Коротко |
|---|---|
| **Ловушка** | `sion_five_wire` (аналитическая геометрия) или **GDS** — ваши слои и datatype → ваши каналы |
| **Управление** | `θ ∈ ℝ⁸`: 2 гена time-warp + 6 генов SVD-коррекции (низкая размерность для closed-loop) |
| **Железо** | DAC (биты, LSB, клиппинг), ZOH, точный RC-каскад, gain/offset, задержка, джиттер, slew |
| **Динамика** | 3D локальная гармоника, точный пропагатор или DOP853, ковариация 6×6 |
| **Приёмка** | номинал, multi-fidelity валидация, робастный ансамбль, hold-скан, safety-коды |
| **Оптимизация** | Powell (8 параметров) + островной GA (5 стратегий) + closed-loop с бюджетами |
| **Артефакты** | ~34 файла на прогон: конфиг, волновые формы, пути, диагностики, `status.json` |

| Проблема | Что делает SION | Результат |
|:---|:---|:---|
| Идеальный `V(t)` не доедет до электродов: ЦАП квантует, фильтр сглаживает, усилитель даёт gain/offset, есть задержка и slew-лимит. А геометрия электродов делает потенциал нелинейным — 1D-парабола здесь врёт. | Считает поле по реальной геометрии, инвертирует напряжения под нужную яму, прогоняет их через модель железа и только потом — через 3D-динамику. Ошибка каждого слоя видна отдельным числом. | `n_total = 0.0071` фонона, ошибка позиции `1.38 нм`, `f_min = 688 кГц` — и всё это проверено multi-fidelity валидацией, робастным ансамблем и safety-гейтами. |

**Как читать цифры ниже** (одна строка на метрику):

| Метрика | Хорошо | Плохо | Почему это важно |
|---|---|---|---|
| `n_total` | `0.001 … 0.01` фонона | `> 0.5` | нагрев, который потом надо охлаждать |
| `position_error_m` | нанометры | `> 5 мкм` | ион приехал не туда |
| `minimum_frequency_hz` | `> 400 кГц` | `< 400 кГц` | слабый конфайнмент = потеря иона |
| `max_deviation_V` / `rms` | десятки мВ | клиппинг, slew | железо не может доставить команду |

<a id="toc"></a>

## Содержание

**Начать**
- [Быстрый старт](#quickstart) · [Python API](#api) · [Что получается на выходе](#output)

**Понять**
- [Как это работает: 8 слоёв](#architecture) · [Модель железа](#hardware) · [Метрики и приоритеты](#metrics)
- [Геометрия из GDS](#gds) · [Конфиги](#configs)

**Посмотреть результаты**
- [Номинальный прогон](#results-nominal) · [Валидация](#results-validation) · [Робастность](#results-robust)
- [Островной GA](#results-ga) · [Живой дашборд](#dashboard)

**Поддерживать**
- [Как добавить картинку, схему или анимацию в README](#assets) · [Ограничения и честность](#honesty)
- [FAQ](#faq) · [Структура проекта](#structure) · [Цитирование](#cite)

---

<a id="quickstart"></a>

## Быстрый старт

```bash
git clone https://github.com/LucysFan/Sion_ionfork && cd Sion_ionfork
pip install -r requirements-shuttle.txt      # numpy, scipy, jinja2, matplotlib, python-gdsii, …
# LAMMPS/M-MPI НЕ нужны для shuttle_sim — они только для старого sion/sion.py
```

Один прогон целиком (номинал: 30 мкм, 15 мкс на плечо, round-trip, five-wire):

```bash
python -m shuttle_sim --config configs/shuttle_reference.json --out output/shuttle_reference
```

Через ~2–3 минуты в `output/shuttle_reference/` появляются 34 файла и печатается статус:

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

Остальные сценарии — та же команда, другой конфиг:

```bash
python -m shuttle_sim --config configs/shuttle_oneway.json      --out output/shuttle_oneway
python -m shuttle_sim --config configs/shuttle_repeated.json    --out output/shuttle_repeated
python -m shuttle_sim --config configs/shuttle_gds.json         --out output/shuttle_gds
python -m shuttle_sim --config configs/shuttle_variable_freq.json --out output/shuttle_variable_freq
python -m shuttle_sim --config configs/shuttle_reference.json   --out output/ref_full --full-field
```

<a id="api"></a>

### Python API

```python
import json, numpy as np
from pathlib import Path
from shuttle_sim.pipeline import Runner

cfg = json.loads(Path("configs/shuttle_reference.json").read_text())
runner = Runner(cfg)
runner.prepare()                     # 31 опорная точка, SVD-моды, сплайны по маршруту

theta = np.zeros(8)                  # 2 warp + 6 SVD-коррекций
res = runner.evaluate(theta, baseline="sta", exact_nodes=321, subdivisions=8)
print(res["n_total"], res["position_error_m"], res["minimum_frequency_hz"])
# 0.0049  1.4e-09  687910.0

robust = runner.robust(theta, count=12)      # ансамбль по объявленным неопределённостям
print(robust["pass_fraction"], robust["mean_n"], robust["worst_n"])

aware = runner.optimize(robust=True)         # Powell, bounds, чекпоинт, бюджет вызовов
print(aware["theta"], aware["loss"])
```

<a id="output"></a>

### Что получается на выходе

| Файл | Что внутри |
|---|---|
| `status.json` | приёмка: номинал / валидация / робаст / hold-скан / safety |
| `ideal/dac/delivered_waveform.csv` | команда → коды ЦАП → доставленный сигнал (15 каналов) |
| `delivered_path.csv` | реализованный путь иона `x,y,z(t)` после железа |
| `route_diagnostics.json`, `delivered_local_diagnostics.json` | позиция, ω, моды, перекрытия мод по маршруту |
| `comparison.json` | `smooth` / `sta` / `unaware` / `aware` на одной метрике |
| `validation.json` | cheap ROM vs 321 / 1281 узел vs DOP853, дрейфы `Δn` |
| `robust.json`, `hold_scan.json` | сценарии шума, mean/worst/CVaR, зависимость от паузы |
| `safety_report.json`, `fallback_state.json` | проверки границ и последнее безопасное состояние |
| `calibration_*.json`, `closed_loop_*.json` | идентификация gain/offset, ресинтез, low-dim closed-loop |
| `manifest.json`, `waveform_metadata.json` | версии, SHA256 входов и исходников, порядок каналов, единицы |

<a id="architecture"></a>

## Как это работает: 8 слоёв

![Архитектура](assets/fig-architecture.png)

Каждый слой — отдельный модуль с явным API и своим артефактом. Ничего не спрятано в ноутбуке.
Глубже: [`docs/architecture.md`](docs/architecture.md) — послойная сверка со спецификацией,
[`docs/ga_architecture.md`](docs/ga_architecture.md) — острова, миграция, зал славы и дашборд.

| № | Слой | Модуль | Ключевая ответственность |
|---|---|---|---|
| 1 | Поле | `field.py` | `SionTrap`: псевдопотенциал RF + DC, поиск минимума, гессиан, моды |
| 2 | Управление | `control.py` | `θ ∈ ℝ⁸`: time-warp, огибающие, SVD-моды; fixed/variable ω |
| 3 | Инверсия | `voltage.py` | 3 силы + 1 кривизна, `SLSQP`, bounds, регуляризация, null-space |
| 4 | Железо | `hardware.py` | DAC/ZOH, RC-каскад через `expm`, gain/offset/skew/delay, slew |
| 5 | Динамика | `dynamics.py` | 3D гауссов пакет, `exact` и `DOP853`, ковариация и `n_total` |
| 6 | Робастность | `pipeline.robust` | ансамбль по gain/offset/delay/фильтру/stray/ω |
| 7 | Калибровка | `calibration.py` | `identify_from_residuals` → `update_model` → ресинтез |
| 8 | Closed-loop | `closed_loop.py` | 8 параметров, `SafetyLayer`, бюджет, история |

<details>
<summary><b>Формальная постановка и что именно оптимизируется</b></summary>

Энергия: `U(r) = U_RF(r) + Σ Vᵢ·φᵢ(r) − q·E_stray·r`, где `φᵢ` — потенциал i-го электрода из
`sion.electrode.System`, `U_RF` — псевдопотенциал. Производные берутся через
`system.individual_potential(r, derivative)`.

Инверсия напряжений (слой 3):

```
A = [ Gᵀ·L/q ; aᵀ·H·a·L²/q ]          (3 силы + 1 осевая кривизна)
b = [ −∇U_RF·L/q ; (mω² − aᵀH_RF a)·L²/q ]
min ½‖V−prev‖² + ½·reg·‖V‖²   s.t. A·V = b,  V ∈ [lo, hi]
```

Затем проверка `‖local − target‖ < 5 нм`. Радиальные моды свободны: лапласиан не даёт задать весь
гессиан.

Три «пути» разведены и не путаются (см. `path_conventions.json`):

| Путь | Откуда берётся |
|---|---|
| `user_route` | `start + s·distance·direction`, `s ∈ [0,1]` — то, что задал пользователь |
| `nominal_well_path` | `voltage_spline(s)` → `trap.local()` → `route_r`, `route_H` |
| `realized_path` | доставленные напряжения → точная ROM-реконструкция → динамика |
| `optimized_path` | `q(route)` в пространстве параметров; оценивается **после** железа |

Оптимизируется `q(route)`, валидируется `realized_path`.

</details>

<details>
<summary><b>Параметризации: почему 8 чисел, а не 200 отсчётов</b></summary>

| Семейство | Где | Параметры | Идея |
|---|---|---|---|
| Trajectory | `control.TrajectoryParameterization` | `θ₀, θ₁` | `s(t) = x + θ₀·env + θ₁·env·(2s−1)`, `env = 16s²(1−s)²` — ноль значения и наклона на границах |
| FrequencyProfile | `control.FrequencyProfile` | `a, b` | `ω(s) = ω₀·(1 + a·env + b·env·(2s−1))`, ограничение 15% |
| VoltageKnot | `control.VoltageKnotParameterization` | узлы сплайна | для сравнения: кусочно-сплайн по опорным напряжениям |
| CorrectiveBasis | `control.CorrectiveBasis` | `θ₂…θ₇` | `ΔV = (e₀,e₁,e₂)·Θ·R_svd`, топ-2 SVD `route_V` |

Итого `θ ∈ ℝ⁸`: 2 гена времени + 6 генов в подпространстве, которое реально двигает минимум.
Попытка подать 1000 параметров → `Rejected('high_dim')`.

</details>

<a id="hardware"></a>

## Модель железа: что именно съедает идеальный сигнал

![Волновые формы](assets/fig-waveforms.png)

```
commands_V (N × C)
  → квантование DAC:  lsb = (max−min)/(2^bits − 1)
  → ZOH на сетке k/fs
  → плотная сетка dt = 1/(fs·subdiv), subdiv 4…16
  → RC-каскад: точное expm(M·dt) на ZOH-сетке, состояние с предысторией
  → gain·held + offset (поканально)
  → skew + задержка + per-sample time-jitter (детерминирован по хешу команд)
  → проверки: клиппинг → hardware_saturation,  max|ΔV| → slew_limit
```

В номинальном прогоне железо «забирает» совсем немного, и это видно числом:
`max|ΔV| = 96.7 мВ`, `rms = 27.2 мВ`, `LSB = 366 мкВ`, клиппингов **0**.
Идеальный `V(t)` и доставленный `V_delivered(t)` — разные сигналы, и оптимизатор видит именно второй.

<details>
<summary><b>Тайминг, джиттер и отчёт об искажениях</b></summary>

- `trigger_latency_s` — общий статический сдвиг.
- `jitter_s` — per-sample глобальный time-jitter, `rng.uniform(-jitter, jitter, len(t))`,
  применяется как `y(t − delay − trig − time_jitter(t))` с интерполяцией.
- `skew_per_channel_s` — статический поканальный сдвиг (бывший `jitter_per_channel_s`).
- `distortion.json`: `max_command_jump_V`, `lsb_V`, `max_deviation_V`, `rms_deviation_V`,
  `channel_max_deviation_V`, `delay_s`, `trigger_latency_s`, `jitter_s`, `skew_per_channel_s`,
  `time_jitter_s`, `max_time_jitter_s`, `jitter_model`, `hold`, `filter`, `hardware_identifier`,
  `calibration_state`.

`ideal=True` — прогон без квантования и фильтра, один стейт на команду (для сравнения
«hardware-unaware» с «hardware-aware»).

</details>

<a id="metrics"></a>

## Метрики, приоритеты, приёмка

Иерархия из спеки §8 — нарушение более высокого уровня отменяет всё остальное:

```
safety  →  confinement  →  hardware realizability  →  excitation / accuracy  →  speed
```

Скалярные потери (единая метрика, покомпонентно пишется в артефакты):

```
L = log1p(n/n₀) + log1p((pos/tol)²) + log1p((well/tol)²) + 1e-4·‖θ‖²  + hardware-штрафы
robust: 0.35·mean + 0.65·max по train-сценариям
```

| Метрика | Где | Смысл |
|---|---|---|
| `n_total`, `n_modes` | везде | нагрев в фононах, бюджет `≤ 0.5` |
| `position_error_m` | везде | ошибка позиции, допуск `5 мкм` |
| `minimum_frequency_hz` | везде | конфайнмент, порог `400 кГц` |
| `max_deviation_V`, `rms_deviation_V` | `distortion.json` | насколько железо исказило команду |
| `mean_n`, `worst_n`, `cvar_20pct_n`, `pass_fraction` | `robust.json` | хвост распределения, а не только среднее |
| `refinement_delta_n`, `integrator_delta_n` | `validation.json` | доверие к дешёвой модели |
| `optimization_improved_over_baseline` | `status.json` | честный флаг: оптимизация реально лучше нулевой `θ` |

Коды отказа (`Rejected`): `no_minimum`, `weak_confinement`, `infeasible_voltages`,
`hardware_saturation`, `slew_limit`, `route_corridor`, `rom_trust_region`, `numerical_failure`,
`integrity_failure`, `blocked_by_missing_data`, `high_dim`, `invalid_input`,
`optimizer_budget_exhaustion`.

<a id="results-nominal"></a>

## Результаты: номинальный прогон

![Траектория](assets/fig-trajectory.png)

30 мкм туда и обратно за 15 мкс на плечо, пауза 2 мкс: **`n_total = 0.0071` фонона** (это ~1.4 %
бюджета 0.5), ошибка позиции **1.38 нм** при допуске 5 мкм, конфайнмент ни разу не падает ниже
**688 кГц** при пороге 400 кГц.

![Частоты вдоль маршрута](assets/fig-frequencies.png)

<sub>Секулярные частоты вдоль маршрута — ROM-диагностика по 41 опорной точке. Слабая ось (ω₁) задаёт
запас по конфайнменту, две другие — радиальные моды.</sub>

<a id="results-validation"></a>

## Результаты: валидация (multi-fidelity)

![Валидация](assets/fig-validation.png)

Один и тот же прогон считается четырьмя способами. Ответ почти не двигается:

| Сравнение | `Δn` (модуль) |
|---|---|
| дешёвая ROM → точная (321 узел) | `1.7e-4` |
| уточнение сетки до 1281 узла | `1.4e-4` |
| грубая → тонкая модель | `2.2e-3` |
| смена интегратора (`exact` → `DOP853`) | `3.0e-7` |

Все дрейфы ниже допуска `0.01` фонона, `convergence_passed = true`. Это и есть смысл
multi-fidelity: оптимизатор может бегать на дешёвой модели, а решение проверяется на точной.

<a id="results-robust"></a>

## Результаты: робастность

![Робастность](assets/fig-robust.png)

6 сценариев с объявленными (не «измеренными») возмущениями: gain, offset, задержка, фильтр,
stray-поле, масштаб частоты. Все приняты (`pass_fraction = 1.0`), но хвост честно показывается:

| | `n_total` | доля бюджета 0.5 |
|---|---|---|
| mean | `0.0269` | 5 % |
| CVaR 20 % | `0.0744` | 15 % |
| worst | `0.0764` | 15 % |

Обратите внимание: `mean` в 6 раз меньше `worst`. Именно поэтому в пайплайне агрегация
`0.35·mean + 0.65·max`, а не среднее.

<a id="results-ga"></a>

## Островной GA: не один локальный оптимум, а семейство маршрутов

![Island GA](assets/fig-ga.png)

Пять островов живут одновременно и не схлопываются в одну нишу:

| Остров | Baseline | Уклон | Геном |
|---|---|---|---|
| **S** | `smooth` (10s³−15s⁴+6s⁵) | гладкая адиабатика, малая мутация | 8 |
| **A** | `sta` (+ `s''/(Tω)²`) | любит короткие `T`, чувствителен к фильтру | 8 |
| **V** | `sta` | переменная частота, спасает 100 мкм при ±12 В | 10 |
| **R** | `sta` | отбор по `worst`, а не по `mean` | 8 |
| **H** | `sta` | штраф за искажение фильтром (`rms`, `max_deviation`) | 8 |

Миграция по кольцу `S→A→V→R→H→S` каждые 5 поколений, зал славы с фильтром разнообразия
(нормализованное θ-расстояние ≥ 0.12), чекпоинт после каждой генерации.

Демо-прогон `output/ga_demo` (8 поколений, 106 с): лучший фитнес **0.00236**, `n_total = 0.00125`,
ошибка позиции **1.25 нм**. Лучший геном — 10 чисел вместо 200 отсчётов напряжения.

```bash
# островной GA с живым дашбордом
python -m shuttle_sim.ga --config configs/shuttle_ga_island.json --out output/ga_run \
       --dashboard --port 8000 --generations 40
```

<a id="dashboard"></a>

## Живой дашборд

![Дашборд](assets/fig-dashboard.png)

GA **никогда не ждёт дашборд**: состояние пишется атомарно (`state.json.tmp → state.json`) после
каждой генерации, сервер только читает файл.

```bash
# отдельный процесс, читает output/<run> и отдаёт статику + /api/state
python -m shuttle_sim.dashboard.server --out output/ga_demo --port 8000
```

В превью-окружении это `https://8000-<sandbox>.e2b.app`. Панели: фитнес по поколениям и островам,
разнообразие, гистограмма `n_total`, зал славы, Парето `n` vs `worst`, 15 каналов `V(t)`,
маршрут иона `r(t)`, вид ловушки сверху с анимацией кандидатов, граф миграций, статус CUDA
(если `cupy` установлен — иначе прозрачный numpy-fallback).

<a id="gds"></a>

## Геометрия из GDS

![GDS](assets/fig-gds.png)

Никакого «неявного порядка полигонов»: канал — это явная пара `(layer, datatype)` из `electrode_map`.
RF задаётся отдельно, пересечение RF и DC-каналов проверяется.

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
<summary><b>Почему в примере GDS масштаб условный</b></summary>

`configs/generated_fivewire.gds` — файл-пример для проверки маппинга слоёв, а не реальный чип:
у него в записи UNITS стоит заглушка, поэтому координаты топологии показываются в единицах GDS.
Figure `fig-gds.png` подписана именно так, чтобы никто не прочитал картинку как размеры устройства.
Для настоящей геометрии укажите корректный UNITS в GDS — `load_trap` переводит единицы файла в
метры через `lib_unit / L`.

</details>

<a id="configs"></a>

## Конфиги

| Конфиг | Ловушка | Протокол | Что демонстрирует |
|---|---|---|---|
| `shuttle_reference.json` | five-wire, RF 20 МГц | round-trip 30 мкм / 15 мкс | номинал, базовая приёмка |
| `shuttle_oneway.json` | five-wire | one-way 30 мкм | тот же пайплайн, другой протокол |
| `shuttle_repeated.json` | five-wire | repeated ×2, 20 мкм | фильтр и динамика не сбрасываются на разворотах |
| `shuttle_gds.json` | GDS, слои 10–26 | round-trip 30 мкм | маппинг слоёв без скрытого порядка |
| `shuttle_variable_freq.json` | five-wire | round-trip, `coeff [0.03, −0.015]` | переменная частота конфайнмента |
| `shuttle_90um_30MHz.json` | five-wire, h = 90 мкм, RF 30 МГц, ±12 В | one-way 30 мкм | строгая аппаратная постановка |
| `shuttle_90um_60MHz.json` | five-wire, h = 90 мкм, RF 60 МГц, ±12 В | one-way 30 мкм | та же геометрия на удвоенной RF |
| `shuttle_90um_100um_strict_30MHz.json` | h = 90 мкм | round-trip 100 мкм / 10 мкс | где кончается бюджет напряжений |
| `shuttle_100um_10us.json`, `_50us.json` | ±20 В | one-way 100 мкм, 10 и 50 мкс | насколько длительность спасает дальний и быстрый шаттл |
| `shuttle_100um_strict.json` | ±20 В, GA | round-trip 100 мкм | GA-конфиг для строгого случая |
| `shuttle_ga_island.json` | five-wire | round-trip 30 мкм | 5 островов, миграция, зал славы |
| `shuttle_ga_isolated.json` | h = 90 мкм, variable ω | round-trip 30 мкм | остров с 10-генным геномом |

<details>
<summary><b>Все поля конфига — в одном месте</b></summary>

| Параметр | Поле | Ед. | Смысл |
|---|---|---|---|
| Масса / заряд | `trap.mass_amu`, `trap.charge_e` | а.е.м. / e | ⁴⁰Ca⁺ → 40, 1 |
| Масштаб длины | `trap.length_scale_m` | м | внутренний `L`, обычно `1e-6` |
| RF | `trap.rf_frequency_hz`, `trap.rf_peak_V` | Гц, В | одна частота, общая фаза, псевдопотенциал |
| Геометрия | `trap.geometry` или `trap.path` + `electrode_map` | мкм / м | аналитика или GDS |
| DC-каналы | `trap.dc_indices`, `lower_V`, `upper_V`, `initial_dc_V` | В | только эти электроды крутим |
| Старт | `trap.minimum_seed_m` | м | затравка для поиска минимума |
| Железо | `hardware.*` | Гц, В, с | дискретизация, фильтр, gain/offset/delay, slew |
| Транспорт | `task.distance_m`, `duration_s`, `hold_s`, `direction`, `protocol`, `cycles` | м, с, — | что и как везём |
| Качество | `task.min_frequency_hz`, `max_n_total`, `position_tolerance_m` | Гц, фонон, м | приёмка |
| Неопределённости | `uncertainty.*_sigma` | В, с, — | объявленные, а не измеренные |

</details>

<a id="assets"></a>

## Как добавить картинку, схему или анимацию в README

Все иллюстрации в этом файле лежат в `assets/` и **генерируются из артефактов прогонов**, а не
рисуются руками «на глазок». Это принципиально: картинка не может разойтись с числами.

```bash
python tools/make_docs_assets.py                 # все PNG + GIF
python tools/make_docs_assets.py --only wave traj   # только нужные
```

| Что нужно | Как это делается здесь |
|---|---|
| **График из результатов симуляции** | функция `fig_*(...)` в `tools/make_docs_assets.py` читает `output/<run>/*.json|csv` и сохраняет PNG в `assets/` |
| **Схема / чертёж** | `matplotlib.patches` (см. `fig-architecture`, `fig-gds`) или SVG из draw.io/Inkscape — GitHub рендерит SVG |
| **Анимация** | `matplotlib.animation` + `PillowWriter` → `assets/anim-*.gif`; для живого дашборда — скринкаст и `ffmpeg -i in.mp4 -vf "fps=12,scale=960:-1" out.gif` |
| **Скриншот дашборда** | `python -m shuttle_sim.dashboard.server --out output/ga_demo --port 8000`, затем снимок окна → `assets/dashboard-live.png` |
| **Картинка-«карточка» с числами** | `fig_scorecard` — читает `status.json` + `comparison.json` всех прогонов |

Добавить новую фигуру — это ~15 строк: функция читает артефакт и сохраняет PNG.

```python
# tools/make_docs_assets.py
def fig_duration_scan():
    rows = json.loads((RUN / "duration_scan.json").read_text())
    t = [r["duration_s"] * 1e6 for r in rows]
    n = [r["metrics"]["n_total"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, n, "o-", color=ACCENT)
    _style(ax, "Длительность плеча vs нагрев", "T, мкс", "n_total, фононы")
    _save(fig, "fig-duration-scan.png")

JOBS["duration"] = fig_duration_scan        # регистрируем в словаре JOBS
# запуск: python tools/make_docs_assets.py --only duration
```

Вставка в README — обычный markdown:

```markdown
![Команда и доставленный сигнал](assets/fig-waveforms.png)
<sub>Подпись: что это, откуда данные, какие единицы.</sub>

![Анимация](assets/anim-shuttle.gif)
```

Правила, которые здесь соблюдаются (и стоит соблюдать дальше):

1. **Имя = смысл**: `fig-<тема>.png`, `anim-<тема>.gif`, никаких `image1.png`.
2. **Ширина**: графики — 1200–1300 px по ширине, dpi 145–160; GIF — ≤ 1 МБ (GitHub показывает
   большие файлы медленно), поэтому кадры 960–1200 px и 12–15 fps.
3. **Подпись обязательна**: что на картинке, из какого прогона, какие единицы.
4. **Числа не вписываются руками** — только чтение из `output/`. Если прогон пересчитан,
   `python tools/make_docs_assets.py` обновит и картинки.
5. **Схема ≠ измерение**: если это схема электродов, в подписи так и написано.

<a id="honesty"></a>

## Ограничения и честность

Это **симуляция**, а не работающая установка, и репозиторий не притворяется обратным:

- `calibration_state: unmeasured_simulation_assumptions`, `hardware_identifier: declared-simulation-chain-v1`,
  `experiment_ready: false` до подключения реального железа;
- модель поля — 3D локальная гармоника; полное RF-микродвижение и негауссова квантовая эволюция
  **не моделируются** (явно записано в `manifest.json` и `validation.json`);
- распределения неопределённостей в `robust` — объявленные, не измеренные;
- калибровочный демо-датасет синтетический: `calibration_resynthesis.json` честно пишет
  `improved: false` (0.0049 → 0.0236) — идентификация по трём точкам ухудшила модель, и это видно,
  а не спрятано;
- closed-loop демо (`closed_loop_emulated.json`) упирается в бюджет: `evaluations 11`,
  `optimizer_success: false`, `termination: Maximum number of function evaluations…`;
  loss при этом упал с `0.0472` до `0.0158`;
- попытка подать в closed-loop 1000 параметров → `closed_loop_param_check.json`:
  `rejected_as_expected: true`;
- калибровка без внешних данных не выдумывает решение: `calibration_blocked.json` →
  `blocked_by_missing_external_data`;
- сравнение `GA smooth` vs `pipeline sta` валидно только на одинаковых `trap`/`hardware`/
  `exact_nodes`/бюджете — иначе это разные задачи, а не «лучше/хуже».

<a id="faq"></a>

## FAQ / возможные проблемы

<details>
<summary><b>infeasible_voltages на 100 мкм за 10 мкс при ±12 В</b></summary>

Ожидаемо: `SLSQP` не находит `V ∈ [−12, 12]` для дальней точки при `ω = 400 кГц`. Рычаги по порядку
дешевизны: (1) увеличить `duration_s` — адиабатика требует времени; (2) поднять границы до ±20 В
и не забыть про `hardware.dac_min/max`; (3) включить variable-ω (`coeff [0.03, −0.015]`), чтобы
провалить частоту в середине маршрута. Если задан `task.durations_scan_s`, свежий прогон сам
положит `duration_scan.json` — по нему и видно, где бюджет кончается.

</details>

<details>
<summary><b>slew_limit / hardware_saturation</b></summary>

Уменьшить `max_jump_V`, поднять `sample_rate_hz` или отодвинуть `θ` (warp) ближе к нулю.
Проверьте `distortion.json`: `max_command_jump_V` покажет, где именно перегруз.

</details>

<details>
<summary><b>weak_confinement / rom_trust_region</b></summary>

Провалилась `f_min` или минимум уехал дальше `rom_max_shift_m` (20 мкм) от номинала.
Увеличьте `reg`, уменьшите `θ₂…θ₇`, проверьте `delivered_local_diagnostics.json` —
`mode_rotation_risk` и `relative_min_gap` подскажут, где ломается картина мод.

</details>

<details>
<summary><b>numerical_failure / no_minimum</b></summary>

Затравка `minimum_seed_m` слишком близко к поверхности или `least_squares` не сошёлся.
Поднимите `z` затравки (например, 70 → 200 мкм) и проверьте, что `initial_dc_V` внутри `[lo, hi]`.

</details>

<details>
<summary><b>GDS: «mapped electrode has no polygons»</b></summary>

Сверьте `layer/datatype` с содержимым файла (наш `tools/gds_reader.py` печатает все полигоны:
`python tools/gds_reader.py configs/generated_fivewire.gds --scale 1e-6`; `--scale` нужен, если в
файле заглушка вместо UNITS). Полигоны должны быть CCW — код
разворачивает их сам, если площадь отрицательна. И помните про UNITS: масштаб берётся из файла.

</details>

<details>
<summary><b>LAMMPS: «identical uids» (старый sion/sion.py)</b></summary>

Перезапустите kernel (IPython): старый стек LAMMPS держит состояние между вызовами.
Для `shuttle_sim` LAMMPS не нужен вообще.

</details>

<a id="structure"></a>

## Структура проекта

```
Sion_ionfork/
├── README.md                     # этот файл
├── assets/                       # все картинки и GIF для README (генерируются из output/)
├── tools/
│   ├── make_docs_assets.py       # сборка фигур и анимации из артефактов прогонов
│   └── gds_reader.py             # минимальный GDSII-ридер для картинок (без gdspy)
├── shuttle_sim/                  # симуляционный стек (СИ, bounded, чекпоинты)
│   ├── __main__.py               # CLI: python -m shuttle_sim --config … --out …
│   ├── pipeline.py               # Runner: prepare/grid/commands/rom/evaluate/optimize/robust/run
│   ├── field.py                  # SionTrap, load_trap, Local, Rejected
│   ├── control.py                # 4 семейства параметризаций
│   ├── voltage.py hardware.py    # инверсия и модель железа
│   ├── dynamics.py               # 3D гаусс, exact/dop853, валидация центроида
│   ├── safety.py observables.py  # сквозные слои
│   ├── calibration.py closed_loop.py transport.py
│   ├── ga/                       # островной GA: engine, island, migration, fitness, cuda
│   └── dashboard/                # сервер + статика живого дашборда
├── sion/                         # оригинальный пакет: electrode + pylion + sion.py (LAMMPS)
├── configs/                      # 13 конфигов + пример GDS
├── docs/                         # architecture.md, ga_architecture.md
├── examples/                     # ноутбуки (старый SION, GDS, режимы)
├── output/                       # артефакты прогонов (по папке на прогон)
└── tests/                        # test_regression.py (safety, калибровка, closed-loop)
```

Проверки: `pytest tests/test_regression.py -q` (быстрый конфиг: 15 узлов, 4 оценки).

<a id="cite"></a>

## Цитирование

Если публикуете результаты, полученные с `SION`, цитируйте:

> *Surface trap with adjustable ion couplings for scalable and parallel gates*,
> Phys. Rev. A **109**, 022605 (2024) — https://journals.aps.org/pra/abstract/10.1103/PhysRevA.109.022605

Лицензия: **GPL-3.0-or-later**.

---

<sub>**Всё, что заявлено как измеренное, объявлено как симуляция.** `calibration_state: unmeasured_simulation_assumptions`,
`hardware_identifier: declared-simulation-chain-v1`, `experiment_ready: false` — до подключения реальной установки.</sub>
