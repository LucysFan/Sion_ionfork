# SION — Surface Ion Shuttling

### Транспорт одиночного иона в планарной ловушке: расчёт от геометрии электродов до числа фононов

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

*Рис. 1.* Результаты пяти конфигураций, входящих в репозиторий. Значения взяты из файлов `status.json`
и `comparison.json` соответствующих прогонов.

Репозиторий содержит вычислительный конвейер для синтеза и проверки протоколов шаттлинга одиночного
иона в планарной ловушке. Потенциал вычисляется по заданной геометрии электродов; напряжения
находятся решением задачи инверсии с ограничениями; затем они проходят через модель аппаратного
тракта (квантование ЦАП, ZOH, RC-фильтр, усиление, смещение, задержка, slew-лимит) и только после
этого используются в трёхмерной динамике гауссова пакета. Результат выражается в единицах,
принятых в эксперименте: нагрев `n_total` в фононах, ошибка позиции и минимальная секулярная
частота. Все величины задаются в СИ.

![Ион в ловушке](assets/anim-shuttle.gif)

*Рис. 2.* Номинальный прогон `output/shuttle_reference`: положение иона при движении на 30 мкм и
обратно (слева), доставленные напряжения и координата `x(t)` (справа). Анимация построена по данным
`delivered_path.csv` и `delivered_waveform.csv`; порядок построения описан в разделе
[«Иллюстрации в документации»](#assets).

**Состав конвейера**

| Компонент | Реализация |
|---|---|
| Ловушка | `sion_five_wire` (аналитическая геометрия) или `gds` — каналы задаются парами `(layer, datatype)` |
| Управление | `θ ∈ ℝ⁸`: 2 параметра time-warp и 6 параметров SVD-коррекции |
| Аппаратный тракт | квантование ЦАП (разрядность, LSB, клиппинг), ZOH, RC-каскад, усиление и смещение, задержка, джиттер, slew-лимит |
| Динамика | трёхмерная локальная гармоническая модель, точный пропагатор или DOP853, ковариация 6×6 |
| Приёмка | номинальный прогон, multi-fidelity валидация, робастный ансамбль, сканирование паузы, коды отказа |
| Оптимизация | Powell по 8 параметрам, островной генетический алгоритм (5 стратегий), closed-loop с фиксированным бюджетом |
| Артефакты | около 34 файлов на прогон: конфигурация, волновые формы, траектория, диагностики, `status.json` |

| Задача | Реализация | Результат номинального прогона |
|:---|:---|:---|
| Идеальный сигнал `V(t)` искажается аппаратурой (квантование, фильтрация, усиление, задержка, slew), а потенциал нелинейно зависит от геометрии электродов; аппроксимация одномерной параболой вносит систематическую ошибку. | Потенциал вычисляется по геометрии; напряжения находятся инверсией при ограничениях; искажения вносятся моделью тракта до расчёта динамики; каждая ступень имеет отдельный артефакт с числовой диагностикой. | `n_total = 0.0071` фонона, ошибка позиции `1.38 нм`, `f_min = 688 кГц`; результат подтверждён multi-fidelity валидацией и робастным ансамблем. |

**Интерпретация метрик**

| Метрика | Допустимое значение | Условие отказа | Физический смысл |
|---|---|---|---|
| `n_total` | ≤ 0.5 фонона | `> 0.5` | нагрев, требующий последующего охлаждения |
| `position_error_m` | ≤ 5 мкм | `> position_tolerance_m` | отклонение от целевой позиции |
| `minimum_frequency_hz` | ≥ 400 кГц | `weak_confinement` | ослабление удержания иона |
| `max_deviation_V`, `rms_deviation_V` | десятки мВ, клиппинг отсутствует | `hardware_saturation`, `slew_limit` | команда не воспроизводится аппаратурой |

<a id="toc"></a>

## Содержание

**Начало работы**
- [Быстрый старт](#quickstart) · [Программный интерфейс](#api) · [Артефакты прогона](#output)

**Описание метода**
- [Архитектура: восемь слоёв](#architecture) · [Аппаратная модель](#hardware) · [Метрики и приоритеты](#metrics)
- [Геометрия из GDS](#gds) · [Конфигурации](#configs)

**Результаты**
- [Номинальный прогон](#results-nominal) · [Валидация](#results-validation) · [Робастность](#results-robust)
- [Островной генетический алгоритм](#results-ga) · [Дашборд в реальном времени](#dashboard)

**Сопровождение**
- [Иллюстрации в документации](#assets) · [Область применимости и ограничения](#honesty)
- [Диагностика отказов](#faq) · [Структура проекта](#structure) · [Цитирование](#cite)

---

<a id="quickstart"></a>

## Быстрый старт

```bash
git clone https://github.com/LucysFan/Sion_ionfork && cd Sion_ionfork
pip install -r requirements-shuttle.txt      # numpy, scipy, jinja2, matplotlib, python-gdsii, …
# LAMMPS и MS-MPI требуются только для устаревшего модуля sion/sion.py
```

Полный номинальный прогон (30 мкм, 15 мкс на плечо, round-trip, геометрия five-wire):

```bash
python -m shuttle_sim --config configs/shuttle_reference.json --out output/shuttle_reference
```

По завершении (около 2–3 минут) в каталоге `output/shuttle_reference/` записываются 34 файла и
выводится сводка приёмки:

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

Остальные конфигурации запускаются той же командой:

```bash
python -m shuttle_sim --config configs/shuttle_oneway.json        --out output/shuttle_oneway
python -m shuttle_sim --config configs/shuttle_repeated.json      --out output/shuttle_repeated
python -m shuttle_sim --config configs/shuttle_gds.json           --out output/shuttle_gds
python -m shuttle_sim --config configs/shuttle_variable_freq.json --out output/shuttle_variable_freq
python -m shuttle_sim --config configs/shuttle_reference.json     --out output/ref_full --full-field
```

<a id="api"></a>

### Программный интерфейс

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

aware = runner.optimize(robust=True)         # Powell, ограничения, чекпоинт, фиксированный бюджет
print(aware["theta"], aware["loss"])
```

<a id="output"></a>

### Артефакты прогона

| Файл | Содержание |
|---|---|
| `status.json` | сводка приёмки: номинал, валидация, робастность, сканирование паузы, safety |
| `ideal_waveform.csv`, `dac_waveform.csv`, `delivered_waveform.csv` | команда, коды ЦАП и доставленный сигнал по 15 каналам |
| `delivered_path.csv` | реализованная траектория иона `x, y, z (t)` после аппаратного тракта |
| `route_diagnostics.json`, `delivered_local_diagnostics.json` | позиция, частоты, моды и их перекрытие вдоль маршрута |
| `comparison.json` | сравнение `smooth`, `sta`, `unaware` и `aware` на единой метрике |
| `validation.json` | сравнение дешёвой ROM, моделей на 321 и 1281 узле, DOP853; величины дрейфа `Δn` |
| `robust.json`, `hold_scan.json`, `duration_scan.json` | сценарии возмущений, статистика `mean`/`worst`/`CVaR`, зависимость от паузы и длительности |
| `safety_report.json`, `fallback_state.json` | результаты проверок и последнее безопасное состояние |
| `calibration_*.json`, `closed_loop_*.json` | идентификация `gain`/`offset`, ресинтез, low-dim closed-loop |
| `manifest.json`, `waveform_metadata.json` | версии среды, SHA-256 входных данных и исходников, порядок каналов, единицы |

<a id="architecture"></a>

## Архитектура: восемь слоёв

![Архитектура](assets/fig-architecture.png)

*Рис. 3.* Слои конвейера, соответствующие модули и артефакты.

Каждый слой реализован отдельным модулем с явным программным интерфейсом и собственным артефактом.
Дополнительные сведения: [`docs/architecture.md`](docs/architecture.md) — послойное сопоставление со
спецификацией; [`docs/ga_architecture.md`](docs/ga_architecture.md) — структура островного алгоритма,
миграция, зал славы и дашборд.

| № | Слой | Модуль | Ответственность |
|---|---|---|---|
| 1 | Поле | `field.py` | `SionTrap`: псевдопотенциал RF и вклад DC, поиск минимума, гессиан, моды |
| 2 | Управление | `control.py` | `θ ∈ ℝ⁸`: time-warp, огибающие, SVD-моды; фиксированная и переменная частота |
| 3 | Инверсия | `voltage.py` | три условия баланса сил и одно условие кривизны, `SLSQP`, ограничения, регуляризация |
| 4 | Аппаратный тракт | `hardware.py` | ЦАП и ZOH, RC-каскад через `expm`, усиление, смещение, задержка, slew |
| 5 | Динамика | `dynamics.py` | трёхмерный гауссов пакет, методы `exact` и `DOP853`, ковариация и `n_total` |
| 6 | Робастность | `pipeline.robust` | ансамбль по `gain`, `offset`, задержке, фильтру, stray-полю и масштабу частоты |
| 7 | Калибровка | `calibration.py` | `identify_from_residuals` → `update_model` → ресинтез |
| 8 | Closed-loop | `closed_loop.py` | 8 параметров, `SafetyLayer`, фиксированный бюджет, журнал итераций |

<details>
<summary><b>Формальная постановка и объект оптимизации</b></summary>

Энергия системы: `U(r) = U_RF(r) + Σ Vᵢ·φᵢ(r) − q·E_stray·r`, где `φᵢ` — потенциал i-го электрода
из `sion.electrode.System`, `U_RF` — псевдопотенциал. Производные вычисляются через
`system.individual_potential(r, derivative)`.

Инверсия напряжений (слой 3):

```
A = [ Gᵀ·L/q ; aᵀ·H·a·L²/q ]          (3 условия баланса сил + 1 условие кривизны)
b = [ −∇U_RF·L/q ; (mω² − aᵀH_RF a)·L²/q ]
min ½‖V−prev‖² + ½·reg·‖V‖²   при  A·V = b,  V ∈ [lo, hi]
```

Далее проверяется условие `‖local − target‖ < 5 нм`. Радиальные моды остаются свободными: лапласиан
не позволяет задать полный гессиан.

Используются четыре различных определения маршрута (файл `path_conventions.json`):

| Маршрут | Определение |
|---|---|
| `user_route` | `start + s·distance·direction`, `s ∈ [0,1]`, задаётся конфигурацией |
| `nominal_well_path` | `voltage_spline(s)` → `trap.local()` → `route_r`, `route_H` |
| `realized_path` | доставленные напряжения → точная ROM-реконструкция → расчёт динамики |
| `optimized_path` | `q(route)` в пространстве параметров; оценивается после аппаратного тракта |

Оптимизации подлежит `q(route)`, валидации — `realized_path`.

</details>

<details>
<summary><b>Параметризации: обоснование размерности 8</b></summary>

| Семейство | Модуль | Параметры | Определение |
|---|---|---|---|
| `TrajectoryParameterization` | `control.py` | `θ₀, θ₁` | `s(t) = x + θ₀·env + θ₁·env·(2s−1)`, `env = 16s²(1−s)²`; значение и наклон на границах равны нулю |
| `FrequencyProfile` | `control.py` | `a, b` | `ω(s) = ω₀·(1 + a·env + b·env·(2s−1))`, ограничение на отклонение 15 % |
| `VoltageKnotParameterization` | `control.py` | узлы сплайна | кусочно-сплайновая аппроксимация опорных напряжений (для сравнения) |
| `CorrectiveBasis` | `control.py` | `θ₂…θ₇` | `ΔV = (e₀,e₁,e₂)·Θ·R_svd`, где `R_svd` — две старшие сингулярные моды `route_V` |

Итоговый вектор `θ ∈ ℝ⁸` содержит 2 параметра временной деформации и 6 параметров в подпространстве,
которое непосредственно смещает минимум потенциала. Запрос размерности 1000 отклоняется
(`Rejected('high_dim')`).

</details>

<a id="hardware"></a>

## Аппаратная модель

![Волновые формы](assets/fig-waveforms.png)

*Рис. 4.* Изменение команды `ΔV(t)` относительно начального значения (вверху) и отклонение
доставленного сигнала от команды по каналам (внизу).

Последовательность преобразований:

```
commands_V (N × C)
  → квантование ЦАП:  lsb = (max − min)/(2^bits − 1)
  → ZOH на сетке k/fs
  → плотная сетка dt = 1/(fs·subdiv), subdiv 4…16
  → RC-каскад: точное вычисление expm(M·dt) на ZOH-сетке с предысторией состояния
  → gain·held + offset (поканально)
  → skew, задержка и per-sample time-jitter (детерминированный по хешу команд)
  → проверки: клиппинг → hardware_saturation;  max|ΔV| → slew_limit
```

В номинальном прогоне искажение составляет `max|ΔV| = 96.7 мВ`, `rms = 27.2 мВ`, `LSB = 366 мкВ`;
случаев клиппинга нет. Идеальный и доставленный сигналы различаются, и оптимизатор использует
именно доставленный сигнал.

<details>
<summary><b>Синхронизация, джиттер и отчёт об искажениях</b></summary>

- `trigger_latency_s` — общий статический сдвиг.
- `jitter_s` — per-sample временной джиттер, `rng.uniform(-jitter, jitter, len(t))`, учитывается как
  `y(t − delay − trig − time_jitter(t))` с интерполяцией.
- `skew_per_channel_s` — статический поканальный сдвиг (прежнее имя `jitter_per_channel_s`).
- `distortion.json`: `max_command_jump_V`, `lsb_V`, `max_deviation_V`, `rms_deviation_V`,
  `channel_max_deviation_V`, `delay_s`, `trigger_latency_s`, `jitter_s`, `skew_per_channel_s`,
  `time_jitter_s`, `max_time_jitter_s`, `jitter_model`, `hold`, `filter`, `hardware_identifier`,
  `calibration_state`.

Режим `ideal=True` исключает квантование и фильтрацию (одно состояние на команду) и служит для
сопоставления решений, полученных без учёта и с учётом аппаратных ограничений.

</details>

<a id="metrics"></a>

## Метрики, приоритеты и приёмка

Приоритеты соответствуют спецификации (§8): нарушение условия более высокого уровня отменяет
результат независимо от остальных показателей.

```
safety  →  confinement  →  hardware realizability  →  excitation / accuracy  →  speed
```

Скалярная функция потерь (единая для всех сравнений; компоненты записываются отдельно):

```
L = log1p(n/n₀) + log1p((pos/tol)²) + log1p((well/tol)²) + 1e-4·‖θ‖²  + штрафы аппаратного тракта
robust: 0.35·mean + 0.65·max по обучающим сценариям
```

| Метрика | Источник | Содержание |
|---|---|---|
| `n_total`, `n_modes` | все артефакты | нагрев в фононах, бюджет ≤ 0.5 |
| `position_error_m` | все артефакты | ошибка позиции, допуск 5 мкм |
| `minimum_frequency_hz` | все артефакты | удержание, порог 400 кГц |
| `max_deviation_V`, `rms_deviation_V` | `distortion.json` | отклонение доставленного сигнала от команды |
| `mean_n`, `worst_n`, `cvar_20pct_n`, `pass_fraction` | `robust.json` | статистика хвоста распределения |
| `refinement_delta_n`, `integrator_delta_n` | `validation.json` | устойчивость результата к выбору модели |
| `optimization_improved_over_baseline` | `status.json` | превышение результата над нулевым значением `θ` |

Коды отказа (`Rejected`): `no_minimum`, `weak_confinement`, `infeasible_voltages`,
`hardware_saturation`, `slew_limit`, `route_corridor`, `rom_trust_region`, `numerical_failure`,
`integrity_failure`, `blocked_by_missing_data`, `high_dim`, `invalid_input`,
`optimizer_budget_exhaustion`.

<a id="results-nominal"></a>

## Результаты: номинальный прогон

![Траектория](assets/fig-trajectory.png)

*Рис. 5.* Реализованная траектория, скорость и минимальная секулярная частота для номинального
прогона.

Параметры: 30 мкм в прямом и обратном направлении, 15 мкс на плечо, пауза 2 мкс. Полученные
значения: `n_total = 0.0071` фонона (1.4 % бюджета 0.5), ошибка позиции `1.38 нм` при допуске 5 мкм,
минимальная секулярная частота `688 кГц` при пороге 400 кГц.

![Частоты вдоль маршрута](assets/fig-frequencies.png)

*Рис. 6.* Секулярные частоты вдоль маршрута по 41 опорной точке (ROM-диагностика). Старшая мода
определяет запас по удержанию, две остальные соответствуют радиальным колебаниям.

<a id="results-validation"></a>

## Результаты: валидация (multi-fidelity)

![Валидация](assets/fig-validation.png)

*Рис. 7.* Значения `n_total` на четырёх уровнях точности и величины дрейфа при изменении модели.

Один и тот же прогон рассчитан четырьмя способами; результаты различаются незначительно.

| Сравнение | `Δn` (модуль) |
|---|---|
| дешёвая ROM → точная модель (321 узел) | `1.7e-4` |
| уточнение сетки до 1281 узла | `1.4e-4` |
| переход от грубой модели к тонкой | `2.2e-3` |
| смена интегратора (`exact` → `DOP853`) | `3.0e-7` |

Все величины дрейфа меньше допуска `0.01` фонона, `convergence_passed = true`. Дешёвая модель
пригодна для поиска, а принимаемое решение проверяется на точной.

<a id="results-robust"></a>

## Результаты: робастность

![Робастность](assets/fig-robust.png)

*Рис. 8.* Распределение `n_total` по сценариям возмущений и запас до бюджета 0.5 фонона.

Рассмотрено 6 сценариев с объявленными (не измеренными) возмущениями: `gain`, `offset`, задержка,
фильтр, stray-поле, масштаб частоты. Все сценарии приняты (`pass_fraction = 1.0`); распределение
приведено полностью.

| Статистика | `n_total` | Доля бюджета 0.5 |
|---|---|---|
| `mean` | `0.0269` | 5 % |
| `CVaR 20 %` | `0.0744` | 15 % |
| `worst` | `0.0764` | 15 % |

Среднее значение в 6 раз меньше наихудшего. По этой причине агрегация выполняется как
`0.35·mean + 0.65·max`, а не как среднее арифметическое.

<a id="results-ga"></a>

## Островной генетический алгоритм

![Island GA](assets/fig-ga.png)

*Рис. 9.* Динамика минимального фитнеса по островам, разнообразие популяции, распределение
кандидатов зала славы и лучший геном.

Пять островов функционируют одновременно и сохраняют различие стратегий поиска.

| Остров | Базовая траектория | Стратегия | Размерность генома |
|---|---|---|---|
| **S** | `smooth` (10s³ − 15s⁴ + 6s⁵) | гладкая адиабатика, малая мутация | 8 |
| **A** | `sta` (слагаемое `s''/(Tω)²`) | предпочтителен при малых `T`, чувствителен к фильтру | 8 |
| **V** | `sta` | переменная частота; обеспечивает допустимость на 100 мкм при ±12 В | 10 |
| **R** | `sta` | отбор по `worst`, а не по `mean` | 8 |
| **H** | `sta` | штраф за искажение фильтром (`rms`, `max_deviation`) | 8 |

Миграция выполняется по кольцу `S→A→V→R→H→S` каждые 5 поколений. Зал славы использует фильтр
разнообразия (нормализованное расстояние по `θ` не менее 0.12). Чекпоинт записывается после каждой
генерации.

Демонстрационный прогон `output/ga_demo` (8 поколений, 106 с): минимальный фитнес `0.00236`,
`n_total = 0.00125` фонона, ошибка позиции `1.25 нм`. Лучшее решение описывается 10 параметрами
вместо 200 отсчётов напряжения.

```bash
# островной алгоритм с дашбордом
python -m shuttle_sim.ga --config configs/shuttle_ga_island.json --out output/ga_run \
       --dashboard --port 8000 --generations 40
```

<a id="dashboard"></a>

## Дашборд в реальном времени

![Дашборд](assets/fig-dashboard.png)

*Рис. 10.* Панели дашборда, построенные по данным `output/ga_demo`.

Запись состояния не блокирует эволюцию: файл `state.json` обновляется атомарно
(`state.json.tmp → state.json`) после каждой генерации, сервер работает в отдельном процессе и
только читает его.

```bash
# отдельный процесс: статические файлы и /api/state по каталогу прогона
python -m shuttle_sim.dashboard.server --out output/ga_demo --port 8000
```

В среде предварительного просмотра адрес имеет вид `https://8000-<sandbox>.e2b.app`. Панели:
фитнес по поколениям и островам, разнообразие, гистограмма `n_total`, зал славы, Парето-фронт
`n`–`worst`, 15 каналов `V(t)`, траектория `r(t)`, вид ловушки сверху с анимацией кандидатов, граф
миграций, состояние CUDA (при отсутствии `cupy` используется numpy-реализация без изменения
результатов).

<a id="gds"></a>

## Геометрия из GDS

![GDS](assets/fig-gds.png)

*Рис. 11.* Топология примера GDS и соответствие слоёв каналам.

Принадлежность канала определяется явной парой `(layer, datatype)` в `electrode_map`; порядок
полигонов не используется. RF-электроды задаются отдельно; пересечение RF- и DC-каналов
проверяется при загрузке.

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
<summary><b>О единицах измерения в файле примера</b></summary>

`configs/generated_fivewire.gds` — демонстрационный файл для проверки соответствия слоёв, а не
описание реального устройства: запись UNITS в нём содержит условное значение, поэтому координаты
на рисунке приведены в единицах GDS. Для реальной геометрии достаточно указать корректную запись
UNITS: функция `load_trap` переводит единицы файла в метры по коэффициенту `lib_unit / L`.

</details>

<a id="configs"></a>

## Конфигурации

Всего в каталоге `configs/` 13 конфигураций.

| Конфигурация | Ловушка | Протокол | Назначение |
|---|---|---|---|
| `shuttle_reference.json` | five-wire, RF 20 МГц | round-trip 30 мкм, 15 мкс | номинальный режим, базовая приёмка |
| `shuttle_oneway.json` | five-wire | one-way 30 мкм | тот же конвейер с другим протоколом |
| `shuttle_repeated.json` | five-wire | repeated ×2, 20 мкм | сохранение состояния фильтра на разворотах |
| `shuttle_gds.json` | GDS, слои 10–26 | round-trip 30 мкм | соответствие слоёв каналам без скрытого порядка |
| `shuttle_variable_freq.json` | five-wire | round-trip, `coeff [0.03, −0.015]` | переменная частота удержания |
| `shuttle_90um_30MHz.json` | five-wire, h = 90 мкм, RF 30 МГц, ±12 В | one-way 30 мкм | строгая аппаратная постановка |
| `shuttle_90um_60MHz.json` | five-wire, h = 90 мкм, RF 60 МГц, ±12 В | one-way 30 мкм | та же геометрия при удвоенной RF |
| `shuttle_90um_100um_strict_30MHz.json` | h = 90 мкм | round-trip 100 мкм, 10 мкс | предельный случай по напряжениям |
| `shuttle_100um_10us.json`, `shuttle_100um_50us.json` | ±20 В | one-way 100 мкм, 10 и 50 мкс | влияние длительности на дальний перенос |
| `shuttle_100um_strict.json` | ±20 В, GA | round-trip 100 мкм | конфигурация генетического алгоритма |
| `shuttle_ga_island.json` | five-wire | round-trip 30 мкм | пять островов, миграция, зал славы |
| `shuttle_ga_isolated.json` | h = 90 мкм, переменная `ω` | round-trip 30 мкм | остров с геномом размерности 10 |

<details>
<summary><b>Поля конфигурации</b></summary>

| Параметр | Поле | Единица | Содержание |
|---|---|---|---|
| Масса и заряд | `trap.mass_amu`, `trap.charge_e` | а.е.м., e | ⁴⁰Ca⁺: 40 и 1 |
| Масштаб длины | `trap.length_scale_m` | м | внутренний масштаб `L`, как правило `1e-6` |
| RF | `trap.rf_frequency_hz`, `trap.rf_peak_V` | Гц, В | одна частота, общая фаза, псевдопотенциал |
| Геометрия | `trap.geometry` либо `trap.path` с `electrode_map` | мкм, м | аналитическое описание или GDS |
| DC-каналы | `trap.dc_indices`, `lower_V`, `upper_V`, `initial_dc_V` | В | управляемые электроды и их границы |
| Начальное приближение | `trap.minimum_seed_m` | м | затравка поиска минимума |
| Аппаратный тракт | `hardware.*` | Гц, В, с | дискретизация, фильтр, усиление, задержка, slew |
| Протокол | `task.distance_m`, `duration_s`, `hold_s`, `direction`, `protocol`, `cycles` | м, с, — | параметры переноса |
| Критерии качества | `task.min_frequency_hz`, `max_n_total`, `position_tolerance_m` | Гц, фонон, м | условия приёмки |
| Неопределённости | `uncertainty.*_sigma` | В, с, — | объявленные распределения |

</details>

<a id="assets"></a>

## Иллюстрации в документации

Все иллюстрации размещены в каталоге `assets/` и формируются по артефактам прогонов. Ручной ввод
числовых значений в изображения не применяется, поэтому иллюстрации не могут разойтись с данными.

```bash
python tools/make_docs_assets.py                    # все PNG и GIF
python tools/make_docs_assets.py --only wave traj   # отдельные группы
```

| Тип иллюстрации | Способ получения |
|---|---|
| График по результатам расчёта | функция `fig_*(...)` в `tools/make_docs_assets.py` читает `output/<run>/*.json` и `*.csv`, сохраняет PNG в `assets/` |
| Схема или чертёж | `matplotlib.patches` (примеры: `fig_architecture`, `fig_gds`) либо SVG из векторного редактора; GitHub отображает SVG |
| Анимация | `matplotlib.animation` с `PillowWriter` → `assets/anim-*.gif`; для записи дашборда — видеозахват и `ffmpeg -i in.mp4 -vf "fps=12,scale=960:-1" out.gif` |
| Снимок дашборда | `python -m shuttle_sim.dashboard.server --out output/ga_demo --port 8000`, затем снимок окна → `assets/dashboard-live.png` |
| Карточки с числовыми значениями | `fig_scorecard` читает `status.json` и `comparison.json` всех прогонов |

Добавление новой иллюстрации требует около 15 строк: функция читает артефакт и сохраняет файл.

```python
# tools/make_docs_assets.py
def fig_duration_scan():
    rows = json.loads((RUN / "duration_scan.json").read_text())
    t = [r["duration_s"] * 1e6 for r in rows]
    n = [r["metrics"]["n_total"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, n, "o-", color=ACCENT)
    _style(ax, "Длительность плеча и нагрев", "T, мкс", "n_total, фононы")
    _save(fig, "fig-duration-scan.png")

JOBS["duration"] = fig_duration_scan        # регистрация в словаре JOBS
# запуск: python tools/make_docs_assets.py --only duration
```

Вставка в документ выполняется стандартными средствами Markdown:

```markdown
![Команда и доставленный сигнал](assets/fig-waveforms.png)

*Рис. 4.* Изменение команды ΔV(t) и отклонение доставленного сигнала по каналам.
```

Требования к иллюстрациям:

1. Имя файла отражает содержание: `fig-<тема>.png`, `anim-<тема>.gif`.
2. Ширина графиков 1200–1300 px, разрешение 145–160 dpi; размер GIF не более 1 МБ (частота
   12–15 кадров/с, ширина 960–1200 px).
3. Каждая иллюстрация сопровождается подписью с указанием содержания, источника данных и единиц.
4. Числовые значения не вводятся вручную: источником служат файлы в `output/`, повторный запуск
   `tools/make_docs_assets.py` обновляет изображения.
5. Схемы помечаются как схемы и не интерпретируются как результат измерения.

<a id="honesty"></a>

## Область применимости и ограничения

Приведённые результаты получены расчётом, а не измерением на установке. Ограничения зафиксированы
в артефактах:

- `calibration_state: unmeasured_simulation_assumptions`, `hardware_identifier:
  declared-simulation-chain-v1`, `experiment_ready: false` — до подключения реальной установки;
- модель поля — трёхмерная локальная гармоническая; полное RF-микродвижение и негауссова квантовая
  эволюция не моделируются (указано в `manifest.json` и `validation.json`);
- распределения неопределённостей в `robust` объявлены, а не измерены;
- демонстрационный набор данных калибровки синтетический: `calibration_resynthesis.json` содержит
  `improved: false` (0.0049 → 0.0236) — идентификация по трём точкам ухудшила модель, и этот
  результат сохранён без изменений;
- демонстрация closed-loop ограничена бюджетом: `evaluations 11`, `optimizer_success: false`,
  `termination: Maximum number of function evaluations…`; значение потерь снизилось с `0.0472`
  до `0.0158`;
- запрос размерности 1000 для closed-loop отклоняется: `closed_loop_param_check.json`,
  `rejected_as_expected: true`;
- при отсутствии внешних данных калибровка не подменяется расчётом: `calibration_blocked.json`,
  `blocked_by_missing_external_data`;
- сопоставление результатов генетического алгоритма и конвейера корректно только при совпадении
  `trap`, `hardware`, `exact_nodes` и бюджета вычислений.

<a id="faq"></a>

## Диагностика отказов

<details>
<summary><b><code>infeasible_voltages</code> на 100 мкм за 10 мкс при ±12 В</b></summary>

Такое поведение ожидаемо: `SLSQP` не находит `V ∈ [−12, 12]` для дальней точки при `ω = 400 кГц`.
Способы устранения в порядке возрастания затрат: увеличить `duration_s`; расширить границы до ±20 В,
согласовав значения с `hardware.dac_min/max`; включить переменную частоту (`coeff [0.03, −0.015]`),
снижая частоту в середине маршрута. При заданном `task.durations_scan_s` прогон записывает
`duration_scan.json`, по которому определяется предельная длительность.

</details>

<details>
<summary><b><code>slew_limit</code> и <code>hardware_saturation</code></b></summary>

Необходимо уменьшить `max_jump_V`, увеличить `sample_rate_hz` либо приблизить параметры `θ`
(деформацию времени) к нулю. Значение `max_command_jump_V` в `distortion.json` указывает канал и
участок, на котором возникает ограничение.

</details>

<details>
<summary><b><code>weak_confinement</code> и <code>rom_trust_region</code></b></summary>

Условие `f_min` не выполнено либо минимум сместился дальше `rom_max_shift_m` (20 мкм) от
номинального положения. Требуется увеличить `reg` и уменьшить `θ₂…θ₇`. Файл
`delivered_local_diagnostics.json` содержит поля `mode_rotation_risk` и `relative_min_gap`,
указывающие участок с нарушением структуры мод.

</details>

<details>
<summary><b><code>numerical_failure</code> и <code>no_minimum</code></b></summary>

Начальное приближение `minimum_seed_m` расположено слишком близко к поверхности либо `least_squares`
не сошёлся. Следует увеличить `z` затравки (например, с 70 до 200 мкм) и проверить, что
`initial_dc_V` лежит внутри `[lo, hi]`.

</details>

<details>
<summary><b>GDS: сообщение «mapped electrode has no polygons»</b></summary>

Необходимо сопоставить `layer` и `datatype` с содержимым файла. Перечень полигонов выводит
`python tools/gds_reader.py configs/generated_fivewire.gds --scale 1e-6`; параметр `--scale`
требуется, если запись UNITS содержит условное значение. Полигоны обходятся против часовой
стрелки — при отрицательной площади направление меняется автоматически.

</details>

<details>
<summary><b>LAMMPS: сообщение «identical uids» (модуль <code>sion/sion.py</code>)</b></summary>

Требуется перезапуск ядра (IPython): устаревший стек LAMMPS сохраняет состояние между вызовами.
Для `shuttle_sim` LAMMPS не используется.

</details>

<a id="structure"></a>

## Структура проекта

```
Sion_ionfork/
├── README.md                     # настоящий документ
├── assets/                       # иллюстрации для документации (формируются по output/)
├── tools/
│   ├── make_docs_assets.py       # построение рисунков и анимации по артефактам прогонов
│   ├── gds_reader.py             # чтение GDSII без внешних зависимостей
│   └── preview_readme.py         # локальный рендер документа для проверки вёрстки
├── shuttle_sim/                  # вычислительный конвейер (СИ, ограничения, чекпоинты)
│   ├── __main__.py               # CLI: python -m shuttle_sim --config … --out …
│   ├── pipeline.py               # Runner: prepare/grid/commands/rom/evaluate/optimize/robust/run
│   ├── field.py                  # SionTrap, load_trap, Local, Rejected
│   ├── control.py                # четыре семейства параметризаций
│   ├── voltage.py hardware.py    # инверсия напряжений и аппаратная модель
│   ├── dynamics.py               # трёхмерный гауссов пакет, exact и dop853
│   ├── safety.py observables.py  # сквозные проверки и наблюдаемые
│   ├── calibration.py closed_loop.py transport.py
│   ├── ga/                       # островной алгоритм: engine, island, migration, fitness, cuda
│   └── dashboard/                # сервер и статические файлы дашборда
├── sion/                         # исходный пакет: electrode, pylion, sion.py (LAMMPS)
├── configs/                      # 13 конфигураций и пример GDS
├── docs/                         # architecture.md, ga_architecture.md
├── examples/                     # ноутбуки (исходный SION, GDS, моды)
├── output/                       # артефакты прогонов, по каталогу на прогон
└── tests/                        # test_regression.py: safety, калибровка, closed-loop
```

Проверка: `pytest tests/test_regression.py -q` (ускоренная конфигурация: 15 узлов, 4 оценки).

<a id="cite"></a>

## Цитирование

При публикации результатов, полученных с использованием `SION`, следует цитировать:

> *Surface trap with adjustable ion couplings for scalable and parallel gates*,
> Phys. Rev. A **109**, 022605 (2024) — https://journals.aps.org/pra/abstract/10.1103/PhysRevA.109.022605

Лицензия: **GPL-3.0-or-later**.

---

<sub>Результаты получены расчётом. Состояние калибровки `unmeasured_simulation_assumptions`,
идентификатор аппаратной модели `declared-simulation-chain-v1`, признак `experiment_ready: false`
сохраняются до подключения реальной установки.</sub>
