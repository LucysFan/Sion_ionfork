# Аудит соответствия спецификации v2 — поэтапная сверка

**Объект:** репозиторий `Sion_ionfork`, ветка `arena/c222e84f-sion-ionfork`, коммит `f1b832d`.
**Эталон:** `planar_trap_shuttling_agent_specification_v2.pdf` (23 стр., уровни приёмки
*Simulation-complete* и *Experiment-integrable*, критерии §1–§20 сводной таблицы приёмки).
**Дата аудита:** 2026-10-22. **Метод:** полный обход дерева файлов (296 файлов), чтение всех
модулей `shuttle_sim/` (12 модулей, ~5 100 строк), `shuttle_sim/ga/` (6 модулей),
`shuttle_sim/dashboard/`, ядра `sion/` (backend), `tools/`, `tests/`, 13 конфигов `configs/*.json`,
всех сохранённых артефактов `output/*/`; запуск тестов и свежий end-to-end прогон.

## Итоговый вердикт

| Уровень приёмки (спец. §15) | Вердикт |
|---|---|
| **Simulation-complete** | ⚠️ **Условно выполнен** — цепочка работает end-to-end на текущем коде (свежий прогон: `simulation_acceptance_passed=true`), но сохранённые в репозитории артефакты `output/*` получены на **чужом/старом коммите**, а ряд пунктов закрыт частично (см. реестр расхождений). |
| **Experiment-integrable** | ❌ **Не выполнен** — реального experiment-facing интерфейса нет: `execution_authorized` жёстко `False`, калибровка/closed-loop демонстрируются только на синтетическом perturbed hardware, коды отказа `experiment_communication_failure` / `invalid_observable_extraction` отсутствуют. Сам код это честно декларирует (`experiment_ready: false`), поэтому это не «подмена», а незакрытый уровень. |
| Подмена задачи (routing/QCCD/compiler) | ✅ Не обнаружена — предмет строго control-level single-ion. |
| Критические заглушки в `shuttle_sim` | ⚠️ Одна формальная: `Calibration.make_synthetic_dataset()` → `raise NotImplementedError` (редирект на helper Runner'а). Остальные «пустые» места честно возвращают `blocked_by_missing_external_data`. `NotImplementedError` в `sion/electrode/*` — легаси upstream-backend, не shuttle-путь. |

---

## Этап 0. Инвентаризация (все файлы проекта)

| Группа | Состав | Роль по спецификации |
|---|---|---|
| `shuttle_sim/` | `field, control, voltage, hardware, dynamics, pipeline, robust(в pipeline), calibration, closed_loop, observables, safety, transport` + `__main__` | Основная система, 8 слоёв |
| `shuttle_sim/ga/` | `engine, island, genome, fitness, migration, cuda` + `__main__` | Дополнительный островной GA (не обязателен спецификацией, baseline-сравнение) |
| `shuttle_sim/dashboard/` | `server.py`, `static/index.html` | Мониторинг GA (презентационный слой) |
| `sion/` | `electrode/` (System, PolygonPixelElectrode, basis potentials), `pylion/` (LAMMPS), `sion.py` (3231 стр.) | Trap-specific backend / reference (§2: «Sion должен быть изучен… как backend») |
| `configs/` | 13 JSON: five-wire ×11, GDS ×1, GA ×2, variable-freq, repeated, one-way, 100 µm, strict 30/60 MHz | Machine-readable конфиги (§13) |
| `output/` | 5 pipeline-прогонов ×34 файла + 2 GA-прогона | Артефакты (§16) — **устарели, см. Р-1** |
| `tests/` | `test_regression.py` (11 тестов), 2 скрипта-симуляции, notebook | Тесты — только shuttle-регрессия |
| `examples/` | 7 notebooks (upstream Sion) | Презентационный слой; shuttle-pipeline из них не запускается |
| `tools/` | `gds_reader.py`, `make_docs_assets.py`, `preview_readme.py` | Утилиты GDS/графиков README |
| Корень | `README.md`, `docs/architecture.md`, `docs/ga_architecture.md`, `pyproject.toml`, `requirements-shuttle.txt` | Документация и сборка |

## Этап 1. Физическая постановка (§2, §13 входов)

✅ Ион Ca-40 (`mass_amu=40, charge_e=1`), геометрия five-wire или GDS, RF 20–60 МГц / 160 В,
электроды с метаданными (`dc_indices`, имена, bounds ±12 В), hardware-ограничения
(`sample_rate, dac_bits, lowpass, gain/offset, delay, trigger_latency, jitter, max_jump`),
start/target/duration в `TransportTask`, quality targets (`max_n_total, position_tolerance,
min_frequency`), uncertainty-набор в конфиге. Всё SI на границах, `finite()`-валидация.
⚠️ **Initial-state offsets** в uncertainty-ансамбль не входят (см. Р-7); `nbar` фиксирован 0.

## Этап 2. TrapAdapter и модель поля (§4, критерий 3)

✅ `SionTrap` работает с любым `electrode.System`: `basis()`, `rf()` (псевдопотенциал),
`local()` (минимум через `least_squares`, гессиан, `eigh`, secular frequencies, моды),
`invert()`; два бэкенда: `sion_five_wire` и `gds` (gdspy → fallback python-gdsii,
явный `electrode_map`, запрет дублей layer/datatype). Не зашит под одну геометрию.
✅ 3D локальная гармоническая модель — нижняя граница соблюдена; отсутствие micromotion /
не-гауссовой динамики явно задекларировано в `validation.json.limitation` и manifest.
⚠️ GDS-путь **не покрыт тестами** и в свежих прогонах аудита не перезапускался (только stored-run).

## Этап 3. Путь и transport task (§4, критерий 11)

✅ Три пути разведены и задокументированы (`path_conventions.json`): `user_route`,
`nominal_well_path` (voltage_spline → `trap.local`), `realized_path` (delivered → ROM → dynamics).
✅ `TransportTask`: one_way / round_trip / repeated×cycles, hold, corridor, durations-scan,
hold-scan; round-trip и repeated реально присутствуют в конфигах и stored-прогонах.

## Этап 4. Параметризации и частотные режимы (§5, критерий 12)

✅ Четыре семейства: `TrajectoryParameterization` (time-warp, env `16s²(1-s)²`),
`FrequencyProfile` fixed/variable, `VoltageKnotParameterization`, `CorrectiveBasis`
(2 SVD-моды × 3–5 огибающих, θ∈ℝ⁸/¹⁰/¹²); closed-loop строго 8-мерен, θ=1000 отклоняется.
⚠️ **Variable-frequency профиль не оптимизируется**: `frequency_coeffs` берутся из конфига
фиксированными, оптимизатор (и GA-остров «V») меняет только вольтажные гены. Спецификация
требует, чтобы в variable-режиме профиль частоты «сам становился частью оптимизации» — закрыто частично.
⚠️ Клиппинг профиля ±18% в `prepare()` против «≤15%» в документации `control.py` — рассогласование док/код.

## Этап 5. Инверсия напряжений (§5)

✅ `SionTrap.invert`: 3 условия баланса сил + 1 аксиальной кривизны, SLSQP, bounds,
регуляризация, min-change к `prev` (null-space/smoothness de facto), проверка `‖local−target‖<5 нм`,
отказ `infeasible_voltages`. Радиальные кривизны свободны — честно обосновано уравнением Лапласа.
⚠️ Явного штрафа гладкости по времени и multipole-разложения нет (есть ROM на сплайнах гессиана).

## Этап 6. Hardware forward model, постпроцессинг, timing (§6–7, критерии 4–5)

✅ DAC-квантование (LSB, коды), ZOH, точный RC-каскад `expm(M·dt)` на плотной сетке,
gain/offset per-channel, clipping→`hardware_saturation`, slew→`slew_limit`, delay/trigger/jitter,
distortion-report (max/rms/channel deviation, LSB, задержки), экспорт CSV + `waveform_metadata.json`
(channel order, sample period, units, hw-id, calibration-state, SHA-256; `verify_export()` проверен аудитом — checksum сходится).
✅ Distortion-порог исполняемый: `delivered_intent_violated` → ограниченный re-synthesis под delivered-моделью.
⚠️ Нет **цифрового фильтра** в тракте (только аналоговый RC) и нет явной **precompensation** —
только re-optimization при нарушении порога.
⚠️ Jitter: seed = `int(sum(commands)*1e6) % 2^31` — детерминирован, но слаб (коллизии для близких
waveform); per-channel skew не является независимым параметром конфига, а выводится из `jitter_s/10`
(диагностический `skew_draw`), основной skew — это `delay_s`.

## Этап 7. Динамика и независимая валидация (§8, критерий 7)

✅ Multi-fidelity: дешёвый linear-response ROM (поиск) → exact-node реконструкция `trap.local`
(reranking/финал) → DOP853 (`rtol 2e-10`) как независимый интегратор; ковариация 6×6,
проверки PSD и соотношения неопределённостей; есть `validate_full_field_centroid` (полное поле Sion).
⚠️ Независимость — **только по численному методу** (одна и та же 3D-гармоническая аппроксимация).
Это формально допустимо спецификацией, но full-field проверка опциональна (`--full-field`),
по умолчанию выключена и отсутствует во **всех** stored- и в свежем validation.json.
✅ Свежий прогон: `refinement_delta_n=1.4e-4`, `integrator_delta_n=1.4e-7` при допуске `1e-2`.

## Этап 8. Objective, метрики, агрегация, иерархия (§8, критерии 6, 10)

✅ Loss: `log1p(n/n₀)+log1p((pos/tol)²)+log1p((well/tol)²)+1e-4‖θ‖²` + hardware-штрафы (GA-fitness);
метрики: `n_total/n_modes`, coherent/covariance, residual energy (Дж), phase-space displacement,
`max_transport_error_m` по траектории (не только финал), confinement indicators.
✅ Агрегация явная: `weighted_worst (0.35/0.65)`, `worst_case`, `mean`, `cvar_20` — единая для sim и observables.
✅ Иерархия safety→confinement→hardware→excitation→speed задекларирована и отражена в safety-gating.
✅ Baseline-сравнение (§15 спец.): smooth vs STA vs hardware-unaware vs hardware-aware, честный
rerank на одинаковых exact-nodes; флаг `optimization_improved_over_baseline` считается из чисел.
⚠️ В **свежем** прогоне текущего кода: `selected="unaware"`, `optimization_improved_over_baseline=false` —
т.е. hardware-aware оптимизация на reference-конфиге не улучшила baseline. В README/stored это не отражено.

## Этап 9. Robust evaluation (§10, критерий 8)

✅ Сценарии: stray (3D), gain, offset, delay, filter (log), frequency_scale; seed фиксирован;
отчёт: mean / worst / CVaR-20% / pass_fraction. Свежий прогон: 6 сценариев, pass=1.0, worst_n=0.076.
⚠️ Нет initial-state offsets и явного «calibration drift / model mismatch по пути» (только freq-scale прокси).

## Этап 10. Calibration / identification (§9–10, критерий 9)

✅ Отдельный слой, не black-box: `identify_from_residuals` (per-channel gain/offset, только на
«settled» участках ZOH, диагностика идентифицируемости), пустые данные → `blocked_by_missing_external_data`,
константные данные → отказ; последовательность identify→update_model→resynthesis→safety→отдельный
calibrated-экспорт (3 CSV + SHA) реализована в pipeline.
❌/⚠️ Идентифицируются **только gain/offset**. Filter: `identify_filter` — отдельный single-pole fit,
в `update_model`/pipeline **не интегрирован** (полюса «retained», честно пишется в provenance).
Delay/skew и stray-field **не идентифицируются** (`updated_stray_V_m` всегда `[0,0,0]`),
observable-only датасет → blocked. Спецификация требует «хотя бы часть» параметров — формально
gain/offset достаточно, но заявленный в `docs/architecture.md` объём («gain/offset и фильтра» как workflow) завышен.
⚠️ `make_synthetic_dataset()` — `raise NotImplementedError` (единственная формальная заглушка слоя).

## Этап 11. Observables и closed-loop (§10–12, критерии 10, 12)

✅ Observables: sideband red/blue proxies, asymmetry `1/(2n+1)`, 2nd-order, recooling, displacement α;
цепочка raw→observable→loss есть в двух видах: прямой proxy и честный raw-контракт
(`synthesize_raw_sideband_measurement` → `infer_observables_from_raw`, без утечки ground truth — при
насыщении ratio возвращается flagged `n=10`, а не `true_n`).
✅ Closed-loop: 8-мерный базис, Powell с жёстким бюджетом (считая initial point), safety-gate
**до** измерения, история, checkpoint, `all_failed→best=None` + fallback. Покрыто тестами.
❌ Нет реального experiment-интерфейса: только инъектируемый `measure_fn`; демо — эмуляция на
perturbed hardware (честно помечена). Отсутствуют failure-коды `experiment_communication_failure`
и `invalid_observable_extraction`. `fallback_theta=[0]*8` в closed-loop возврате — константа,
не провалидированная в этом модуле (pipeline валидирует fallback отдельно).
➡️ Итог: **closed-loop-ready**, но не experiment-integrable.

## Этап 12. Safety и failure classification (§13–14, критерии 13–14)

✅ `SafetyLayer`: bounds, slew, clipping, filter-consistency с исполняемым порогом, confinement
(min frequency), fail-closed на NaN/Inf во всех входах; `fallback_state.json` с SHA последнего safe.
✅ Коды: `no_minimum, weak_confinement, infeasible_voltages, hardware_saturation, slew_limit,
route_corridor, rom_trust_region, numerical_failure, integrity_failure, delivered_intent_violated,
calibration_inconsistency, missing_external_data, optimizer_budget_exhaustion(_or_stagnation)`.
⚠️ `excessive_excitation` существует только как строка в `quality_failures`, не как Rejected-код;
`experiment_communication_failure` и `invalid_observable_extraction` отсутствуют;
`optimizer_stagnation` слит с budget-exhaustion в одну строку.

## Этап 13. Воспроизводимость и артефакты (§15–16, критерии 15–16)

✅ Свежий прогон текущего кода производит 44 артефакта: config, trap_snapshot, transport_task,
path_conventions, route_diagnostics, ideal/dac/delivered CSV, calibrated-* CSV, distortion(+resynthesis),
comparison, candidate_reranking, validation, robust, hold/duration scans, observables + raw-chain,
calibration demo/blocked/state/params/export, closed-loop emulated/param-check, safety, fallback,
waveform_metadata (SHA-256), manifest (версии python/numpy/scipy, platform, input/source SHA, git commit).
❌ **Р-1 (главное расхождение):** сохранённые `output/*` получены на коммите `c3551aa`, которого
**нет в текущей истории git**, пакет в manifest — `0.2.0` при `__version__='0.3.0'` в коде (и manifest
текущего кода по-прежнему хардкодит `'shuttle_sim 0.2.0'`). Stored-набор (34 файла) не содержит
calibrated-*, raw-*, duration_scan, calibration_state — т.е. **артефакты не доказывают текущий код**,
а цифры README (selected=sta и т.д.) свежим прогоном не воспроизводятся (selected=unaware).
⚠️ Seeds фиксированы; зависимости: в чистом окружении аудита тесты падали на отсутствии
scipy/h5py/tqdm и т.д. — `pyproject.toml` не включает shuttle-зависимости (`python-gdsii` и др. только
в `requirements-shuttle.txt`), установка «из коробки» не воспроизводится одной командой.

## Этап 14. Доказательная демонстрация (§17) и запреты (§18–20)

| Пункт §17 | Статус |
|---|---|
| 1. Nominal open-loop | ✅ comparison smooth/sta ideal+delivered |
| 2. Hardware-aware waveform | ✅ dac/delivered + metadata + checksum |
| 3. Независимая валидация | ⚠️ только другой интегратор; full-field выключен |
| 4. Robust-ансамбль | ✅ mean/worst/CVaR |
| 5. Closed-loop-ready параметризация | ✅ 8-мерная, эмуляция работает |
| 6. Round-trip сценарий | ✅ reference/gds/variable_freq + repeated + one-way |
| 7. Артефакты для повторного запуска | ❌ stored устарели (Р-1); свежий прогон — полон |

GA (`shuttle_sim/ga/`) спецификацией не требуется; как baseline-инструмент допустим, его stored-выводы
(`output/ga_*`) также относятся к старому коду. Notebooks — только upstream-демо Sion.

## Этап 15. Проверка запуском (фактические результаты аудита)

```
pip install scipy h5py jinja2 termcolor tqdm matplotlib scikit-optimize cvxopt python-gdsii shapely pytest
python -m pytest tests/test_regression.py -q      → 11 passed (75 с)
python -m shuttle_sim --config configs/shuttle_reference.json --out /tmp/audit_reference
                                                  → 185 с, simulation_acceptance_passed=true
  n_total=0.00492, position_error=1.43 нм, robust worst_n=0.0764 (pass 1.0),
  selected="unaware", optimization_improved_over_baseline=false,
  experiment_ready=false, execution_authorized=false
verify_export(/tmp/audit_reference)               → OK, SHA-256 сходится
```

## Реестр расхождений (приоритет)

| # | Серьёзность | Расхождение | Где |
|---|---|---|---|
| Р-1 | **Высокая** | Stored `output/*` и цифры README — со старого коммита `c3551aa` (нет в git-истории), manifest врёт про версию `0.2.0`; текущий код даёт другой selected/результат | `output/`, `pipeline.py:1045`, `README.md` |
| Р-2 | **Высокая** | Experiment-integrable не достигнут: нет реального интерфейса установки, только `measure_fn`/эмуляция | `closed_loop.py`, `__main__.py` |
| Р-3 | Средняя | Calibration идентифицирует только gain/offset; filter-fit не подключён, delay/stray не идентифицируются | `calibration.py` |
| Р-4 | Средняя | Variable-frequency профиль не входит в оптимизацию (фиксирован конфигом) | `pipeline.prepare/optimize` |
| Р-5 | Средняя | Независимая валидация — только смена интегратора при той же модели; full-field отключён по умолчанию | `pipeline.run`, `--full-field` |
| Р-6 | Средняя | Нет uncertainty по initial-state; нет digital-filter и precompensation в hardware-тракте | `pipeline.scenarios`, `hardware.py` |
| Р-7 | Низкая | `make_synthetic_dataset` → `NotImplementedError`; закрыт редиректом, но формально заглушка | `calibration.py:59` |
| Р-8 | Низкая | Пропущены failure-коды `experiment_communication_failure`, `invalid_observable_extraction`; `excessive_excitation`/`stagnation` — только строки | `closed_loop/observables/pipeline` |
| Р-9 | Низкая | Установка зависимостей не одной командой (`pyproject` ≠ `requirements-shuttle.txt`); GDS-путь без тестов; тесты только в одном файле | `pyproject.toml`, `tests/` |
| Р-10 | Низкая | Jitter-seed слабый; skew — производная jitter, не параметр; док «≤15%» vs код «±18%» для ω-профиля | `hardware.py`, `control.py`/`pipeline.py` |

**Что нужно для закрытия:** (1) перегенерировать все 5 pipeline-прогонов текущим кодом и обновить
README/manifest-версию; (2) подключить `identify_filter`+delay в `update_model` или сузить заявления
в docs; (3) включить `--full-field` хотя бы в reference-прогон; (4) добавить freq-coeffs в пространство
оптимизации variable-режима; (5) для experiment-integrable — реальный `measure_fn`-адаптер установки
с кодами Р-8. До этого честный статус системы: **simulation-complete (условно), experiment-ready — нет**,
что сам код в `status.json` не скрывает.

---

## Исправления по итогам аудита (2026-10-22, эта же ветка)

| # | Исправление |
|---|---|
| Р-1 | Manifest пишет реальную версию из `shuttle_sim.__version__`; все 5 прогонов `output/shuttle_*` перегенерированы текущим кодом (manifest `source_base_commit` = HEAD ветки), цифры README сверены со свежими `status/validation/robust.json`. |
| Р-2 | Добавлен `shuttle_sim/experiment.py`: `ExperimentInterface(transport=...)` / `BlockedExperiment`. Без транспорта — честный `experiment_communication_failure` (`blocked_by_missing_external_data`), не выдуманный score; `execution_authorized` по-прежнему `False` до реальной установки. |
| Р-3 | `Calibration.identify_from_residuals` интегрирует `identify_filter` (step-response → полюс), оценку delay (xcorr-лаг × dt) и `identify_stray_from_positions` (least_squares по измеренным минимумам); всё применяется через `update_model`. |
| Р-4 | `Runner.optimize_frequency_profile()`: в variable-режиме коэффициенты профиля частоты оптимизируются (Powell, bounds ±0.15, re-`prepare()` на каждой оценке), артефакт `frequency_optimization.json`. Клип профиля унифицирован ±15%. |
| Р-5 | Full-field валидация включена флагом `full_field_validation: true` в `shuttle_reference.json` и `shuttle_gds.json` → `validation.json.nonlinear_centroid`. |
| Р-6 | Uncertainty: добавлен `initial_nbar_sigma` (начальное тепловое заселение, `propagate(initial_nbar=...)`) во все конфиги; hardware: цифровой FIR (`digital_filter_b`), `Hardware.precompensate()` (приближённая инверсия RC-каскада). |
| Р-7 | `make_synthetic_dataset` реализован (deliver команд через true hardware), `NotImplementedError` удалён. |
| Р-8 | Реестр `FAILURE_CODES` в `field.py`, добавлены `experiment_communication_failure`, `invalid_observable_extraction` (валидация observables в closed-loop и experiment-интерфейсе), `excessive_excitation` продублирован в `failure_codes`, stagnation отделён от budget-exhaustion. |
| Р-9 | `pyproject.toml`: `[project.optional-dependencies] shuttle = [...]` (установка одной командой `pip install -e ".[shuttle]"`); тест загрузки GDS-ловушки, относительный GDS-путь резолвится и без CLI. |
| Р-10 | Jitter-seed — SHA-256 от waveform; skew — явный параметр `skew_per_channel_s`; док/код по ω-профилю сведены к ±15%. |

Честное ограничение, которое кодом не «исправляется», а декларируется: живой прогон на
реальной установке невозможен без доступа к ней — уровень *Experiment-integrable* закрыт
на уровне интерфейса (`ExperimentInterface` + safety-gated closed-loop + export с checksum),
`experiment_ready` остаётся `false` до подключения транспорта установки.
