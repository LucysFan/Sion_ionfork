# Island GA — полная архитектура для shuttling + GDS + experiment

> Цель: генерить **много разных маршрутов иона** (не один `Powell` локальный), из них собрать Парето-фронт для GDS, добить лучший на эксперименте `closed-loop 8 dim`. Дашборд — в реальном времени, без замедления GA, плюс `CUDA` если есть.

## 1. Чем и зачем управлять — геном

**Базовый геном 8 float — это и есть низкоразмерное управление, которое потом поедет на установку:**

* `g0,g1` — `time-warp` на `env=16s²(1-s)²` (полином `P[0,0,16,-32,16]`). Меняет закон времени `s(t)` не ломая `v=a=0` на границах. Физический смысл: задержаться в середине (где `Φ` пологий) или ускориться.
* `g2..g7` — `3 огибающие × 2 SVD-моды геометрии` `ΔV = (e0,e1,e2)·Θ·R_svd`. `e0=env`, `e1=env·(2s-1)`, `e2=env·(6s²-6s+1)` — все нуль на краях, `R_svd` — топ-2 сингулярных вектора `route_V-mean` (null-space лапласиана). То есть ген не правит 200 вольтовых отсчётов, а правит 6 чисел в подпространстве которое реально двигает минимум.

**Расширенный геном 10 для островов `V`:** `g8,g9` — `variable frequency` `ω(s)=ω0·(1+a·env+b·env·(2s-1))`, клип `±15%`. Позволяет провалить частоту в середине, чтобы уложиться в `±12В` на `100мкм`.

Границы жёсткие:
```
g0,g1 ∈ [-0.10,0.10]
g2..g7 ∈ [-0.4,0.4]
g8,g9 ∈ [-0.15,0.15] (если остров variable)
```
Это и есть `search_bounds` — не `1000` семплов.

**Что можно дотюнить:**

* **В симуляции (дешево, точно):** весь геном `10`, `route_nodes`, `regularisation`, `frequency_mode`. Можно даже менять `SVD-моды` (пересчитать `R` на новом `sroute`).
* **На эксперименте (дорого, шумно):** тот же `8-10` генов через `ClosedLoopRefiner` + `SafetyLayer`. Плюс калибровка `gain/offset/delay/filter` — отдельный слой `calibration.py`, не чёрный ящик. То есть GA ищет в симуляторе, эксперимент только `fine-tune` 8 чисел.

## 2. Конечные цели — не один лосс

Иерархия §8 спеки:
1. **Safety** — `Rejected` → `fitness=1e6` (no_minimum, infeasible, saturation, weak_confinement, slew)
2. **Confinement** — `fmin < min_frequency_hz` → `+log1p`
3. **Hardware** — `max_deviation_V`, `rms`, `clipping`, `slew`
4. **Возбуждение** — `log1p(n/n0)` где `n0=max_n_total` (фононы) + `log1p((pos/tol)²)` + `log1p((well/tol)²)`
5. **Робаст** — `0.35mean+0.65max` по `3 train` сценариям (`gain/offset/delay/filter/stray/freq_scale`) + `CVaR20` для хвоста
6. **Эстетика** — `1e-4·||θ||²`

Итоговый скаляр `fitness(θ) = scalar(primary)`, но в дашборд пишем все компоненты отдельно: `n_total`, `n_modes`, `pos_err`, `well_err`, `fmin`, `slew`, `distortion`, `mean_n`, `worst_n`, `cvar` — чтобы видеть **вариации**, а не одну линию.

## 3. Острова — зачем 5 разных

Каждый остров — отдельная популяция `20` особей, свой оператор и свой уклон. Диверсити сохраняется не насильно, а нишами:

* **Island-S (Smooth):** инициализация около `smooth` траектории, мутация `σ0=0.04` малая, `cx 0.7` `tournament k=3`. Штраф за `n_total` ×1.5 — ищет адиабатику.
* **Island-A (STA):** старт около `sta`, мутация `σ=0.06`, `k=5` — любит короткие `T`.
* **Island-V (Variable):** геном 10, `σ 0.08`, `cx 0.9` — единственный кто может менять `ω(s)`, спасает `100мкм ±12В`.
* **Island-R (Robust):** фитнес считается на `3 train` сценариях сразу, популяция `25` — отбирает не `mean` а `worst`.
* **Island-H (Hardware):** в `fitness` добавлен `0.5·log1p(rms_dev/0.1)` — штрафует за искажение фильтром, любит большие `lowpass`.

Размер `5×20=100` особей — каждую генерацию `100` `Runner.evaluate(cheap ROM)` ~ `0.4с×100=40с` но с `multiprocessing 4` → ~10с. Дешёвая `ROM` без `trap.local exact`, точные `321 node` только для `hall of fame`.

## 4. Миграция и штрафы

Топология **кольцо** `S→A→V→R→H→S`, каждые `5` поколений мигрирует `2` особи: `лучшая` + `самая дальняя по θ`-расстоянию (поддерживает диверсити). Замещение — убиваем `2 худших` острова-приёмника. Раз в `15` поколений — глобальный `shuffle` 1 случайной особи во все острова (избегаем дрейфа).

Штрафы реализованы как **жёсткие + мягкие:**
* жёсткие `Rejected` → `fitness 1e6` и `status=code` логируется (не трейсбек)
* мягкие: `if fmin<0.4МГц: fitness+=2·log1p((0.4-fmin)/0.1)`, `if slew>0.9·max_jump: +1`, `if |θ|>bounds: +inf` (клип)

Так острова не схлопываются в одну нишу — `S` остаётся гладким, `V` — частотным.

## 5. CUDA — если есть, не мешает если нет

`shuttle_sim/ga/cuda.py`:
```python
try: import cupy as cp; HAS_CUPY=True
except: cp=np; HAS_CUPY=False
try: import numba; HAS_NUMBA=True
except: HAS_NUMBA=False
```
* `dynamics.propagate` матричные `S·C·Sᵀ` 6×6 — если `HAS_CUPY` и `N>500` (длинный `repeated`), то `C` как `cp.ndarray` и `cupy.matmul` (~1.8× быстрее на `V100`).
* `Hardware.deliver` RC-каскад `expm` — остаётся `scipy.linalg.expm` (CPU), CUDA не трогает `trap.local` (`least_squares` — CPU).
* Параллельность GA — `multiprocessing.Pool(4)` по островам, каждый воркер может внутри использовать `cupy` если `has_cupy`. Если `HAS_CUPY==False` — прозрачный `numpy` fallback, обучение не замедляется.

Проверка `cuda.is_available()` пишет в `manifest.json`.

## 6. Дашборд реального времени — не тормозит GA

**Архитектура без замедления:**
* GA пишет **атомарно** `output/ga_run/state.json.tmp → state.json` после каждой генерации (~50мс) + `history/ gen_*.json`
* Отдельный процесс `dashboard/server.py` (Flask `0.0.0.0:8000` → preview `https://8000-xxx.e2b.app`) **только читает** файл, `GA` его не ждёт. Поллинг `500мс` или `SSE`.

**Что видно (не просто loss):**
* `loss` — `best/mean/worst` на остров и глобально, `worst` робаста отдельно
* `diversity` — `σ(g0..g7)` и pairwise `θ`-расстояние
* `n_total` гистограмма на популяцию (лог шкала), `pos/well err` scatter
* `hall of fame` — 10 лучших `θ`, их `V(t)` 15 каналов, `r(t)` траектория, `f(t)` профили
* `hardware` — `max_deviation_V` vs поколение, `slew`
* `migration` — граф кольца, вспышки при `migration`
* `Pareto` — `n vs robust worst` фронт

Фронт — `Chart.js` + `Plotly` heatmaps, `index.html` статика, `fetch /api/state` каждые `500мс`. GA продолжает мутировать.

Запуск:
```bash
python -m shuttle_sim.ga --config configs/shuttle_ga_island.json --out output/ga_100um --dashboard --port 8000
# → http://localhost:8000  (в E2B → Live Preview)
```
Если `--dashboard` не указан — GA всё равно пишет `state.json`, дашборд можно поднять потом.

Так получаем: **острова держат диверсити, миграции не дают застрять, штрафы делают физику, CUDA ускоряет если есть, дашборд показывает всё живое без `GIL` блокировки.**
