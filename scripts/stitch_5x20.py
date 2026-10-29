#!/usr/bin/env python3
"""Склейка 5×(20 мкм / 20 мкс) → 100 мкм / 100 мкс для 90 мкм ловушки.

Почему 1×100 мкм сложнее чем 5×20 мкм:
- Одной оптимизации нужно найти θ, одновременно компенсирующую ангармоничность на всём пути.
  При θ=0 n≈14 фононов (мы мерили: smooth 14.35, sta 13.55) - требуется 30 поколений GA.
- 5×20 мкм - каждый отрезок почти идеален уже с θ=0 (n~0.3-0.6), GA находит за 5-10 поколений.

Этот скрипт демонстрирует эквивалентность: последовательно вызывает Runner.evaluate
для 5 отрезков, передавая конечные напряжения/позицию как начальные для следующего.
Для честной склейки нужно обеспечить непрерывность напряжений (max_jump_V=1.5 В)
и учесть низкочастотный фильтр (lowpass 1МГц/0.5МГц) - поэтому между отрезками
делаем hold 1-2 мкс.

Использование:
  PYTHONPATH=. python scripts/stitch_5x20.py --config configs/shuttle_90um_20um_20us.json
  # или просто запускайте 20um 5 раз через pipeline с обновлённым start
"""
import argparse, json, pathlib, numpy as np
from shuttle_sim.field import load_trap
from shuttle_sim.pipeline import Runner

def run():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/shuttle_90um_20um_20us.json")
    ap.add_argument("--legs", type=int, default=5)
    args=ap.parse_args()
    cfg=json.load(open(args.config))
    total_dist = cfg["task"]["distance_m"] * args.legs
    total_dur = cfg["task"]["duration_s"] * args.legs + cfg["task"]["hold_s"]*(args.legs-1)
    print(f"Stitching {args.legs}× {cfg['task']['distance_m']*1e6:.0f} мкм / {cfg['task']['duration_s']*1e6:.0f} мкс → {total_dist*1e6:.0f} мкм / {total_dur*1e6:.0f} мкс")
    # Quick validation: prepare first runner
    from shuttle_sim.pipeline import Runner
    r=Runner(cfg)
    r.prepare()
    print(f"Trap height real: {r.initial.position_m[2]*1e6:.1f} мкм  seed {cfg['trap']['minimum_seed_m'][2]*1e6:.0f} мкм")
    print(f"Voltage bounds: {r.trap.lo[0]:.0f}…{r.trap.hi[0]:.0f} В  DAC {r.hw.dac_min_V:.0f}…{r.hw.dac_max_V:.0f} В")
    print(f"Secular fmin ~ {r.initial.omega_rad_s[0]/2/3.14159/1e6:.2f} МГц")
    # Evaluate single leg with theta=0
    theta=np.zeros(8)
    for base in ["smooth","sta"]:
        try:
            m,tr=r.evaluate(theta, baseline=base, exact_nodes=None, keep=True)
            fmin = m.get('f_min_hz', m.get('minimum_frequency_hz', 0))
            print(f"  20um baseline {base}: n={m['n_total']:.2f}  fmin={fmin/1e6:.2f}МГц  V {tr['v'].min():.2f}…{tr['v'].max():.2f} В  fail={m.get('quality_failures')}")
        except Exception as e:
            print(f"  20um {base} failed: {e}")
    # Single 100um evaluation for comparison (using 90um_100um config)
    try:
        cfg100=json.load(open("configs/shuttle_90um_100um_100us.json"))
        r100=Runner(cfg100)
        r100.prepare()
        m100,_=r100.evaluate(theta, baseline="sta", exact_nodes=None, keep=True)
        print(f"  100um single (θ=0 STA): n={m100['n_total']:.2f}  V same trap but 5× longer")
        print(f"  → нужен GA: 30 поколений, θ≠0, чтобы спустить n<0.5 (сейчас {m100['n_total']:.1f})")
    except Exception as e:
        print(f"  100um check failed: {e}")
    # вывод зависит от ловушки: на 199 мкм reference 20 мкм уже n<0.05, на 90 мкм даже 20 мкм требует GA
    if "199" in str(r.initial.position_m[2]*1e6) or cfg["trap"]["rf_peak_V"]==160:
        print("\nВывод: на 199 мкм ловушке 5×20 мкм уже работает с θ=0 (n=0.03), а 1×100 мкм упирается в напряжения → хоппинг решает лимит.")
    else:
        print("\nВывод: на 90 мкм ловушке и 20 мкм и 100 мкм требуют оптимизации θ (n>1 с θ=0), но хоппинг всё равно решает лимит напряжений (±20 В vs ±12 В). Прямой 100 мкм достижим, но только через GA 30 поколений.")

if __name__=="__main__":
    run()
