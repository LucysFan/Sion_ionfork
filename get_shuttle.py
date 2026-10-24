#!/usr/bin/env python3
"""
get_shuttle.py — один файл-вход, куча хороших алгоритмов на выходе
========================================================
Даёшь JSON (тот самый общий файл) — получаешь оптимизированные шаттлы.

Что это за общий файл?
  configs/unified_live.json  (или любой из configs/*.json)
  Содержит: trap.kind/path/electrode_map (+ RF, height 90µm) +
            hardware (DAC/RC/jitter) +
            task (distance_m, duration_s, hold_s, protocol) +
            ga (острова, generations) +
            uncertainty (robust)

Пример вызова:
  python get_shuttle.py --config configs/unified_live.json --out output/my_shuttles
  python get_shuttle.py --config configs/shuttle_gds.json --out output/ga_big --generations 30 --dashboard
  python get_shuttle.py --config my_trap.json --out output/big_trap --mode pipeline   # быстрый 1 алгоритм без GA

Выход — папка output/my_shuttles/:
  - ga_hall.json          — топ-N разнообразных алгоритмов (theta, fitness, n_total, pos_err, baseline, island)
  - algorithms/algo_000.json ... algo_00K.json — каждый алгоритм отдельно (для импорта в закрытый контур)
  - best_waveform.csv / dac_waveform.csv / delivered_waveform.csv — лучший (для прошивки)
  - summary.md            — читаемая таблица
  - state.json + history/ — live для дашборда (если --dashboard)

Дашборд всегда: python -m shuttle_sim.dashboard.server --out output/my_shuttles --port 8000
  Открой http://localhost:8000 → вкладки Setup / Training / GA / Trap (анимация ямы+иона)
"""
import argparse, json, pathlib, sys, time, subprocess, shlex

def load_config(path: pathlib.Path):
    cfg = json.loads(path.read_text())
    # resolve GDS path robustly for cross-platform (Windows/Linux) and for absolute paths baked on other machine
    if cfg.get("trap", {}).get("kind") == "gds":
        raw = cfg["trap"]["path"]
        p = pathlib.Path(raw)
        # if path is absolute but does not exist (e.g. /home/user/... on Windows), fallback to basename search
        candidates = []
        if p.is_absolute():
            if p.exists():
                cfg["trap"]["path"] = str(p.resolve())
                return cfg
            # fallback: look for same filename near config and in repo
            candidates.append(path.resolve().parent / p.name)
            candidates.append(pathlib.Path.cwd() / p.name)
            candidates.append(pathlib.Path.cwd() / "configs" / p.name)
            candidates.append(pathlib.Path("configs") / p.name)
            candidates.append(pathlib.Path("configs/generated_fivewire.gds").resolve())
        else:
            candidates.append(path.resolve().parent / p)
            candidates.append(pathlib.Path.cwd() / p)
            candidates.append(pathlib.Path.cwd() / "configs" / p)
            # also try absolute resolve of candidate
            if len(candidates) and not candidates[0].exists():
                # try generated_fivewire.gds in configs as ultimate fallback for demo
                fb = pathlib.Path("configs/generated_fivewire.gds")
                if fb.exists():
                    candidates.append(fb.resolve())
        for cand in candidates:
            try:
                if cand.exists():
                    cfg["trap"]["path"] = str(cand.resolve())
                    break
            except: pass
        # if still not resolved, try to keep original but will fail later with clear error — also try python-gdsii friendly relative
        # ensure fallback file exists for demo
        if not pathlib.Path(cfg["trap"]["path"]).exists():
            # last resort: search for any .gds with same stem in configs
            try:
                stem = pathlib.Path(raw).stem
                for g in pathlib.Path("configs").glob("*.gds"):
                    if g.stem == stem:
                        cfg["trap"]["path"] = str(g.resolve())
                        break
                else:
                    for g in pathlib.Path("examples/gds").glob("*.GDS"):
                        if g.stem.lower() == stem.lower():
                            cfg["trap"]["path"] = str(g.resolve())
                            break
            except: pass
    return cfg

def main():
    ap = argparse.ArgumentParser(description="Один JSON → куча хороших шаттлов (GA hall)")
    ap.add_argument("--config", required=True, help="Путь к общему JSON (configs/unified_live.json или shuttle_gds.json)")
    ap.add_argument("--out", default="output/shuttles", help="Куда писать кучу алгоритмов")
    ap.add_argument("--generations", type=int, default=None, help="Переопределить generations (иначе из JSON)")
    ap.add_argument("--hall-size", type=int, default=20, help="Сколько алгоритмов в выходе (по умолчанию 20)")
    ap.add_argument("--mode", choices=["ga","pipeline","both"], default="ga", help="ga = много разнообразных (hall), pipeline = один быстрый робастный, both = и то и то")
    ap.add_argument("--dashboard", action="store_true", help="Поднять дашборд на --port (не блокирует оптимизацию)")
    ap.add_argument("--port", type=int, default=8000, help="Порт дашборда")
    ap.add_argument("--no-robust", action="store_true", help="Отключить robust остров (только для отладки)")
    args = ap.parse_args()

    cfg_path = pathlib.Path(args.config)
    if not cfg_path.exists():
        print(f"[get_shuttle] config не найден: {cfg_path}", file=sys.stderr)
        return 2
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(cfg_path)
    # also save a frozen copy as unified entry file
    (out / "input_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    # ensure dashboard server if requested
    dash_proc = None
    if args.dashboard:
        print(f"[get_shuttle] поднимаю дашборд на 0.0.0.0:{args.port} → http://localhost:{args.port}  (вкладки Setup/Training/GA/Trap)")
        dash_proc = subprocess.Popen([sys.executable, "-m", "shuttle_sim.dashboard.server", "--out", str(out), "--port", str(args.port), "--host", "0.0.0.0"])
        time.sleep(1.2)
        if dash_proc.poll() is not None:
            print("[get_shuttle] дашборд не стартанул — продолжаю без него", file=sys.stderr)
            dash_proc = None

    # --- pipeline быстрый (один) ---
    if args.mode in ("pipeline","both"):
        print(f"[get_shuttle] pipeline robust (1 θ) → {out}/pipeline_ ...")
        try:
            from shuttle_sim.pipeline import Runner
            runner = Runner(cfg)
            runner.prepare()
            # быстрый pipeline прогон (не весь run, а только optimize + validate, чтобы не ждать 30с)
            # используем runner.run для полной воспроизводимости, но с текущими nodes
            status = runner.run(out / "pipeline_run", full_field=False)
            print(f"[get_shuttle] pipeline done: {status}")
        except Exception as e:
            print(f"[get_shuttle] pipeline упал: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()

    # --- GA много алгоритмов (hall) ---
    if args.mode in ("ga","both"):
        print(f"[get_shuttle] GA много алгоритмов → {out}  (hall_size={args.hall_size})")
        # prepare config for GA
        ga_cfg = cfg.get("ga", {})
        if args.generations is not None:
            ga_cfg["generations"] = args.generations
        if "generations" not in ga_cfg:
            ga_cfg["generations"] = 12
        if "hall_size" not in ga_cfg:
            ga_cfg["hall_size"] = args.hall_size
        if args.no_robust:
            # убрать остров R
            ga_cfg["islands"] = [i for i in ga_cfg.get("islands",[]) if i.get("strategy")!="R"]

        # runner_cfg без ga
        runner_cfg = dict(cfg)
        runner_cfg.pop("ga", None)

        try:
            from shuttle_sim.pipeline import Runner
            from shuttle_sim.ga.engine import GAEngine
            runner = Runner(runner_cfg)
            print("[get_shuttle] готовлю ловушку (SVD, support voltages)...")
            runner.prepare()
            print(f"[get_shuttle] trap={runner.cfg['trap']['kind']}  distance={runner.task['distance_m']*1e6:.0f}µm  duration={runner.task['duration_s']*1e6:.0f}µs  route_nodes={len(runner.sroute)}")
            engine = GAEngine(runner, ga_cfg, out_dir=str(out))
            print(f"[get_shuttle] острова={len(engine.islands)}  поколений={engine.generations}  hall={engine.hall_size}")
            hall = engine.run(verbose=True)
            print(f"[get_shuttle] GA done — hall {len(hall)}  best f={hall[0].fitness:.4g} island={hall[0].island}")

            # --- выгрузка кучи алгоритмов ---
            import numpy as np
            hall_dir = out / "algorithms"
            hall_dir.mkdir(exist_ok=True, parents=True)

            # сохраняем га_холл в удобном формате
            hall_json = []
            for rank, g in enumerate(hall[:args.hall_size]):
                # baseline острова
                baseline = "sta"
                try:
                    for isl in engine.islands:
                        if isl.strategy == g.island:
                            baseline = isl.baseline
                            break
                except: pass
                # метрики уже внутри g.metrics, но пересчитаем exact для консистентности (N4)
                # для скорости hall используем 161 узел, только для best потом 321
                try:
                    eval_nodes = 161 if rank > 0 else runner.cfg.get("validation_nodes",321)
                    metrics = runner.evaluate(np.asarray(g.theta), baseline=baseline, exact_nodes=eval_nodes, subdivisions=runner.cfg.get("validation_subdivisions",4))
                except Exception:
                    metrics = g.metrics if hasattr(g,"metrics") else {}
                rec = {
                    "rank": rank,
                    "theta": g.theta.tolist(),
                    "fitness": float(g.fitness),
                    "island": g.island,
                    "baseline": baseline,
                    "status": getattr(g,"status","ok"),
                    "n_total": float(metrics.get("n_total", 0)) if isinstance(metrics, dict) else None,
                    "pos_err_m": float(metrics.get("position_error_m", 0)) if isinstance(metrics, dict) else None,
                    "fmin_hz": float(metrics.get("minimum_frequency_hz", 0)) if isinstance(metrics, dict) else None,
                }
                hall_json.append(rec)
                # отдельный файл на алгоритм
                (hall_dir / f"algo_{rank:03d}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
                # также waveform для топ-1 (не 3, экономия)
                if rank < 1:
                    try:
                        from shuttle_sim.pipeline import Runner as R2  # reuse
                        tr = runner.evaluate(np.asarray(g.theta), baseline=baseline, exact_nodes=161, subdivisions=runner.cfg.get("validation_subdivisions",4), keep=True)
                        # tr is (result, trace)
                        if isinstance(tr, tuple):
                            _, trace = tr
                            import csv
                            p = hall_dir / f"algo_{rank:03d}_waveform.csv"
                            with open(p,"w",newline="") as f:
                                w=csv.writer(f); w.writerow(["time_s"]+runner.trap.names)
                                for ti, row in zip(trace["ct"], trace["cmd"]):
                                    w.writerow([ti]+row.tolist())
                    except Exception as e:
                        print(f"[get_shuttle] waveform for rank {rank} fail: {e}")

            (out / "ga_hall.json").write_text(json.dumps(hall_json, indent=2, ensure_ascii=False), encoding="utf-8")

            # итоговый best — для прошивки
            best = hall[0]
            baseline0 = hall_json[0]["baseline"]
            try:
                metrics0 = runner.evaluate(np.asarray(best.theta), baseline=baseline0, exact_nodes=321)
                (out / "ga_best.json").write_text(json.dumps({"theta": best.theta.tolist(), "fitness": float(best.fitness), "baseline": baseline0, "metrics": metrics0, "island": best.island}, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception as e:
                print(f"best export fail {e}")

            # summary.md
            md = [f"# get_shuttle — {out}", "", f"config: `{cfg_path}` → `{out}/input_config.json`", f"hall: {len(hall_json)} алгоритмов", ""]
            md.append("| rank | island | baseline | fitness | n_total | pos µm | fmin MHz | theta |")
            md.append("|---:|---|---|---:|---:|---:|---:|---|")
            for r in hall_json[:min(12, len(hall_json))]:
                md.append(f"| {r['rank']} | {r['island']} | {r['baseline']} | {r['fitness']:.4f} | {r['n_total']:.3g} | {r['pos_err_m']*1e6:.2f} | {(r['fmin_hz'] or 0)/1e6:.2f} | `{','.join(f'{x:.2f}' for x in r['theta'][:4])}...` |")
            md += ["", "## Как прошить", "```bash", f"# лучший (rank 0) — из {out}/algorithms/algo_000.json", f"python -m shuttle_sim --config {out}/input_config.json --out output/flash --full-field  # проверка", "```", ""]
            md += ["## Дашборд", f"Если запускал с --dashboard — открой http://localhost:{args.port} → вкладки Setup/Training/GA/Trap (анимация)", ""]
            (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
            print(f"[get_shuttle] записал {out}/ga_hall.json  + {hall_dir}/*  + summary.md")
            print(f"[get_shuttle] лучший: island={best.island} baseline={baseline0} f={best.fitness:.4g} n={hall_json[0]['n_total']:.3g}")
            print(f"[get_shuttle] пример: cat {out}/algorithms/algo_000.json | python -m json.tool")

        except Exception as e:
            print(f"[get_shuttle] GA упал: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()
            if dash_proc: dash_proc.terminate()
            return 1

    print(f"[get_shuttle] готово → {out}  (куча алгоритмов в ga_hall.json и algorithms/)")
    if dash_proc:
        print(f"[get_shuttle] дашборд всё ещё висит на http://localhost:{args.port} — Ctrl+C чтобы остановить (или оставь)")
        try:
            dash_proc.wait()
        except KeyboardInterrupt:
            dash_proc.terminate()
            print("дашборд остановлен")
    return 0

if __name__ == "__main__":
    sys.exit(main())
