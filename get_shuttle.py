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
        print(f"[get_shuttle] поднимаю дашборд на 127.0.0.1:{args.port} → http://localhost:{args.port}  (вкладки Setup/Training/GA/Trap)")
        dash_proc = subprocess.Popen([sys.executable, "-m", "shuttle_sim.dashboard.server", "--out", str(out), "--port", str(args.port), "--host", "127.0.0.1"])
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
                # метрики уже внутри g.metrics, но пересчитаем exact для консистентности (N4) через единый диспетчер
                # один геном -> один протокол везде (audit 2026-10-30): используем fitness.evaluate_exact для K/P/W
                try:
                    from shuttle_sim.ga.fitness import evaluate_exact as _eval_exact
                    # для скорости hall используем 161 узел для rank>0, но через unified dispatcher (изоляция W, K/P)
                    # временно подменяем validation_nodes/subdivisions в cfg, evaluate_exact возьмет их
                    orig_nodes = runner.cfg.get('validation_nodes')
                    orig_sub = runner.cfg.get('validation_subdivisions')
                    try:
                        eval_nodes = 161 if rank > 0 else runner.cfg.get("validation_nodes",321)
                        runner.cfg['validation_nodes'] = eval_nodes
                        # keep subdivisions as is
                        metrics = _eval_exact(g, runner, baseline=baseline)
                        # metrics already contains exact_nodes and exact_validated
                    finally:
                        if orig_nodes is not None:
                            runner.cfg['validation_nodes'] = orig_nodes
                        elif 'validation_nodes' in runner.cfg:
                            del runner.cfg['validation_nodes']
                        if orig_sub is not None:
                            runner.cfg['validation_subdivisions'] = orig_sub
                except Exception as _e:
                    # fallback to genome metrics if exact fails
                    try:
                        metrics = g.metrics if hasattr(g,"metrics") else {}
                    except: metrics = {}
                # hash export control from actual command array (audit: not just theta)
                try:
                    import hashlib, numpy as _np
                    # try to get actual cmd for hash (canonical: time, channel order, voltages)
                    cmd_hash = None
                    control_hash_theta = hashlib.sha256(json.dumps(g.theta.tolist()).encode()).hexdigest()[:12]
                    metrics['_export_control_hash_theta'] = control_hash_theta
                    try:
                        # generate actual commands via runner (island-aware) for hash
                        theta_arr = _np.asarray(g.theta)
                        # handle W split for cmd generation
                        if g.island in ('W','frequency','freq') and len(theta_arr)>=10:
                            theta_cmd = theta_arr[:8]
                            # need to prepare W to get correct voltage basis (use runner's current prepared state)
                            # For hash we can approximate via runner.commands with prepared W if needed
                            # To avoid mutating runner, we try direct commands for generic, else fallback to theta hash
                            try:
                                from shuttle_sim.ga.fitness import _prepare_for_W, _restore_after_W, _w_cache_key, _get_runner_w_cache
                                freq_c = theta_arr[8:10]
                                stashed_cmd = dict(task_mode=runner.task.get('frequency_mode'), task_coeffs=runner.task.get('frequency_coeffs'),
                                                   frequency_mode=getattr(runner,'frequency_mode','fixed'), frequency_coeffs=getattr(runner,'frequency_coeffs',[0.0,0.0]),
                                                   sroute=getattr(runner,'sroute',None), route_V=getattr(runner,'route_V',None), route_r=getattr(runner,'route_r',None),
                                                   route_H=getattr(runner,'route_H',None), route_G=getattr(runner,'route_G',None), route_BH=getattr(runner,'route_BH',None),
                                                   correction_modes=getattr(runner,'correction_modes',None), support_diagnostics=getattr(runner,'support_diagnostics',None),
                                                   voltage_spline=getattr(runner,'voltage_spline',None), Hspline=getattr(runner,'Hspline',None),
                                                   Gspline=getattr(runner,'Gspline',None), BHspline=getattr(runner,'BHspline',None))
                                _prepare_for_W(runner, freq_c)
                                # now generate cmd for the prepared runner
                                if g.island.startswith('K') or g.island.startswith('M') or g.island.startswith('C') or g.island in ('STITCH','KRAT'):
                                    import re
                                    m=re.match(r'^[KMC](\d+)$', g.island)
                                    segs=int(m.group(1)) if m else 'auto'
                                    _, cmd_arr, _ = runner.commands_segmented(theta_cmd, baseline=baseline, segments=segs)
                                elif g.island.startswith('P') or g.island in ('PERIODIC',):
                                    import re
                                    m=re.match(r'^P(\d+)$', g.island)
                                    reps=int(m.group(1)) if m else 'auto'
                                    _, cmd_arr, _ = runner.commands_krat_periodic(theta_cmd, baseline=baseline, repeats=reps)
                                else:
                                    _, cmd_arr, _ = runner.commands(theta_cmd, baseline=baseline)
                                _restore_after_W(runner, stashed_cmd)
                            except:
                                try: _restore_after_W(runner, stashed_cmd)
                                except: pass
                                # fallback generic
                                _, cmd_arr, _ = runner.commands(theta_arr, baseline=baseline)
                        else:
                            if g.island.startswith('K') or g.island.startswith('M') or g.island.startswith('C') or g.island in ('STITCH','KRAT'):
                                import re
                                m=re.match(r'^[KMC](\d+)$', g.island)
                                segs=int(m.group(1)) if m else 'auto'
                                _, cmd_arr, _ = runner.commands_segmented(theta_arr, baseline=baseline, segments=segs)
                            elif g.island.startswith('P') or g.island in ('PERIODIC',):
                                import re
                                m=re.match(r'^P(\d+)$', g.island)
                                reps=int(m.group(1)) if m else 'auto'
                                _, cmd_arr, _ = runner.commands_krat_periodic(theta_arr, baseline=baseline, repeats=reps)
                            else:
                                _, cmd_arr, _ = runner.commands(theta_arr, baseline=baseline)
                        # canonical payload hash: version, times, order, voltages, dtype, shape (audit)
                        try:
                            _t_for_hash = runner.grid()[0]
                            _times_bytes = _t_for_hash.tobytes() if hasattr(_t_for_hash,'tobytes') else str(_t_for_hash).encode()
                            cmd_bytes = cmd_arr.tobytes() if isinstance(cmd_arr, _np.ndarray) else str(cmd_arr).encode()
                            order = ','.join(getattr(runner.trap, 'names', []))
                            payload = b'v1|'+ _times_bytes + b'|' + order.encode() + b'|' + cmd_bytes + f"|{cmd_arr.dtype if hasattr(cmd_arr,'dtype') else 'unknown'}|{cmd_arr.shape if hasattr(cmd_arr,'shape') else 'unknown'}".encode()
                            try:
                                hw_id = f"{runner.hw.sample_rate_hz}|{getattr(runner.trap,'kind','unknown')}"
                                payload += b'|' + hw_id.encode()
                            except: pass
                            cmd_hash = hashlib.sha256(payload).hexdigest()[:16]
                            order_hash = hashlib.sha256(order.encode()).hexdigest()[:8]
                            metrics['_export_control_hash'] = f"{cmd_hash[:12]}-{order_hash}"
                            metrics['_export_control_hash_cmd'] = cmd_hash
                            metrics['_export_payload_schema'] = "v1_times_order_voltage_dtype_shape_hw"
                        except Exception as _e_hash2:
                            cmd_bytes = cmd_arr.tobytes() if isinstance(cmd_arr, _np.ndarray) else str(cmd_arr).encode()
                            order = ','.join(getattr(runner.trap, 'names', []))
                            cmd_hash = hashlib.sha256(cmd_bytes).hexdigest()[:16]
                            order_hash = hashlib.sha256(order.encode()).hexdigest()[:8]
                            metrics['_export_control_hash'] = f"{cmd_hash[:12]}-{order_hash}"
                            metrics['_export_control_hash_cmd'] = cmd_hash
                    except Exception as _e_hash:
                        metrics['_export_control_hash'] = control_hash_theta
                        metrics['_export_control_hash_error'] = str(_e_hash)[:200]
                    metrics['_export_protocol'] = runner.task.get('protocol','one_way')
                    metrics['_export_cycles'] = runner.task.get('cycles',1)
                    # also store actual fidelity from metrics
                    metrics['_export_actual_subdivisions'] = metrics.get('actual_subdivisions')
                    metrics['_export_actual_exact_nodes'] = metrics.get('actual_exact_nodes')
                    metrics['_export_hardware_dt_s'] = metrics.get('hardware_dt_s')
                except: pass
                # Use unified EvaluationResult: serialize actual verdict from exact assessment (audit)
                _accepted = bool(metrics.get("accepted", False)) if isinstance(metrics, dict) else False
                _validated = bool(metrics.get("exact_validated", metrics.get("validation_passed", False))) if isinstance(metrics, dict) else False
                _quality = metrics.get("quality_failures", []) if isinstance(metrics, dict) else []
                _status = getattr(g,"status","ok")
                # override status to reflect exact verdict if available
                if isinstance(metrics, dict) and "accepted" in metrics:
                    if not metrics.get("accepted", False):
                        _status = ",".join(metrics.get("quality_failures", ["quality_fail"])) if metrics.get("quality_failures") else "quality_fail"
                    elif not metrics.get("exact_validated", True):
                        # accepted but not numerically validated
                        _status = "ok_not_validated"
                rec = {
                    "rank": rank,
                    "theta": g.theta.tolist(),
                    "fitness": float(g.fitness),
                    "island": g.island,
                    "baseline": baseline,
                    "status": _status,
                    "accepted": _accepted,
                    "exact_validated": _validated,
                    "exact_constraints_passed": bool(metrics.get("exact_constraints_passed", _accepted)) if isinstance(metrics, dict) else None,
                    "exact_numerical_validation_passed": bool(metrics.get("exact_numerical_validation_passed", metrics.get("numerical_validation_passed", False))) if isinstance(metrics, dict) else None,
                    "quality_failures": _quality,
                    "validation_passed": bool(metrics.get("validation_passed", False)) if isinstance(metrics, dict) else None,
                    "n_total": float(metrics.get("n_total", 0)) if isinstance(metrics, dict) else None,
                    "pos_err_m": float(metrics.get("position_error_m", 0)) if isinstance(metrics, dict) else None,
                    "fmin_hz": float(metrics.get("minimum_frequency_hz", 0)) if isinstance(metrics, dict) else None,
                    "actual_subdivisions": metrics.get("actual_subdivisions") if isinstance(metrics, dict) else None,
                    "actual_exact_nodes": metrics.get("actual_exact_nodes") if isinstance(metrics, dict) else None,
                    "hardware_dt_s": metrics.get("hardware_dt_s") if isinstance(metrics, dict) else None,
                }
                hall_json.append(rec)
                # отдельный файл на алгоритм
                (hall_dir / f"algo_{rank:03d}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
                # также waveform для топ-1 — через единый диспетчер (K/P/W корректно)
                if rank < 1:
                    try:
                        from shuttle_sim.ga.fitness import _dispatch_evaluate, _restore_after_W
                        # waveform need keep=True: use dispatcher with keep
                        # _dispatch_evaluate doesn't support keep, so call runner.* directly via island logic but with stash handling
                        # Reuse _dispatch_evaluate path but with keep: we manually dispatch
                        import numpy as np
                        # isolate W if needed via _dispatch_evaluate's stash mechanism, but for keep we do direct
                        # For simplicity, call _dispatch_evaluate then also get trace via runner.evaluate with keep handling through same island
                        # We'll attempt to get trace via unified path: use _dispatch but with keep via underlying runner calls
                        # Fallback: direct runner dispatch based on island
                        island = g.island
                        theta_arr = np.asarray(g.theta)
                        # Determine if stitched/periodic/W
                        is_stitched = isinstance(island, str) and (island.startswith('K') or island.startswith('M') or island.startswith('C') or island in ('STITCH','KRAT'))
                        is_periodic = isinstance(island, str) and (island.startswith('P') or island in ('PERIODIC',))
                        is_W = island in ('W','frequency','freq') and len(theta_arr)>=10
                        # For W, theta voltage part
                        if is_W:
                            theta_use = theta_arr[:8]
                            freq_coeffs = theta_arr[8:10]
                            # prepare W temporarily
                            from shuttle_sim.ga.fitness import _prepare_for_W, _restore_after_W, _get_runner_w_cache, _w_cache_key
                            stashed=None
                            try:
                                stashed=dict(task_mode=runner.task.get('frequency_mode'), task_coeffs=runner.task.get('frequency_coeffs'),
                                             frequency_mode=getattr(runner,'frequency_mode','fixed'), frequency_coeffs=getattr(runner,'frequency_coeffs',[0.0,0.0]),
                                             sroute=getattr(runner,'sroute',None), route_V=getattr(runner,'route_V',None), route_r=getattr(runner,'route_r',None),
                                             route_H=getattr(runner,'route_H',None), route_G=getattr(runner,'route_G',None), route_BH=getattr(runner,'route_BH',None),
                                             correction_modes=getattr(runner,'correction_modes',None), support_diagnostics=getattr(runner,'support_diagnostics',None),
                                             voltage_spline=getattr(runner,'voltage_spline',None), Hspline=getattr(runner,'Hspline',None),
                                             Gspline=getattr(runner,'Gspline',None), BHspline=getattr(runner,'BHspline',None))
                                _prepare_for_W(runner, freq_coeffs)
                                # now dispatch based on protocol
                                if is_stitched:
                                    import re
                                    m=re.match(r'^[KMC](\d+)$', island)
                                    segs=int(m.group(1)) if m else 'auto'
                                    tr = runner.evaluate_segmented(theta_use, baseline=baseline, segments=segs, keep=True)
                                elif is_periodic:
                                    import re
                                    m=re.match(r'^P(\d+)$', island)
                                    reps=int(m.group(1)) if m else 'auto'
                                    tr = runner.evaluate_krat_periodic(theta_use, baseline=baseline, repeats=reps, keep=True)
                                else:
                                    tr = runner.evaluate(theta_use, baseline=baseline, keep=True)
                            finally:
                                if stashed is not None:
                                    try: _restore_after_W(runner, stashed)
                                    except: pass
                        else:
                            if is_stitched:
                                import re
                                m=re.match(r'^[KMC](\d+)$', island)
                                segs=int(m.group(1)) if m else 'auto'
                                tr = runner.evaluate_segmented(theta_arr, baseline=baseline, segments=segs, keep=True)
                            elif is_periodic:
                                import re
                                m=re.match(r'^P(\d+)$', island)
                                reps=int(m.group(1)) if m else 'auto'
                                tr = runner.evaluate_krat_periodic(theta_arr, baseline=baseline, repeats=reps, keep=True)
                            else:
                                tr = runner.evaluate(theta_arr, baseline=baseline, keep=True)
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

            # итоговый best — для прошивки через unified dispatcher (хеш control vs final evaluation фиксируется)
            best = hall[0]
            baseline0 = hall_json[0]["baseline"]
            try:
                from shuttle_sim.ga.fitness import evaluate_exact as _eval_best
                import hashlib, json as _json
                # ensure full validation nodes for best
                orig_nodes_best = runner.cfg.get('validation_nodes')
                try:
                    runner.cfg['validation_nodes'] = 321
                    metrics0 = _eval_best(best, runner, baseline=baseline0)
                finally:
                    if orig_nodes_best is not None:
                        runner.cfg['validation_nodes'] = orig_nodes_best
                    elif 'validation_nodes' in runner.cfg:
                        del runner.cfg['validation_nodes']
                # hash separation: control (actual cmd) vs final metrics for round-trip/repeated semantics
                try:
                    import numpy as _np2
                    # control hash from actual cmd (same as hall, but for best)
                    try:
                        theta_arr2 = _np2.asarray(best.theta)
                        if best.island in ('W','frequency','freq') and len(theta_arr2)>=10:
                            theta_cmd2 = theta_arr2[:8]
                            freq_c2 = theta_arr2[8:10]
                            stashed2 = dict(task_mode=runner.task.get('frequency_mode'), task_coeffs=runner.task.get('frequency_coeffs'),
                                            frequency_mode=getattr(runner,'frequency_mode','fixed'), frequency_coeffs=getattr(runner,'frequency_coeffs',[0.0,0.0]),
                                            sroute=getattr(runner,'sroute',None), route_V=getattr(runner,'route_V',None), route_r=getattr(runner,'route_r',None),
                                            route_H=getattr(runner,'route_H',None), route_G=getattr(runner,'route_G',None), route_BH=getattr(runner,'route_BH',None),
                                            correction_modes=getattr(runner,'correction_modes',None), support_diagnostics=getattr(runner,'support_diagnostics',None),
                                            voltage_spline=getattr(runner,'voltage_spline',None), Hspline=getattr(runner,'Hspline',None),
                                            Gspline=getattr(runner,'Gspline',None), BHspline=getattr(runner,'BHspline',None))
                            from shuttle_sim.ga.fitness import _prepare_for_W, _restore_after_W
                            _prepare_for_W(runner, freq_c2)
                            if baseline0.startswith('K') or baseline0.startswith('M') or baseline0.startswith('C') or baseline0 in ('STITCH','KRAT'):
                                import re as _re2
                                m2=_re2.match(r'^[KMC](\d+)$', best.island)
                                segs2=int(m2.group(1)) if m2 else 'auto'
                                _, cmd_arr2, _ = runner.commands_segmented(theta_cmd2, baseline=baseline0, segments=segs2)
                            elif best.island.startswith('P') or best.island in ('PERIODIC',):
                                import re as _re2b
                                m2b=_re2b.match(r'^P(\d+)$', best.island)
                                reps2=int(m2b.group(1)) if m2b else 'auto'
                                _, cmd_arr2, _ = runner.commands_krat_periodic(theta_cmd2, baseline=baseline0, repeats=reps2)
                            else:
                                _, cmd_arr2, _ = runner.commands(theta_cmd2, baseline=baseline0)
                            _restore_after_W(runner, stashed2)
                        else:
                            if best.island.startswith('K') or best.island.startswith('M') or best.island.startswith('C') or best.island in ('STITCH','KRAT'):
                                import re as _re3
                                m3=_re3.match(r'^[KMC](\d+)$', best.island)
                                segs3=int(m3.group(1)) if m3 else 'auto'
                                _, cmd_arr2, _ = runner.commands_segmented(theta_arr2, baseline=baseline0, segments=segs3)
                            elif best.island.startswith('P') or best.island in ('PERIODIC',):
                                import re as _re3b
                                m3b=_re3b.match(r'^P(\d+)$', best.island)
                                reps3=int(m3b.group(1)) if m3b else 'auto'
                                _, cmd_arr2, _ = runner.commands_krat_periodic(theta_arr2, baseline=baseline0, repeats=reps3)
                            else:
                                _, cmd_arr2, _ = runner.commands(theta_arr2, baseline=baseline0)
                        # canonical best hash with times
                        try:
                            _t2 = runner.grid()[0]
                            _t2b = _t2.tobytes() if hasattr(_t2,'tobytes') else str(_t2).encode()
                            _payload2 = b'v1|'+ _t2b + b'|' + ','.join(getattr(runner.trap,'names',[])).encode() + b'|' + cmd_arr2.tobytes() + f"|{cmd_arr2.dtype}|{cmd_arr2.shape}".encode()
                            ctrl_h = hashlib.sha256(_payload2).hexdigest()[:16]
                        except:
                            ctrl_h = hashlib.sha256(cmd_arr2.tobytes()).hexdigest()[:16]
                        order2 = ','.join(getattr(runner.trap, 'names', []))
                        order_h2 = hashlib.sha256(order2.encode()).hexdigest()[:8]
                        metrics0['_control_hash'] = f"{ctrl_h[:12]}-{order_h2}"
                        metrics0['_control_hash_cmd'] = ctrl_h
                    except Exception as _e_best_hash:
                        ctrl_h = hashlib.sha256(_json.dumps(best.theta.tolist()).encode()).hexdigest()[:16]
                        metrics0['_control_hash'] = ctrl_h
                        metrics0['_control_hash_error'] = str(_e_best_hash)[:200]
                    metrics_h = hashlib.sha256(_json.dumps({k: float(v) for k,v in metrics0.items() if isinstance(v,(int,float))}, sort_keys=True).encode()).hexdigest()[:16]
                    metrics0['_metrics_hash'] = metrics_h
                    metrics0['_export_semantics'] = f"{runner.task.get('protocol','one_way')}_c{runner.task.get('cycles',1)}"
                    metrics0['_export_actual_subdivisions'] = metrics0.get('actual_subdivisions')
                    metrics0['_export_actual_exact_nodes'] = metrics0.get('actual_exact_nodes')
                    metrics0['_export_hardware_dt_s'] = metrics0.get('hardware_dt_s')
                except: pass
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
