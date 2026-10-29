"""CLI for island GA: python -m shuttle_sim.ga --config configs/shuttle_ga_island.json --out output/ga_100um --dashboard --port 8000

Never blocks evolution — dashboard (if --dashboard) is forked as separate process via multiprocessing,
or if unavailable, writes state.json for external server to poll.
"""
import argparse, json, pathlib, sys, time, subprocess, os

def main():
    ap=argparse.ArgumentParser(description="Island GA for shuttling")
    ap.add_argument("--config", required=True, help="GA config json (contains trap/task/hardware + ga)")
    ap.add_argument("--out", default="output/ga_run", help="output dir")
    ap.add_argument("--dashboard", action="store_true", help="start live dashboard server on --port")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--generations", type=int, default=None, help="override generations")
    args=ap.parse_args()

    cfg_path=pathlib.Path(args.config)
    cfg=json.loads(cfg_path.read_text())

    # Split: runner config vs ga config
    # If file has top-level "ga" key, use it; otherwise treat whole as runner config + ga defaults.
    ga_cfg=cfg.get("ga", {})
    runner_cfg=cfg
    # remove ga key for runner
    if "ga" in runner_cfg:
        runner_cfg=dict(runner_cfg)  # copy
        runner_cfg.pop("ga")
    if runner_cfg.get("trap", {}).get("kind") == "gds" and runner_cfg["trap"].get("path"):
        gp = pathlib.Path(runner_cfg["trap"]["path"])
        if not gp.is_absolute():
            for cand in (
                cfg_path.resolve().parent / gp,
                pathlib.Path.cwd() / gp,
                pathlib.Path.cwd() / "configs" / gp.name,
                pathlib.Path.cwd() / "configs" / "uploads" / gp.name,
                pathlib.Path.cwd() / "examples" / "gds" / gp.name,
            ):
                if cand.exists():
                    gp = cand.resolve()
                    break
            runner_cfg["trap"] = dict(runner_cfg["trap"])
            runner_cfg["trap"]["path"] = str(gp)
    if args.generations is not None:
        ga_cfg["generations"]=args.generations
    # defaults
    if "generations" not in ga_cfg:
        ga_cfg["generations"]=20
    if "islands" not in ga_cfg:
        ga_cfg["islands"]=[
            {"strategy":"S","pop":20,"baseline":"smooth"},
            {"strategy":"A","pop":20,"baseline":"sta"},
            {"strategy":"V","pop":20,"baseline":"sta"},  # voltage variable ΔV
            {"strategy":"W","pop":20,"baseline":"sta"},  # frequency variable ω(s)
            {"strategy":"R","pop":25,"baseline":"sta"},
            {"strategy":"H","pop":20,"baseline":"smooth"},
        ]
    if "migration_interval" not in ga_cfg: ga_cfg["migration_interval"]=5
    if "n_migrants" not in ga_cfg: ga_cfg["n_migrants"]=2
    if "seed" not in ga_cfg: ga_cfg["seed"]=42

    # Start dashboard server as subprocess if requested
    dash_proc=None
    if args.dashboard:
        out=pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        # ensure dashboard server module exists
        cmd=[sys.executable,"-m","shuttle_sim.dashboard.server","--out",str(out),"--port",str(args.port)]
        # use start_process via subprocess Popen so it doesn't block GA
        print(f"[GA] starting dashboard on 0.0.0.0:{args.port} -> preview https://{args.port}-<sandbox>.e2b.app")
        dash_proc=subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # wait a bit for server to bind
        time.sleep(1.5)
        if dash_proc.poll() is not None:
            out_log=dash_proc.stdout.read() if dash_proc.stdout else ""
            print(f"[GA] dashboard failed to start: {out_log}")
            dash_proc=None
        else:
            print(f"[GA] dashboard PID {dash_proc.pid}")

    # Now run GA (heavy)
    from shuttle_sim.pipeline import Runner
    from shuttle_sim.ga.engine import GAEngine

    print(f"[GA] loading runner config from {cfg_path}")
    runner=Runner(runner_cfg)
    print("[GA] preparing trap (support voltages, SVD modes)...")
    runner.prepare()
    print(f"[GA] prepared route_nodes {len(runner.sroute)} trap={runner.cfg['trap']['kind']} distance {runner.task['distance_m']*1e6:.1f}um duration {runner.task['duration_s']*1e6:.1f}us")
    engine=GAEngine(runner, ga_cfg, out_dir=args.out)
    print(f"[GA] islands {len(engine.islands)} generations {engine.generations} cuda {engine.cuda}")
    hall=engine.run(verbose=True)
    print(f"[GA] done. hall top {hall[0].fitness:.4g} theta {hall[0].theta.round(3).tolist()} island {hall[0].island}")
    # also export best waveform artifacts akin to pipeline
    # N4 fix: hall contains islands with different baselines (smooth vs sta); fitness and metrics must use SAME baseline
    try:
        best=hall[0]
        from pathlib import Path
        out=Path(args.out)
        # Determine baseline for best island (from engine islands, not hardcoded sta)
        best_baseline = 'sta'
        try:
            for isl in engine.islands:
                if isl.strategy == best.island:
                    best_baseline = isl.baseline
                    break
        except Exception:
            best_baseline = 'sta'
        # Re-evaluate with best's own baseline for consistency; use full theta (8/10/12) – N4, R6
        try:
            metrics=runner.evaluate(best.theta, baseline=best_baseline, exact_nodes=runner.cfg.get('validation_nodes',321), subdivisions=runner.cfg.get('validation_subdivisions',4))
        except Exception:
            metrics=runner.evaluate(best.to_theta8(), baseline=best_baseline, exact_nodes=runner.cfg.get('validation_nodes',321), subdivisions=runner.cfg.get('validation_subdivisions',4))
        # Recompute fitness consistently with same baseline metrics (not mixing baselines)
        try:
            from shuttle_sim.ga.fitness import scalar_fitness
            fitness_consistent = scalar_fitness(
                metrics, runner.task, distortion=metrics.get('distortion'), theta=best.theta,
                control_bounds=best.bounds,
                control_regularization_weight=runner.cfg.get('control_regularization_weight',1e-4),
                loss_weights=runner.loss_weights)
        except Exception:
            fitness_consistent = best.fitness
        (out/"ga_best.json").write_text(json.dumps({"theta":best.theta.tolist(),"fitness":fitness_consistent,"metrics":metrics,"island":best.island,"baseline":best_baseline},indent=2, default=str))
        # also save hall
        (out/"ga_hall.json").write_text(json.dumps([{"theta":g.theta.tolist(),"fitness":g.fitness,"island":g.island,"status":g.status,"metrics":g.metrics} for g in hall],indent=2, default=str))
        print(f"[GA] wrote {out/'ga_best.json'} and {out/'ga_hall.json'}")
    except Exception as e:
        print(f"[GA] best export failed {e}")

    if dash_proc:
        print(f"[GA] dashboard still running on port {args.port} — press Ctrl+C to stop. PID {dash_proc.pid}")
        try:
            # keep alive a bit so user can see final state, but not forever in CLI mode
            # If invoked as python -m shuttle_sim.ga --dashboard, we block until interrupted.
            dash_proc.wait()
        except KeyboardInterrupt:
            dash_proc.terminate()
            print("dashboard stopped")
    return 0

if __name__=="__main__":
    sys.exit(main())
