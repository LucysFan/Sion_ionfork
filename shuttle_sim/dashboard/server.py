"""Live dashboard server — reads state.json, serves static + JSON API, never blocks GA.

Run: python -m shuttle_sim.dashboard.server --out output/ga_run --port 8000
Preview in E2B: https://8000-<sandboxId>.e2b.app

Unified always-on mode: serves Setup tab (GDS upload + distance/time editor) + Training + GA + Trap.
POST /api/upload, /api/config, /api/launch work regardless of GA state.
"""
import argparse, json, pathlib, time, base64, subprocess, threading, os, re
from http.server import HTTPServer, SimpleHTTPRequestHandler
import sys

import argparse, json, pathlib, time
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os, sys


# Global job tracker for unified dashboard
_JOBS = {}
_JOBS_lock = threading.Lock()

class Handler(SimpleHTTPRequestHandler):

    def __init__(self, *args, out_dir=None, **kwargs):
        self.out_dir=pathlib.Path(out_dir) if out_dir else pathlib.Path("output/ga_run")
        super().__init__(*args, directory=str(pathlib.Path(__file__).parent / "static"), **kwargs)

    def do_GET(self):
        # CORS preflight is handled in do_OPTIONS
        # API endpoints
        if self.path.startswith("/api/state"):
            # Support ?run=output/xxx query
            out_dir = self._resolve_out_dir()
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.send_header("Access-Control-Allow-Origin","*")
            self.end_headers()
            p=out_dir/"state.json"
            if p.exists():
                self.wfile.write(p.read_bytes())
            else:
                # fallback to pipeline status.json if GA not running but pipeline did
                pp = out_dir/"status.json"
                if pp.exists():
                    try:
                        st=json.loads(pp.read_text())
                        self.wfile.write(json.dumps({"status":"pipeline_done","pipeline_status":st,"message":"pipeline finished, GA not running","global_generation":0}).encode())
                    except:
                        self.wfile.write(b'{"status":"waiting","message":"no state yet, run pipeline or GA from Setup tab"}')
                else:
                    self.wfile.write(b'{"status":"waiting","message":"no state yet, run pipeline or GA from Setup tab"}')
            return
        if self.path.startswith("/api/history"):
            out_dir = self._resolve_out_dir()
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.send_header("Access-Control-Allow-Origin","*")
            self.end_headers()
            hist=[]
            for f in sorted((out_dir/"history").glob("gen_*.json")):
                hist.append(f.name)
            self.wfile.write(json.dumps({"history":hist}).encode())
            return
        if self.path.startswith("/api/cuda"):
            out_dir = self._resolve_out_dir()
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.send_header("Access-Control-Allow-Origin","*")
            self.end_headers()
            p=out_dir/"state.json"
            cuda={}
            if p.exists():
                try:
                    cuda=json.loads(p.read_text()).get("cuda",{})
                except: pass
            self.wfile.write(json.dumps(cuda).encode())
            return
        if self.path.startswith("/api/runs"):
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.send_header("Access-Control-Allow-Origin","*")
            self.end_headers()
            root=pathlib.Path("output")
            runs=[]
            if root.exists():
                for d in sorted(root.iterdir()):
                    if d.is_dir():
                        has_state=(d/"state.json").exists()
                        has_status=(d/"status.json").exists()
                        has_config=(d/"config.json").exists()
                        runs.append({"name":d.name,"path":str(d),"has_state":has_state,"has_status":has_status,"has_config":has_config})
            self.wfile.write(json.dumps({"runs":runs,"current":str(self.out_dir)}).encode())
            return
        if self.path.startswith("/api/config"):
            out_dir = self._resolve_out_dir()
            # GET returns config.json from out_dir or base template
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.send_header("Access-Control-Allow-Origin","*")
            self.end_headers()
            # try out_dir/config.json, then configs/shuttle_gds.json, then unified
            cfg_path=None
            for cand in [out_dir/"config.json", pathlib.Path("configs/shuttle_gds.json"), pathlib.Path("configs/shuttle_reference.json")]:
                if cand.exists():
                    cfg_path=cand
                    break
            if cfg_path and cfg_path.exists():
                try:
                    cfg=json.loads(cfg_path.read_text())
                    # also inject current out_dir and available GDS list
                    gds_list=[]
                    gds_root=pathlib.Path("configs")
                    if gds_root.exists():
                        for g in gds_root.glob("*.gds"):
                            gds_list.append(str(g))
                        for g in pathlib.Path("examples/gds").glob("*.GDS"):
                            gds_list.append(str(g))
                    self.wfile.write(json.dumps({"config":cfg,"config_path":str(cfg_path),"gds_list":gds_list,"out_dir":str(out_dir)}).encode())
                    return
                except Exception as e:
                    self.wfile.write(json.dumps({"error":str(e)}).encode())
                    return
            self.wfile.write(b'{"error":"no config found"}')
            return
        if self.path.startswith("/api/pipeline_status"):
            out_dir = self._resolve_out_dir()
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.send_header("Access-Control-Allow-Origin","*")
            self.end_headers()
            pp=out_dir/"status.json"
            if pp.exists():
                try:
                    self.wfile.write(pp.read_bytes())
                except:
                    self.wfile.write(b'{}')
            else:
                self.wfile.write(b'{"status":"no pipeline yet"}')
            return
        if self.path.startswith("/api/jobs"):
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.send_header("Access-Control-Allow-Origin","*")
            self.end_headers()
            with _JOBS_lock:
                jobs={k:{"cmd":v["cmd"],"started":v["started"],"alive": v["proc"].poll() is None} for k,v in _JOBS.items()}
            self.wfile.write(json.dumps({"jobs":jobs}).encode())
            return
        # serve static
        if self.path=="/":
            self.path="/index.html"
        return super().do_GET()

    def _resolve_out_dir(self):
        # check ?run= query
        try:
            from urllib.parse import urlparse, parse_qs
            qs=parse_qs(urlparse(self.path).query)
            run=qs.get("run",[None])[0]
            if run:
                p=pathlib.Path(run)
                if p.exists() and p.is_dir():
                    return p
        except: pass
        return self.out_dir

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Access-Control-Allow-Methods","GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers","Content-Type")
        self.end_headers()

    def do_POST(self):
        length=int(self.headers.get('Content-Length',0))
        body=self.rfile.read(length) if length else b''
        path=self.path.split('?')[0]
        try:
            data=json.loads(body.decode()) if body else {}
        except:
            data={}
        # CORS
        def _reply(obj, code=200):
            self.send_response(code)
            self.send_header("Content-Type","application/json")
            self.send_header("Access-Control-Allow-Origin","*")
            self.end_headers()
            self.wfile.write(json.dumps(obj).encode())

        if path=="/api/upload":
            # data: {filename, content_base64} or {filename, content_text}
            fname=data.get("filename","uploaded.gds")
            # sanitize
            fname=re.sub(r'[^a-zA-Z0-9_.-]','_', pathlib.Path(fname).name)
            if not fname.lower().endswith(".gds"):
                fname+=".gds"
            out_path=pathlib.Path("configs/uploads")
            out_path.mkdir(parents=True, exist_ok=True)
            dest=out_path/fname
            try:
                if "content_base64" in data:
                    raw=base64.b64decode(data["content_base64"])
                    dest.write_bytes(raw)
                elif "content_text" in data:
                    dest.write_text(data["content_text"])
                else:
                    _reply({"error":"no content_base64 or content_text"},400)
                    return
                _reply({"ok":True,"path":str(dest),"filename":fname})
            except Exception as e:
                _reply({"error":str(e)},500)
            return
        if path=="/api/config":
            # save new config to a temp unified file
            try:
                # data is full config or partial {distance_m, duration_s, trap_path, ...}
                # if partial, merge into base template
                base_path=pathlib.Path("configs/shuttle_gds.json")
                if "trap" in data and "task" in data:
                    cfg=data
                else:
                    base=json.loads(base_path.read_text())
                    cfg=base
                    if "distance_m" in data:
                        cfg["task"]["distance_m"]=float(data["distance_m"])
                    if "duration_s" in data:
                        cfg["task"]["duration_s"]=float(data["duration_s"])
                    if "hold_s" in data:
                        cfg["task"]["hold_s"]=float(data["hold_s"])
                    if "protocol" in data:
                        cfg["task"]["protocol"]=str(data["protocol"])
                    if "gds_path" in data or "trap_path" in data:
                        gpath=data.get("gds_path") or data.get("trap_path")
                        cfg["trap"]["path"]=str(gpath)
                        # also try to update dc_indices if provided
                        if "electrode_map" in data:
                            cfg["trap"]["electrode_map"]=data["electrode_map"]
                    # ga generations if provided
                    if "generations" in data:
                        cfg.setdefault("ga",{})["generations"]=int(data["generations"])
                # write to unified
                dest=pathlib.Path("configs/unified_live.json")
                dest.write_text(json.dumps(cfg,indent=2))
                _reply({"ok":True,"path":str(dest),"config":cfg})
            except Exception as e:
                _reply({"error":str(e)},500)
            return
        if path=="/api/launch":
            # data: {mode: "pipeline"|"ga", distance_m, duration_s, gds_path, generations, out_dir, protocol}
            try:
                mode=data.get("mode","ga")
                distance=data.get("distance_m")
                duration=data.get("duration_s")
                gds_path=data.get("gds_path") or data.get("trap_path")
                gens=int(data.get("generations", 8))
                out_dir_str=data.get("out","output/live_run")
                protocol=data.get("protocol","one_way")
                # Build config from unified or base
                base_cfg_path=pathlib.Path("configs/unified_live.json")
                if not base_cfg_path.exists():
                    base_cfg_path=pathlib.Path("configs/shuttle_gds.json")
                cfg=json.loads(base_cfg_path.read_text())
                # apply overrides
                if distance is not None:
                    cfg["task"]["distance_m"]=float(distance)
                if duration is not None:
                    cfg["task"]["duration_s"]=float(duration)
                if gds_path:
                    cfg["trap"]["path"]=str(gds_path)
                    cfg["trap"]["kind"]="gds"
                if protocol:
                    cfg["task"]["protocol"]=str(protocol)
                # write config for this run
                out_path=pathlib.Path(out_dir_str)
                out_path.mkdir(parents=True, exist_ok=True)
                cfg_path=out_path/"config.json"
                cfg_path.write_text(json.dumps(cfg,indent=2))
                # also write a copy as run config
                # launch subprocess
                if mode=="pipeline":
                    cmd=[sys.executable,"-m","shuttle_sim","--config",str(cfg_path),"--out",str(out_path)]
                else:
                    # GA
                    # ensure ga key exists
                    cfg.setdefault("ga",{})
                    cfg["ga"]["generations"]=gens
                    cfg_path.write_text(json.dumps(cfg,indent=2))
                    cmd=[sys.executable,"-m","shuttle_sim.ga","--config",str(cfg_path),"--out",str(out_path),"--generations",str(gens)]
                # launch detached
                proc=subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                job_id=f"{mode}_{int(time.time())}"
                with _JOBS_lock:
                    _JOBS[job_id]={"proc":proc,"cmd":cmd,"started":time.time(),"out":str(out_path)}
                _reply({"ok":True,"job_id":job_id,"cmd":cmd,"out":str(out_path),"mode":mode})
            except Exception as e:
                _reply({"error":str(e)},500)
            return
        _reply({"error":"unknown POST path"},404)


    def log_message(self, format, *args):
        # quiet except errors
        sys.stderr.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), format%args))

def main():
    ap=argparse.ArgumentParser(description="GA live dashboard")
    ap.add_argument("--out", default="output/ga_run")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    args=ap.parse_args()
    out=pathlib.Path(args.out)
    # ensure static exists
    static=pathlib.Path(__file__).parent/"static"
    static.mkdir(parents=True, exist_ok=True)
    if not (static/"index.html").exists():
        print(f"no {static/'index.html'} — creating minimal placeholder")
        (static/"index.html").write_text("<h1>GA Dashboard — no index.html yet</h1>")
    # handler factory
    def handler(*a, **kw):
        return Handler(*a, out_dir=args.out, **kw)
    # allow preview host: need to handle Host header? SimpleHTTPRequestHandler does not check host.
    # For E2B we need to set CORS and not block iframe — already Access-Control-Allow-Origin *
    server=HTTPServer((args.host, args.port), handler)
    print(f"Dashboard serving {static} and API from {out} at http://{args.host}:{args.port} (preview https://{args.port}-<sandbox>.e2b.app)")
    print(f"Out dir: {out.resolve()} state: {(out/'state.json').resolve()}")
    # also print cuda info
    try:
        from shuttle_sim.ga.cuda import cuda_info
        print("CUDA:", cuda_info())
    except Exception as e:
        print("cuda info fail", e)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stop")
        server.shutdown()

if __name__=="__main__":
    main()
