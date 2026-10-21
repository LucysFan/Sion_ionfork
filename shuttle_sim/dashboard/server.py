"""Live dashboard server — reads state.json, serves static + JSON API, never blocks GA.

Run: python -m shuttle_sim.dashboard.server --out output/ga_run --port 8000
Preview in E2B: https://8000-<sandboxId>.e2b.app
"""
import argparse, json, pathlib, time
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os, sys

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, out_dir=None, **kwargs):
        self.out_dir=pathlib.Path(out_dir) if out_dir else pathlib.Path("output/ga_run")
        super().__init__(*args, directory=str(pathlib.Path(__file__).parent / "static"), **kwargs)

    def do_GET(self):
        # API endpoints
        if self.path.startswith("/api/state"):
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.send_header("Access-Control-Allow-Origin","*")
            self.end_headers()
            p=self.out_dir/"state.json"
            if p.exists():
                self.wfile.write(p.read_bytes())
            else:
                self.wfile.write(b'{"status":"waiting","message":"GA not started or no state yet"}')
            return
        if self.path.startswith("/api/history"):
            # list history files
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.send_header("Access-Control-Allow-Origin","*")
            self.end_headers()
            hist=[]
            for f in sorted((self.out_dir/"history").glob("gen_*.json")):
                hist.append(f.name)
            self.wfile.write(json.dumps({"history":hist}).encode())
            return
        if self.path.startswith("/api/cuda"):
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.send_header("Access-Control-Allow-Origin","*")
            self.end_headers()
            # try to read cuda from state
            p=self.out_dir/"state.json"
            cuda={}
            if p.exists():
                try:
                    cuda=json.loads(p.read_text()).get("cuda",{})
                except: pass
            self.wfile.write(json.dumps(cuda).encode())
            return
        # serve static
        if self.path=="/":
            self.path="/index.html"
        return super().do_GET()

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
