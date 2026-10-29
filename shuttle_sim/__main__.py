import os as _os
_os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import argparse, json, sys
from pathlib import Path
from .pipeline import Runner, write_json
from .field import Rejected


def main():
    p=argparse.ArgumentParser(description='Geometry-aware single-ion simulation; NEVER apparatus control')
    p.add_argument('--config',default='configs/shuttle_reference.json')
    p.add_argument('--out',default='output/shuttle_reference')
    p.add_argument('--full-field',action='store_true',help='also validate centroid with nonlinear Sion forces')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path(a.config).read_text())
    if cfg['trap']['kind']=='gds':
        p = Path(cfg['trap']['path'])
        if not p.is_absolute():
            for cand in (
                Path(a.config).resolve().parent / p,
                Path.cwd() / p,
                Path.cwd() / 'configs' / p.name,
                Path.cwd() / 'configs' / 'uploads' / p.name,
                Path.cwd() / 'examples' / 'gds' / p.name,
            ):
                if cand.exists():
                    p = cand.resolve()
                    break
            else:
                p = (Path(a.config).resolve().parent / p).resolve()
        cfg['trap']['path']=str(p)
    try:
        status=Runner(cfg).run(out,a.full_field)
    except (Rejected,ValueError,FileNotFoundError) as e:
        write_json(out/'failure.json',{'status':getattr(e,'code','invalid_input'),'message':str(e)})
        print(str(e),file=sys.stderr);return 2
    print(json.dumps(status,indent=2));return 0


if __name__=='__main__':sys.exit(main())
