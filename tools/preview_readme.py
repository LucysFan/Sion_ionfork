"""Render README.md to a GitHub-like HTML preview (for local viewing only).

Usage:  python tools/preview_readme.py [--out /tmp/readme.html]

Requires the `markdown` package.  The output is *not* part of the repository.
"""
from __future__ import annotations

import argparse
import pathlib
import re

import markdown

ROOT = pathlib.Path(__file__).resolve().parents[1]

CSS = """
:root{--fg:#1f2328;--muted:#59636e;--border:#d1d9e0;--accent:#0969da;--code:#f6f8fa}
*{box-sizing:border-box}
body{margin:0;background:#fff;color:var(--fg);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans",Helvetica,Arial,sans-serif}
.markdown-body{max-width:980px;margin:0 auto;padding:40px 24px 80px}
img{max-width:100%;height:auto;border-radius:6px}
h1,h2,h3,h4{font-weight:600;line-height:1.25;margin-top:1.6em;margin-bottom:.6em}
h1{font-size:2em;border-bottom:1px solid var(--border);padding-bottom:.3em}
h2{font-size:1.5em;border-bottom:1px solid var(--border);padding-bottom:.3em;margin-top:2.2em}
h3{font-size:1.2em}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
code{background:var(--code);padding:.2em .4em;border-radius:6px;font-size:85%;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:var(--code);padding:16px;overflow:auto;border-radius:6px}
pre code{background:none;padding:0;font-size:88%}
table{border-collapse:collapse;width:100%;margin:1em 0;display:block;overflow:auto}
th,td{border:1px solid var(--border);padding:6px 13px;text-align:left;vertical-align:top}
th{background:var(--code);font-weight:600}
tr:nth-child(2n) td{background:#fafbfc}
blockquote{margin:1em 0;padding:0 1em;color:var(--muted);border-left:.25em solid var(--border)}
details{margin:1em 0;padding:8px 12px;border:1px solid var(--border);border-radius:6px;background:#fbfcfd}
summary{cursor:pointer;font-weight:600}
hr{height:1px;border:0;background:var(--border);margin:2em 0}
sub{color:var(--muted)}
p>img{display:block;margin:0 auto}
"""


def render(md_text: str, asset_prefix: str = "") -> str:
    # GitHub renders markdown inside <details> when a blank line follows <summary>;
    # python-markdown needs an explicit opt-in instead.
    src = md_text.replace("<details>", '<details markdown="1">')
    body = markdown.markdown(
        src,
        extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list", "md_in_html"],
    )
    if asset_prefix:
        body = re.sub(r'src="assets/', f'src="{asset_prefix}assets/', body)
    return body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default=str(ROOT / "README.md"))
    ap.add_argument("--out", default=str(ROOT / "readme-preview.html"))
    ap.add_argument("--asset-prefix", default="", help="prefix so images resolve from another dir")
    args = ap.parse_args()

    body = render(pathlib.Path(args.readme).read_text(), args.asset_prefix)
    out = pathlib.Path(args.out)
    out.write_text(
        "<!doctype html><html lang='ru'><head><meta charset='utf-8'>"
        "<title>README preview</title><style>" + CSS + "</style></head>"
        "<body><article class='markdown-body'>" + body + "</article></body></html>"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
