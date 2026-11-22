"""Execute every code cell of a notebook in one namespace (test harness).

Usage: python -I run_notebook.py <notebook> [--mode quick] [--skip c]
"""
import argparse
import json
import os
import sys
import time
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nb")
    ap.add_argument("--mode", default=None)
    ap.add_argument("--only", default=None, help="substring filter on cell source")
    ap.add_argument("--set", action="append", default=[],
                    help="KEY=VALUE: override a top-level assignment line (test only)")
    args = ap.parse_args()

    nb = json.load(open(args.nb))
    ns = {"__name__": "__main__"}
    t_all = time.time()
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if args.mode:
            src = src.replace('MODE = "full"', f'MODE = "{args.mode}"')
        for kv in args.set:
            k, v = kv.split("=", 1)
            import re as _re
            src = _re.sub(rf"^{_re.escape(k)}\s*=\s*.*$", f"{k} = {v}", src,
                          count=1, flags=_re.M)
        if args.only and args.only not in src:
            continue
        t0 = time.time()
        print(f"\n--- cell {i} ({len(src.splitlines())} lines) ---", flush=True)
        try:
            exec(compile(src, f"<cell {i}>", "exec"), ns)
        except Exception:
            print(f"!!! cell {i} FAILED")
            traceback.print_exc()
            sys.exit(1)
        print(f"    ok  {time.time()-t0:.1f}s", flush=True)
    print(f"\nALL CELLS OK in {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
