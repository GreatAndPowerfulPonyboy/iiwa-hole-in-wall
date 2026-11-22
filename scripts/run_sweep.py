"""Run a sweep of trials into a results.jsonl (resumable; trials already in the
file are skipped).  Each trial runs in its own subprocess under a kill timer.

Run:  .venv/bin/python scripts/run_sweep.py --out results/pilot_m3.jsonl --methods KTO \
          --seeds 0-9 --betas 0 0.1 --n-walls 1 2 --workers 6 --cap 60
      --eps defaults to the approved 6-point grid (20 mm .. 2 mm); pass values in metres to override.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hitw import params
from hitw.sweep import grid, run_sweep


def parse_seeds(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", default=["KTO"],
                    help="KTO GCS IrisZo+GCS IrisNp2+GCS IrisNp+GCS IrisZo+GCS-defmargin RRT-Connect (M1..M4 accepted)")
    ap.add_argument("--seeds", default="0-9", help="e.g. 0-99 or 0,1,5")
    ap.add_argument("--betas", nargs="+", type=float, default=list(params.BETAS))
    ap.add_argument("--n-walls", nargs="+", type=int, default=list(params.N_WALLS))
    ap.add_argument("--eps", nargs="+", type=float, default=list(params.EPS_GRID))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cap", type=float, default=None, help="per-trial wall-clock cap, s (all methods)")
    ap.add_argument("--caps", default=None, help="JSON file {method: cap_s} of frozen per-method caps (overrides --cap)")
    args = ap.parse_args()
    trials = grid(args.methods, parse_seeds(args.seeds), args.betas, args.eps, args.n_walls)
    cap = args.cap
    if args.caps:
        import json
        cap = {m: float(v) for m, v in json.load(open(args.caps)).items()}
    run_sweep(trials, args.out, cap=cap, workers=args.workers)


if __name__ == "__main__":
    main()
