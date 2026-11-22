"""Stage B sweep: one trial per (seed, beta, eps) running B1 (IK restarts)
then B2 (rejection sampling under B1's wall-clock), one JSON line each in
the output file (resumable; each trial in its own subprocess).

Run:  .venv/bin/python scripts/run_stage_b.py --out results/stage_b.jsonl --seeds 0-9 --K 30 --workers 4 --cap 900
      --b2-budget <s> fixes B2's budget instead of pairing it to B1's time.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hitw import params
from hitw.sweep import grid, run_sweep
from run_sweep import parse_seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--betas", nargs="+", type=float, default=list(params.BETAS))
    ap.add_argument("--eps", nargs="+", type=float, default=list(params.EPS_GRID))
    ap.add_argument("--K", type=int, default=30, help="IK restarts per trial")
    ap.add_argument("--bound-factor", type=float, default=0.5, help="IK min-distance bound = factor x eps")
    ap.add_argument("--b2-budget", type=float, default=None, help="B2 wall-clock, s (default: B1's time)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cap", type=float, default=None, help="per-trial process cap, s")
    args = ap.parse_args()
    trials = grid(["B"], parse_seeds(args.seeds), args.betas, args.eps, [1])
    run_sweep(trials, args.out, cap=args.cap, workers=args.workers,
              method_kwargs={"K": args.K, "bound_factor": args.bound_factor, "b2_budget": args.b2_budget})


if __name__ == "__main__":
    main()
