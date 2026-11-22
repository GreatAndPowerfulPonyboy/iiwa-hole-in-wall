"""Milestone 5 cap calibration from a pilot results.jsonl.

For each method, over successful trials: median and p95 of the planner time
(solve_time + region_build_time), of the iteration measure (n_evals for M1,
samples for M4) and of the process time (total_time).  Proposed wall-clock
cap = max(10 x median, 3 x p95) of the planner time, clipped to --ceiling;
proposed iteration cap = 2 x p95.  The process cap enforced by the sweep
driver is the planner cap plus the re-check/spawn allowance (p95 of
total_time - planner time), since the driver kills whole processes.

Run:  .venv/bin/python scripts/calibrate_caps.py results/pilot_m5.jsonl --ceiling 300
Writes <stem>_caps.json beside the input: {method: process cap in seconds}.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from hitw.analysis import results_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--ceiling", type=float, default=300.0, help="hard ceiling on the planner cap, s")
    args = ap.parse_args()
    path = Path(args.results)
    df = results_frame(path)
    df["planner_time"] = df["solve_time"].fillna(0) + df["region_build_time"].fillna(0)
    df["overhead"] = df["total_time"] - df["planner_time"]
    df["iters"] = df["n_evals"].where(df["n_evals"].notna(), df["iterations"])
    rows = []
    caps = {}
    for m, g in df.groupby("method"):
        ok = g[g["success"]]
        if len(ok) == 0:
            rows.append({"method": m, "n": len(g), "n_success": 0})
            continue
        med, p95 = float(ok["planner_time"].median()), float(ok["planner_time"].quantile(0.95))
        cap = min(args.ceiling, max(10 * med, 3 * p95))
        it_med = float(ok["iters"].median()) if ok["iters"].notna().any() else np.nan
        it_p95 = float(ok["iters"].quantile(0.95)) if ok["iters"].notna().any() else np.nan
        over = float(g["overhead"].quantile(0.95))
        rows.append({"method": m, "n": len(g), "n_success": len(ok), "planner_median_s": med, "planner_p95_s": p95,
                     "planner_max_s": float(ok["planner_time"].max()), "iter_median": it_med, "iter_p95": it_p95,
                     "iter_cap_2p95": 2 * it_p95 if not np.isnan(it_p95) else np.nan,
                     "overhead_p95_s": over, "planner_cap_s": cap, "process_cap_s": cap + over,
                     "killed": int(g["killed"].sum()) if "killed" in g else 0,
                     "max_planner_over_cap": int((g["planner_time"] > cap).sum())})
        caps[m] = round(cap + over, 1)
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    out = path.with_name(path.stem + "_caps.json")
    out.write_text(json.dumps(caps, indent=1))
    print(f"\nproposed process caps written to {out}: {caps}")


if __name__ == "__main__":
    main()
