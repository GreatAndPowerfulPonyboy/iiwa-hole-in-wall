"""Summarize a results.jsonl: per-condition success rates with Wilson 95%
intervals, solve-time medians, failure labels, and the success-vs-eps plot.

Run:  .venv/bin/python scripts/plot_results.py results/pilot_m3.jsonl
Writes <stem>_summary.csv and plots/<stem>_success_vs_eps.png beside the input.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from hitw.analysis import plot_success_vs_eps, results_frame, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    args = ap.parse_args()
    path = Path(args.results)
    df = results_frame(path)
    print(f"{len(df)} trials, methods {sorted(df['method'].unique())}, "
          f"status counts {df['status'].value_counts().to_dict()}")
    summ = summarize(df)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    cols = ["method", "beta", "n_walls", "eps", "n", "k", "p", "lo", "hi", "solve_median", "solve_p95", "evals_median", "min_d_min"]
    extra = [c for c in summ.columns if c.startswith("fail_")]
    print(summ[cols + extra].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    csv = path.with_name(path.stem + "_summary.csv")
    summ.to_csv(csv, index=False)
    png = plot_success_vs_eps(df, path.parent / "plots" / f"{path.stem}_success_vs_eps.png")
    print(f"wrote {csv} and {png}")
    killed = df[df.get("killed", False) == True] if "killed" in df else df.iloc[0:0]
    if len(killed):
        print(f"{len(killed)} trials killed by the wall-clock cap")
    if "total_time" in df:
        print(f"per-trial total time: median {df['total_time'].median():.1f} s, p95 {df['total_time'].quantile(0.95):.1f} s "
              f"(solve median {df['solve_time'].median():.1f} s, recheck median {df['recheck_time'].median():.1f} s)")


if __name__ == "__main__":
    main()
