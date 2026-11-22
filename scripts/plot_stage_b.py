"""Summarize a Stage B results.jsonl: per (beta, eps) probabilities with Wilson
intervals and the log-log plot.

Run:  .venv/bin/python scripts/plot_stage_b.py results/stage_b.jsonl
Writes <stem>_summary.csv and plots/<stem>_prob_vs_eps.png beside the input.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from hitw.analysis import plot_stage_b, stage_b_frame, summarize_stage_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    args = ap.parse_args()
    path = Path(args.results)
    df = stage_b_frame(path)
    print(f"{len(df)} Stage B trials; q* valid in {int(df['q_star_valid'].sum())}")
    summ = summarize_stage_b(df)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    csv = path.with_name(path.stem + "_summary.csv")
    summ.to_csv(csv, index=False)
    png = plot_stage_b(df, path.parent / "plots" / f"{path.stem}_prob_vs_eps.png")
    print(f"wrote {csv} and {png}")


if __name__ == "__main__":
    main()
