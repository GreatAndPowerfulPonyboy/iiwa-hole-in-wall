"""Milestone 6 analysis of a Stage A sweep: per-condition success with Wilson
intervals, paired McNemar and Wilcoxon tables, and the plots (success vs eps,
outcome types, planner time, time-to-success CDF, IRIS region volume).

Run:  .venv/bin/python scripts/analyze_sweep.py results/sweep.jsonl [--stage-b results/stage_b.jsonl]
Writes <stem>_summary.csv, <stem>_paired.csv, <stem>_report.md and plots/<stem>_*.png beside the input.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from hitw.analysis import (SERIES, paired_table, plot_failure_types, plot_region_volume, plot_solve_time,
                           plot_stage_b, plot_success_vs_eps, plot_time_to_success_cdf, results_frame,
                           stage_b_frame, summarize, summarize_stage_b)


def md_table(df: pd.DataFrame, cols: list[str], fmt=None) -> str:
    fmt = fmt or {}
    head = "| " + " | ".join(str(c) for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for _, r in df[cols].iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt:
                cells.append(fmt[c](v))
            elif isinstance(v, float):
                cells.append(f"{v:.3g}")
            elif isinstance(v, str) and len(v) > 60:
                cells.append(v)
            else:
                cells.append(str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--stage-b", default=None)
    args = ap.parse_args()
    path = Path(args.results)
    plots = path.parent / "plots"
    df = results_frame(path)
    methods = [m for m in SERIES if m in set(df["method"])]
    print(f"{len(df)} trials, methods {methods}, status counts {df['status'].value_counts().to_dict()}, "
          f"killed {int(df['killed'].sum()) if 'killed' in df else 0}")
    summ = summarize(df)
    summ.to_csv(path.with_name(path.stem + "_summary.csv"), index=False)
    paired = paired_table(df, methods)
    paired.to_csv(path.with_name(path.stem + "_paired.csv"), index=False)
    out = [plot_success_vs_eps(df, plots / f"{path.stem}_success_vs_eps.png"),
           plot_failure_types(df, plots / f"{path.stem}_failure_types.png", methods),
           plot_solve_time(df, plots / f"{path.stem}_solve_time.png", methods),
           plot_time_to_success_cdf(df, plots / f"{path.stem}_time_to_success_cdf.png", methods)]
    if any(m.startswith("Iris") for m in methods):
        out.append(plot_region_volume(df, plots / f"{path.stem}_region_volume.png"))
    sb_md = ""
    if args.stage_b:
        sb = stage_b_frame(args.stage_b)
        sbs = summarize_stage_b(sb)
        out.append(plot_stage_b(sb, plots / f"{Path(args.stage_b).stem}_prob_vs_eps.png"))
        sb_md = "\n## Stage B\n\n" + md_table(
            sbs, ["beta", "eps", "b1_valid", "b1_restarts", "b1_p", "b1_lo", "b1_hi", "b1_p_within_K",
                  "b2_hits", "b2_samples", "b2_p", "b2_upper95"])
    # report
    summ_md = summ.copy()
    summ_md["eps_mm"] = (1e3 * summ_md["eps"]).round(2)
    summ_md["rate"] = summ_md.apply(lambda r: f"{r['k']}/{r['n']} = {r['p']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]", axis=1)
    pivot = summ_md.pivot_table(index=["method", "beta", "n_walls"], columns="eps_mm", values="rate", aggfunc="first")
    pivot = pivot[sorted(pivot.columns, reverse=True)]
    fails = summ_md.groupby("method")[[c for c in summ_md.columns if c.startswith("fail_")]].sum()
    sig = paired[paired["p_mcnemar"] < 0.05].sort_values("p_mcnemar")
    md = [f"# Sweep report: {path.name}\n",
          f"{len(df)} trials; methods {methods}; status counts {df['status'].value_counts().to_dict()}.\n",
          "## Success rate (k/n = p [Wilson 95%]) by eps (mm)\n",
          md_table(pivot.reset_index(), list(pivot.reset_index().columns)), "\n",
          "## Failure labels summed over conditions\n",
          md_table(fails.reset_index(), list(fails.reset_index().columns)), "\n",
          f"## Paired comparisons with exact McNemar p < 0.05 ({len(sig)} of {len(paired)} pairs x conditions)\n",
          md_table(sig, ["beta", "n_walls", "eps", "A", "B", "n", "both", "only_A", "only_B", "neither", "p_mcnemar",
                         "n_joint", "time_A_median", "time_B_median", "p_wilcoxon_time"]) if len(sig) else "none\n",
          "\n## Planner time, Wilcoxon signed-rank on jointly solved seeds (p < 0.05 count per pair)\n",
          md_table(paired[paired["p_wilcoxon_time"] < 0.05].groupby(["A", "B"]).size()
                   .to_frame("conditions_significant").reset_index(), ["A", "B", "conditions_significant"])
          if (paired["p_wilcoxon_time"] < 0.05).any() else "none\n",
          sb_md, "\n## Plots\n"] + [f"- {p}" for p in out]
    rp = path.with_name(path.stem + "_report.md")
    rp.write_text("\n".join(str(x) for x in md) + "\n")
    print(f"wrote {rp} and {len(out)} plots")


if __name__ == "__main__":
    main()
