"""Figures for the paper from results files.

  success_vs_eps.png    connect task: dense re-check success rate vs eps, Wilson 95%,
                        one panel per (beta, walls), one line per run
  planner_time.png      connect task: median planner time (solve + region build) on
                        successful trials vs eps, interquartile band
  find_vs_eps.png       find task: probability of a valid configuration per attempt
                        vs eps, Wilson 95%; zero hits drawn at the 95% upper bound

Run:  python3 scripts/figures.py results/main.jsonl results/find.jsonl --out figures/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hitw.analysis import (check_rng_schemes, display_label, find_frame, planner_time,  # noqa: E402
                           results_frame, summarize, summarize_find, supported)
from hitw.cli import load_planner_sources  # noqa: E402

# One colour per run across every figure; runs not listed take the palette in order.
COLORS = {"kto": "#2a78d6", "gcs-box": "#eb6834", "iris-gcs": "#1baf7a", "rrt-connect": "#eda100",
          "iris-gcs[iris=np2]": "#6250d6", "iris-gcs[iris=np]": "#e34948", "iris-gcs[margin=default]": "#8a8780",
          "ik-restarts": "#6250d6", "uniform-sampling": "#e34948"}
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948", "#8a8780")
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def labels_of(df: pd.DataFrame) -> list[str]:
    present = list(dict.fromkeys(df["config_label"]))
    return [x for x in COLORS if x in present] + sorted(x for x in present if x not in COLORS)


def color(label: str, labels: list[str]) -> str:
    others = [x for x in labels if x not in COLORS]
    return COLORS.get(label) or PALETTE[others.index(label) % len(PALETTE)]


def style(ax, xlabel: str, ylabel: str, title: str, logy: bool = False) -> None:
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xticks([2, 5, 10, 20])
    ax.set_xticklabels(["2", "5", "10", "20"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title(title, loc="left", fontsize=11, color=TEXT)
    ax.set_xlabel(xlabel, color=TEXT2)
    ax.set_ylabel(ylabel, color=TEXT2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT2, length=0)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, path: Path) -> Path:
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def panel_title(beta: float, walls: int) -> str:
    return f"beta = {beta:g}, {walls} wall{'s' if walls != 1 else ''}"


def success_vs_eps(df: pd.DataFrame, path: Path) -> Path:
    summ = summarize(df)
    summ = summ[summ["n"] > 0]
    panels = sorted(set(zip(summ["beta"], summ["n_walls"])))
    labels = labels_of(summ)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6), sharey=True, squeeze=False)
    for ax, (b, w) in zip(axes[0], panels):
        sub = summ[(summ["beta"] == b) & (summ["n_walls"] == w)]
        for i, lab in enumerate(labels):
            g = sub[sub["config_label"] == lab].sort_values("eps")
            if not len(g):
                continue
            # a small multiplicative x offset per run keeps saturated series apart
            x = 1e3 * g["eps"].values * (1.0 + 0.025 * (i - (len(labels) - 1) / 2))
            yerr = [np.maximum(0.0, g["p"] - g["lo"]), np.maximum(0.0, g["hi"] - g["p"])]
            ax.errorbar(x, g["p"], yerr=yerr, color=color(lab, labels), lw=2, marker="o", ms=5, capsize=3,
                        label=f"{g['display'].iloc[0]} (n = {int(g['n'].iloc[0])})")
        ax.set_ylim(-0.03, 1.03)
        style(ax, "clearance eps (mm)", "success rate" if ax is axes[0][0] else "", panel_title(b, w))
    axes[0][0].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=2)
    fig.suptitle("Dense re-check success vs clearance (Wilson 95%)", x=0.01, ha="left", fontsize=12, color=TEXT)
    return save(fig, path)


def planner_time_vs_eps(df: pd.DataFrame, path: Path) -> Path:
    df = supported(df)
    labels = labels_of(df)
    df = df[df["success"]].copy()
    df["planner_time"] = planner_time(df)
    panels = sorted(set(zip(df["beta"], df["n_walls"])))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4), sharey=True, squeeze=False)
    for ax, (b, w) in zip(axes[0], panels):
        sel = df[(df["beta"] == b) & (df["n_walls"] == w)]
        for i, lab in enumerate(labels):
            g = sel[sel["config_label"] == lab]
            if not len(g):
                continue
            q = g.groupby("eps")["planner_time"].quantile([0.25, 0.5, 0.75]).unstack().sort_index()
            x = 1e3 * q.index.values * (1.0 + 0.02 * (i - (len(labels) - 1) / 2))
            c = color(lab, labels)
            ax.fill_between(x, q[0.25], q[0.75], color=c, alpha=0.15, linewidth=0)
            ax.plot(x, q[0.5], color=c, lw=2, marker="o", ms=5, label=display_label(g["method"].iloc[0], lab))
        style(ax, "clearance eps (mm)", "planner time, s (median, IQR)" if ax is axes[0][0] else "",
              panel_title(b, w), logy=True)
    axes[0][0].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=3)
    fig.suptitle("Planner time on successful trials (solve + region build)", x=0.01, ha="left", fontsize=12,
                 color=TEXT)
    return save(fig, path)


def find_vs_eps(df: pd.DataFrame, path: Path) -> Path:
    summ = summarize_find(df)
    summ = summ[summ["attempts"] > 0]
    betas = sorted(summ["beta"].unique())
    labels = labels_of(summ)
    unit = {"ik-restarts": "per restart", "uniform-sampling": "per sample"}
    fig, axes = plt.subplots(1, len(betas), figsize=(4.4 * len(betas), 4.6), sharey=True, squeeze=False)
    for ax, b in zip(axes[0], betas):
        for lab in labels:
            g = summ[(summ["beta"] == b) & (summ["config_label"] == lab)].sort_values("eps")
            if not len(g):
                continue
            method = g["method"].iloc[0]
            name = g["display"].iloc[0] + (f" ({unit[method]})" if method in unit else "")
            x, p, lo, hi = 1e3 * g["eps"].values, g["p"].values, g["lo"].values, g["hi"].values
            c, hit = color(lab, labels), p > 0
            if hit.any():
                ax.errorbar(x[hit], p[hit], yerr=[np.maximum(0, p[hit] - lo[hit]), np.maximum(0, hi[hit] - p[hit])],
                            color=c, lw=2, marker="o", ms=6, capsize=3, label=name)
            if (~hit).any():
                ax.plot(x[~hit], (3.0 / g["attempts"].values)[~hit], ls="none", marker="v", ms=8, mfc="none", mec=c,
                        mew=2, label=f"{name}: 0 hits, 95% upper bound")
        style(ax, "clearance eps (mm)", "probability of a valid crossing" if ax is axes[0][0] else "",
              f"beta = {b:g} ({'3' if b == 0 else '7'}-D)", logy=True)
        ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.suptitle("Find task: probability per attempt (Wilson 95%)", x=0.01, ha="left", fontsize=12, color=TEXT)
    return save(fig, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", nargs="+", help="results files (connect and/or find records)")
    ap.add_argument("--out", default="figures", help="directory for the PNGs")
    args = ap.parse_args()
    out = Path(args.out)
    for p in args.results:
        load_planner_sources(Path(p))
    connect = [results_frame(p) for p in args.results]
    connect = pd.concat([d for d in connect if len(d)], ignore_index=True) if any(len(d) for d in connect) else None
    find = [find_frame(p) for p in args.results]
    find = pd.concat([d for d in find if len(d)], ignore_index=True) if any(len(d) for d in find) else None
    for frame in (connect, find):
        if frame is not None:
            check_rng_schemes(frame)
    written = []
    if connect is not None:
        written += [success_vs_eps(connect, out / "success_vs_eps.png"),
                    planner_time_vs_eps(connect, out / "planner_time.png")]
    if find is not None:
        written.append(find_vs_eps(find, out / "find_vs_eps.png"))
    print("\n".join(str(p) for p in written) or "no records")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
