"""Statistics and plots over results.jsonl.

Success rates carry Wilson 95% intervals.  Paired comparisons across shared
seeds (exact McNemar with the four paired outcome counts) and Wilcoxon
signed-rank on jointly solved seeds are milestone 6; this module holds the
primitives they will use.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .evaluate import load_results

# Reference categorical palette (dataviz skill, fixed slot order), one hue per
# method so colour follows the entity across every plot.
SERIES = {"KTO": "#2a78d6", "GCS": "#eb6834", "IrisZo+GCS": "#1baf7a", "RRT-Connect": "#eda100",
          "IrisNp2+GCS": "#6250d6", "IrisNp+GCS": "#e34948", "IrisZo+GCS-defmargin": "#8a8780",
          "B1": "#6250d6", "B2": "#e34948"}
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float, float]:
    """(p_hat, lower, upper): Wilson score interval for k successes in n."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, centre - half, centre + half)


def results_frame(path: str | Path) -> pd.DataFrame:
    rows = load_results(path)
    df = pd.DataFrame(rows)
    if len(df):
        df["success"] = df["status"] == "success"
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per (method, beta, n_walls, eps): n, successes, Wilson interval, solve
    time median and p95 over successes, and failure-label counts."""
    out = []
    for (m, b, w, e), g in df.groupby(["method", "beta", "n_walls", "eps"]):
        k, n = int(g["success"].sum()), len(g)
        p, lo, hi = wilson(k, n)
        ok = g[g["success"]]["solve_time"]
        labels = g["failure_label"].value_counts(dropna=True).to_dict()
        out.append({"method": m, "beta": b, "n_walls": w, "eps": e, "n": n, "k": k, "p": p,
                    "lo": lo, "hi": hi,
                    "solve_median": float(ok.median()) if len(ok) else np.nan,
                    "solve_p95": float(ok.quantile(0.95)) if len(ok) else np.nan,
                    "evals_median": float(g[g["success"]]["n_evals"].median()) if len(ok) else np.nan,
                    "min_d_min": float(g["min_signed_distance"].dropna().min()) if g["min_signed_distance"].notna().any() else np.nan,
                    **{f"fail_{lab}": cnt for lab, cnt in labels.items()}})
    return pd.DataFrame(out).sort_values(["method", "beta", "n_walls", "eps"], ascending=[True, True, True, False])


def _style(ax, xlabel, ylabel, title):
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


def plot_success_vs_eps(df: pd.DataFrame, path: str | Path, title: str | None = None):
    """Success rate vs eps (log x), one panel per (beta, n_walls), one line
    per method, Wilson 95% intervals as error bars."""
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summ = summarize(df)
    panels = sorted(set(zip(summ["beta"], summ["n_walls"])))
    methods = [m for m in SERIES if m in set(summ["method"])] + sorted(set(summ["method"]) - set(SERIES))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6), sharey=True, squeeze=False)
    for ax, (b, w) in zip(axes[0], panels):
        sub = summ[(summ["beta"] == b) & (summ["n_walls"] == w)]
        for i, m in enumerate(methods):
            g = sub[sub["method"] == m].sort_values("eps")
            if not len(g):
                continue
            # a small multiplicative x offset per method keeps saturated series apart
            x = 1e3 * g["eps"].values * (1.0 + 0.025 * (i - (len(methods) - 1) / 2))
            col = SERIES.get(m, "#52514e")
            yerr = [np.maximum(0.0, g["p"] - g["lo"]), np.maximum(0.0, g["hi"] - g["p"])]
            ax.errorbar(x, g["p"], yerr=yerr, color=col, lw=2, alpha=0.9,
                        marker="o", ms=5, capsize=3, label=f"{m} (n = {int(g['n'].iloc[0])})")
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 20])
        ax.set_xticklabels(["2", "5", "10", "20"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylim(-0.03, 1.03)
        _style(ax, "clearance eps (mm)", "success rate" if ax is axes[0][0] else "",
               f"beta = {b:g}, {w} wall{'s' if w != 1 else ''}")
        if ax is axes[0][0]:
            ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=2)
    fig.suptitle(title or "Dense re-check success vs clearance (Wilson 95%)", x=0.01, ha="left",
                 fontsize=12, color=TEXT)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def mcnemar_exact(both: int, only_a: int, only_b: int, neither: int) -> float:
    """Two-sided exact McNemar p-value from the four paired outcome counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    return float(min(1.0, 2 * stats.binom.cdf(k, n, 0.5)))


def paired_outcomes(df: pd.DataFrame, a: str, b: str, beta: float, n_walls: int, eps: float) -> dict:
    """Four paired counts over the seeds both methods ran."""
    sel = df[(df["beta"] == beta) & (df["n_walls"] == n_walls) & (df["eps"] == eps)]
    sa = sel[sel["method"] == a].set_index("seed")["success"]
    sb = sel[sel["method"] == b].set_index("seed")["success"]
    seeds = sa.index.intersection(sb.index)
    sa, sb = sa.loc[seeds], sb.loc[seeds]
    counts = {"both": int((sa & sb).sum()), "only_a": int((sa & ~sb).sum()),
              "only_b": int((~sa & sb).sum()), "neither": int((~sa & ~sb).sum()), "n": len(seeds)}
    counts["p_mcnemar"] = mcnemar_exact(counts["both"], counts["only_a"], counts["only_b"], counts["neither"])
    return counts


# --- Stage B -----------------------------------------------------------------

def stage_b_frame(path: str | Path) -> pd.DataFrame:
    """One row per Stage B trial with the B1 and B2 summaries flattened."""
    rows = []
    for r in load_results(path):
        if r.get("method") != "B" or "b1" not in r:
            continue
        b1, b2 = r["b1"], r["b2"]
        rows.append({"seed": r["seed"], "beta": r["beta"], "eps": r["eps"], "sequence": r["sequence"][0],
                     "q_star_valid": r["q_star_valid"], "K": b1["K"], "b1_valid": b1["n_valid"],
                     "b1_reported": b1["n_reported_success"], "b1_false_pos": b1["n_false_positive"],
                     "b1_p_hat": b1["p_hat"], "b1_time": b1["total_time"],
                     "b1_restart_median": b1["restart_time_median"], "b1_first_valid_t": b1["time_to_first_valid"],
                     "b2_n": b2["n_samples"], "b2_hits": b2["n_hits"], "b2_rate": b2["hit_rate"],
                     "b2_upper": b2["hit_rate_upper95"], "b2_budget": b2["budget_s"],
                     "b2_per_s": b2["samples_per_s"]})
    return pd.DataFrame(rows)


def summarize_stage_b(df: pd.DataFrame) -> pd.DataFrame:
    """Per (beta, eps): pooled per-restart probability for B1 with its Wilson
    interval and P(success within K) = 1 - (1 - p)^K; pooled per-sample hit
    rate for B2 with its Wilson interval, or the 95% upper bound ~3/n after
    zero hits; B1 and B2 compute (seconds)."""
    out = []
    for (b, e), g in df.groupby(["beta", "eps"]):
        k1, n1 = int(g["b1_valid"].sum()), int(g["K"].sum())
        p1, lo1, hi1 = wilson(k1, n1)
        K = int(g["K"].iloc[0])
        k2, n2 = int(g["b2_hits"].sum()), int(g["b2_n"].sum())
        p2, lo2, hi2 = wilson(k2, n2)
        out.append({"beta": b, "eps": e, "trials": len(g), "K": K,
                    "b1_valid": k1, "b1_restarts": n1, "b1_p": p1, "b1_lo": lo1, "b1_hi": hi1,
                    "b1_p_within_K": 1 - (1 - p1) ** K, "b1_trials_solved": int((g["b1_valid"] > 0).sum()),
                    "b1_false_pos": int(g["b1_false_pos"].sum()), "b1_time_median": float(g["b1_time"].median()),
                    "b2_hits": k2, "b2_samples": n2, "b2_p": p2, "b2_lo": lo2, "b2_hi": hi2,
                    "b2_upper95": (3.0 / n2) if (n2 and k2 == 0) else np.nan,
                    "b2_trials_solved": int((g["b2_hits"] > 0).sum()), "b2_time_median": float(g["b2_budget"].median()),
                    "b2_samples_per_s": float(g["b2_per_s"].median())})
    return pd.DataFrame(out).sort_values(["beta", "eps"], ascending=[True, False])


def plot_stage_b(df: pd.DataFrame, path: str | Path, title: str | None = None):
    """Probability of a valid crossing vs eps, log-log, one panel per beta:
    B1 per-restart probability and B2 per-sample hit rate with Wilson 95%
    intervals; a zero-hit condition is drawn as an open marker at its 95%
    upper bound."""
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summ = summarize_stage_b(df)
    betas = sorted(summ["beta"].unique())
    fig, axes = plt.subplots(1, len(betas), figsize=(4.4 * len(betas), 4.6), sharey=True, squeeze=False)
    for ax, b in zip(axes[0], betas):
        sub = summ[summ["beta"] == b].sort_values("eps")
        x = 1e3 * sub["eps"].values
        for key, label in (("b1", "B1 IK restarts (per restart)"), ("b2", "B2 rejection sampling (per sample)")):
            col = SERIES["B1" if key == "b1" else "B2"]
            p, lo, hi = sub[f"{key}_p"].values, sub[f"{key}_lo"].values, sub[f"{key}_hi"].values
            hit = p > 0
            if hit.any():
                yerr = [np.maximum(0, p[hit] - lo[hit]), np.maximum(0, hi[hit] - p[hit])]
                ax.errorbar(x[hit], p[hit], yerr=yerr, color=col, lw=2, marker="o", ms=6, capsize=3, label=label)
            if (~hit).any():
                ub = (3.0 / sub[f"{key}_samples" if key == "b2" else "b1_restarts"].values)[~hit]
                ax.plot(x[~hit], ub, ls="none", marker="v", ms=8, mfc="none", mec=col, mew=2,
                        label=f"{label}: 0 hits, 95% upper bound")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([2, 5, 10, 20])
        ax.set_xticklabels(["2", "5", "10", "20"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        _style(ax, "clearance eps (mm)", "probability of a valid crossing" if ax is axes[0][0] else "",
               f"beta = {b:g} ({'3' if b == 0 else '7'}-D)")
        ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.suptitle(title or "Stage B: finding a crossing configuration (Wilson 95%)", x=0.01, ha="left",
                 fontsize=12, color=TEXT)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --- Milestone 6: paired comparisons and the remaining plots -------------------

FAILURE_ORDER = ["success", "infeasible_reported", "reported_feasible_but_collides",
                 "collision_free_but_misses_window", "timeout"]
FAILURE_COLORS = {"success": "#c9e7d6", "infeasible_reported": "#eb6834",
                  "reported_feasible_but_collides": "#e34948", "collision_free_but_misses_window": "#eda100",
                  "timeout": "#52514e"}


def planner_time(df: pd.DataFrame) -> pd.Series:
    """solve_time plus region_build_time (0 when absent): the time a method
    needs before it has an answer."""
    return df["solve_time"].fillna(0) + df.get("region_build_time", pd.Series(0, index=df.index)).fillna(0)


def paired_table(df: pd.DataFrame, methods: list[str] | None = None) -> pd.DataFrame:
    """Every method pair at every (beta, n_walls, eps): the four paired
    outcome counts on shared seeds, exact McNemar p, and Wilcoxon signed-rank
    on planner time over jointly solved seeds."""
    df = df.copy()
    df["planner_time"] = planner_time(df)
    methods = methods or [m for m in SERIES if m in set(df["method"])]
    rows = []
    for (b, w, e), sel in df.groupby(["beta", "n_walls", "eps"]):
        for i, a in enumerate(methods):
            for c in methods[i + 1:]:
                sa = sel[sel["method"] == a].set_index("seed")
                sc = sel[sel["method"] == c].set_index("seed")
                seeds = sa.index.intersection(sc.index)
                if len(seeds) == 0:
                    continue
                xa, xc = sa.loc[seeds, "success"], sc.loc[seeds, "success"]
                both, only_a, only_c = int((xa & xc).sum()), int((xa & ~xc).sum()), int((~xa & xc).sum())
                neither = int((~xa & ~xc).sum())
                joint = seeds[(xa & xc).values]
                ta, tc = sa.loc[joint, "planner_time"].values, sc.loc[joint, "planner_time"].values
                if len(joint) >= 5 and np.any(ta != tc):
                    w_p = float(stats.wilcoxon(ta, tc).pvalue)
                else:
                    w_p = np.nan
                rows.append({"beta": b, "n_walls": w, "eps": e, "A": a, "B": c, "n": len(seeds),
                             "both": both, "only_A": only_a, "only_B": only_c, "neither": neither,
                             "p_mcnemar": mcnemar_exact(both, only_a, only_c, neither),
                             "n_joint": len(joint),
                             "time_A_median": float(np.median(ta)) if len(ta) else np.nan,
                             "time_B_median": float(np.median(tc)) if len(tc) else np.nan,
                             "p_wilcoxon_time": w_p})
    return pd.DataFrame(rows)


def _finish(fig, path):
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def _panels(df):
    return sorted(set(zip(df["beta"], df["n_walls"])))


def plot_failure_types(df: pd.DataFrame, path: str | Path, methods: list[str] | None = None):
    """Stacked bars of outcome types per method at each eps; one panel per
    (beta, n_walls)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = methods or [m for m in SERIES if m in set(df["method"])]
    panels = _panels(df)
    eps_vals = sorted(df["eps"].unique(), reverse=True)
    fig, axes = plt.subplots(len(panels), 1, figsize=(1.0 + 0.55 * len(eps_vals) * len(methods), 2.6 * len(panels)),
                             squeeze=False, sharex=True)
    labels = df["failure_label"].fillna("success")
    for ax, (b, w) in zip(axes[:, 0], panels):
        sel = df[(df["beta"] == b) & (df["n_walls"] == w)]
        lab = labels.loc[sel.index]
        x = 0
        ticks, tick_labels = [], []
        for e in eps_vals:
            group_start = x
            for m in methods:
                g = sel[(sel["method"] == m) & (sel["eps"] == e)]
                n = len(g)
                if n == 0:
                    continue
                bottom = 0.0
                counts = lab.loc[g.index].value_counts()
                for f in FAILURE_ORDER:
                    v = counts.get(f, 0) / n
                    if v > 0:
                        ax.bar(x, v, bottom=bottom, width=0.8, color=FAILURE_COLORS[f], edgecolor="white", linewidth=1)
                        bottom += v
                ax.plot([x], [-0.06], marker="s", ms=5, color=SERIES.get(m, "#52514e"), clip_on=False)
                x += 1
            ticks.append((group_start + x - 1) / 2)
            tick_labels.append(f"{1e3*e:.3g} mm")
            x += 1
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_ylim(-0.1, 1.0)
        _style(ax, "", "fraction of trials", f"beta = {b:g}, {w} wall{'s' if w != 1 else ''}")
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAILURE_COLORS[f]) for f in FAILURE_ORDER]
    handles += [plt.Line2D([], [], marker="s", ls="none", color=SERIES.get(m, "#52514e")) for m in methods]
    axes[-1, 0].legend(handles, FAILURE_ORDER + methods, frameon=False, fontsize=7, ncol=3,
                       loc="upper left", bbox_to_anchor=(0.0, -0.25))
    fig.suptitle("Outcome types by method and clearance (square = method, bar order within each eps group)",
                 x=0.01, ha="left", fontsize=11, color=TEXT)
    return _finish(fig, path)


def plot_solve_time(df: pd.DataFrame, path: str | Path, methods: list[str] | None = None):
    """Median planner time (solve + region build) vs eps, log y, IQR band,
    over successful trials; one panel per (beta, n_walls)."""
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = methods or [m for m in SERIES if m in set(df["method"])]
    df = df[df["success"]].copy()
    df["planner_time"] = planner_time(df)
    panels = _panels(df)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4), sharey=True, squeeze=False)
    for ax, (b, w) in zip(axes[0], panels):
        sel = df[(df["beta"] == b) & (df["n_walls"] == w)]
        for i, m in enumerate(methods):
            g = sel[sel["method"] == m].groupby("eps")["planner_time"]
            if g.ngroups == 0:
                continue
            q = g.quantile([0.25, 0.5, 0.75]).unstack().sort_index()
            x = 1e3 * q.index.values * (1.0 + 0.02 * (i - (len(methods) - 1) / 2))
            col = SERIES.get(m, "#52514e")
            ax.fill_between(x, q[0.25], q[0.75], color=col, alpha=0.15, linewidth=0)
            ax.plot(x, q[0.5], color=col, lw=2, marker="o", ms=5, label=m)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([2, 5, 10, 20])
        ax.set_xticklabels(["2", "5", "10", "20"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        _style(ax, "clearance eps (mm)", "planner time, s (median, IQR)" if ax is axes[0][0] else "",
               f"beta = {b:g}, {w} wall{'s' if w != 1 else ''}")
        if ax is axes[0][0]:
            ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=3)
    fig.suptitle("Planner time on successful trials (solve + IRIS region build)", x=0.01, ha="left",
                 fontsize=12, color=TEXT)
    return _finish(fig, path)


def plot_time_to_success_cdf(df: pd.DataFrame, path: str | Path, methods: list[str] | None = None):
    """Fraction of trials solved within t seconds of planner time (failures
    never count, so curves plateau at the success rate); per (beta, n_walls)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = methods or [m for m in SERIES if m in set(df["method"])]
    df = df.copy()
    df["planner_time"] = planner_time(df)
    panels = _panels(df)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4), sharey=True, squeeze=False)
    for ax, (b, w) in zip(axes[0], panels):
        sel = df[(df["beta"] == b) & (df["n_walls"] == w)]
        for m in methods:
            g = sel[sel["method"] == m]
            if len(g) == 0:
                continue
            t = np.sort(g.loc[g["success"], "planner_time"].values)
            if len(t):
                ax.step(np.concatenate([[1e-3], t]), np.concatenate([[0], np.arange(1, len(t) + 1) / len(g)]),
                        where="post", color=SERIES.get(m, "#52514e"), lw=2, label=f"{m} (n = {len(g)})")
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        _style(ax, "planner time t (s)", "fraction solved within t" if ax is axes[0][0] else "",
               f"beta = {b:g}, {w} wall{'s' if w != 1 else ''}")
        if ax is axes[0][0]:
            ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=3)
    fig.suptitle("Time-to-success CDF (all eps pooled; a curve plateaus at the success rate)", x=0.01, ha="left",
                 fontsize=12, color=TEXT)
    return _finish(fig, path)


def region_volume_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One row per IRIS window region: ellipsoid and sampled volume estimates."""
    rows = []
    for _, r in df.iterrows():
        rv = r.get("region_volume")
        if not isinstance(rv, dict):
            continue
        for i, w in enumerate(rv.get("windows", [])):
            rows.append({"method": r["method"], "beta": r["beta"], "eps": r["eps"], "n_walls": r["n_walls"],
                         "seed": r["seed"], "window": i + 1, "dim": w.get("dim"), "ellipsoid": w.get("ellipsoid"),
                         "sampled": w.get("sampled"), "build_s": rv["window_build"][i],
                         "contains_box": w.get("contains_tolerance_box")})
    return pd.DataFrame(rows)


def plot_region_volume(df: pd.DataFrame, path: str | Path):
    """Median inscribed-ellipsoid volume of the IRIS window regions vs eps,
    log-log, one panel per beta (3-D at beta 0, 7-D otherwise: volumes are
    not comparable across panels); IQR band; one line per IRIS variant."""
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rv = region_volume_frame(df)
    rv = rv[rv["ellipsoid"].notna() & (rv["ellipsoid"] > 0)]
    betas = sorted(rv["beta"].unique())
    fig, axes = plt.subplots(1, len(betas), figsize=(4.4 * len(betas), 4.4), squeeze=False)
    for ax, b in zip(axes[0], betas):
        sel = rv[rv["beta"] == b]
        for i, m in enumerate([m for m in SERIES if m in set(sel["method"])]):
            g = sel[sel["method"] == m].groupby("eps")["ellipsoid"]
            q = g.quantile([0.25, 0.5, 0.75]).unstack().sort_index()
            x = 1e3 * q.index.values * (1.0 + 0.02 * i)
            col = SERIES.get(m, "#52514e")
            ax.fill_between(x, q[0.25], q[0.75], color=col, alpha=0.15, linewidth=0)
            ax.plot(x, q[0.5], color=col, lw=2, marker="o", ms=5, label=m)
        # the tolerance box volume as the reference: (2 tol / sqrt(n))^n
        eps_vals = np.array(sorted(sel["eps"].unique()))
        n = 3 if b == 0 else 7
        rho = 1.131 if b == 0 else 1.602
        box = (2 * eps_vals / (2 * rho) / np.sqrt(n)) ** n
        ax.plot(1e3 * eps_vals, box, color=TEXT2, lw=1.5, ls="--", label="tolerance box volume")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([2, 5, 10, 20])
        ax.set_xticklabels(["2", "5", "10", "20"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        _style(ax, "clearance eps (mm)", f"inscribed ellipsoid volume (rad^{n})",
               f"beta = {b:g} ({n}-D)")
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=2)
    fig.suptitle("IRIS window-region volume vs clearance (median, IQR)", x=0.01, ha="left", fontsize=12, color=TEXT)
    return _finish(fig, path)
