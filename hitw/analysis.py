"""Statistics over results files.

Runs are grouped by (method, config_label): one planner with two configs is
two runs.  Success rates carry Wilson 95% intervals; paired comparisons
across shared seeds use the exact McNemar test on the four paired outcome
counts, and Wilcoxon signed-rank on planner time over jointly solved seeds.
Figures are made outside the package (scripts/figures.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .harness import load_results


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float, float]:
    """Return the Wilson score interval for k successes in n trials.

    Parameters
    ----------
    k : int
        Successes.
    n : int
        Trials.
    z : float
        Normal quantile (1.96 for 95%).

    Returns
    -------
    p_hat, lower, upper : float
        NaN when n = 0.
    """
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, centre - half, centre + half)


def display_label(method: str, config_label: str) -> str:
    """Return the label of a run for tables and figures: the planner's display name and its non-default options.

    Parameters
    ----------
    method : str
        Planner id.
    config_label : str
        The run's config label.

    Returns
    -------
    str
        E.g. ``"IRIS + GCS [iris=np2]"``; the id stands in for the display
        name of a planner that is not registered in this process.
    """
    from .api import get_planner
    try:
        name = get_planner(method).display_name
    except KeyError:
        name = method
    suffix = config_label[len(method):]
    return name + (f" {suffix}" if suffix else "")


def check_rng_schemes(df: pd.DataFrame) -> None:
    """Raise ValueError if one (method, config_label) mixes random-stream schemes.

    Records written with different planner random streams are not the same
    experiment, so they are not pooled.
    """
    if not len(df):
        return
    n = df.groupby(["method", "config_label"])["rng_scheme"].nunique()
    mixed = n[n > 1]
    if len(mixed):
        raise ValueError(f"records with different rng_scheme in {list(mixed.index)}; analyse them separately")


def results_frame(path: str | Path) -> pd.DataFrame:
    """Load the connect-task records of a results file.

    Parameters
    ----------
    path : str or Path
        Results file.

    Returns
    -------
    pandas.DataFrame
        One row per record, with a boolean ``success`` column.

    Raises
    ------
    ValueError
        If a (method, config_label) mixes rng schemes, or a record is not in
        the current format.
    """
    df = pd.DataFrame([r for r in load_results(path) if r["task"] == "connect"])
    if len(df):
        df["success"] = df["status"] == "success"
        check_rng_schemes(df)
    return df


def supported(df: pd.DataFrame) -> pd.DataFrame:
    """Return the trials whose planner handles their beta (status not "unsupported")."""
    return df[df["status"] != "unsupported"] if len(df) else df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize connect trials per (method, config_label, beta, n_walls, eps).

    Parameters
    ----------
    df : pandas.DataFrame
        From `results_frame`.

    Returns
    -------
    pandas.DataFrame
        n, successes k, Wilson interval (p, lo, hi), solve time median and
        p95 over successes, evaluations median, smallest re-check distance,
        and one ``fail_<label>`` count column per failure label.
        "unsupported" trials are not in n; they are counted in
        n_unsupported.
    """
    out = []
    for (m, c, b, w, e), g_all in df.groupby(["method", "config_label", "beta", "n_walls", "eps"]):
        g = supported(g_all)
        k, n = int(g["success"].sum()), len(g)
        p, lo, hi = wilson(k, n)
        ok = g[g["success"]]["solve_time"]
        labels = g["failure_label"].value_counts(dropna=True).to_dict()
        out.append({"method": m, "config_label": c, "display": display_label(m, c), "beta": b, "n_walls": w,
                    "eps": e, "n": n, "k": k, "p": p, "lo": lo, "hi": hi, "n_unsupported": len(g_all) - n,
                    "solve_median": float(ok.median()) if len(ok) else np.nan,
                    "solve_p95": float(ok.quantile(0.95)) if len(ok) else np.nan,
                    "evals_median": float(g[g["success"]]["n_evals"].median()) if len(ok) else np.nan,
                    "min_d_min": float(g["min_signed_distance"].dropna().min()) if g["min_signed_distance"].notna().any() else np.nan,
                    **{f"fail_{lab}": cnt for lab, cnt in labels.items()}})
    return pd.DataFrame(out).sort_values(["config_label", "beta", "n_walls", "eps"], ascending=[True, True, True, False])


def mcnemar_exact(both: int, only_a: int, only_b: int, neither: int) -> float:
    """Return the two-sided exact McNemar p-value from the four paired outcome counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    return float(min(1.0, 2 * stats.binom.cdf(k, n, 0.5)))


def planner_time(df: pd.DataFrame) -> pd.Series:
    """Return solve_time plus region_build_time (0 when absent): the time a run needs before it has an answer."""
    solve = pd.to_numeric(df["solve_time"], errors="coerce").fillna(0.0)
    build = pd.to_numeric(df.get("region_build_time", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    return solve + build


def paired_table(df: pd.DataFrame, labels: list[str] | None = None) -> pd.DataFrame:
    """Compare every pair of runs at every (beta, n_walls, eps).

    Parameters
    ----------
    df : pandas.DataFrame
        From `results_frame`.
    labels : list of str, optional
        Config labels to compare; None means every run in `df`.

    Returns
    -------
    pandas.DataFrame
        The four paired outcome counts on shared seeds, exact McNemar p, and
        Wilcoxon signed-rank p on planner time over jointly solved seeds
        (NaN with fewer than 5).
    """
    df = supported(df).copy()
    df["planner_time"] = planner_time(df)
    labels = labels or sorted(df["config_label"].unique())
    rows = []
    for (b, w, e), sel in df.groupby(["beta", "n_walls", "eps"]):
        for i, a in enumerate(labels):
            for c in labels[i + 1:]:
                sa = sel[sel["config_label"] == a].set_index("seed")
                sc = sel[sel["config_label"] == c].set_index("seed")
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


def find_frame(path: str | Path) -> pd.DataFrame:
    """Load the find-task records of a results file, one row per (finder run, wall).

    Parameters
    ----------
    path : str or Path
        Results file.

    Returns
    -------
    pandas.DataFrame
        method, config_label, seed, beta, eps, the hidden pose's name and
        validity, attempts, valid attempts, search time, budget, and for
        IK restarts the reported successes and false positives.

    Raises
    ------
    ValueError
        If a (method, config_label) mixes rng schemes.
    """
    rows = []
    for r in load_results(path):
        if r["task"] != "find" or "n_attempts" not in r:          # unsupported, or no problem for the seed
            continue
        d = r.get("detail", {})          # converted records keep the finder's own summary here
        rows.append({"method": r["method"], "config_label": r["config_label"], "rng_scheme": r["rng_scheme"],
                     "seed": r["seed"], "beta": r["beta"], "eps": r["eps"], "sequence": r["sequence"][0],
                     "q_star_valid": r["q_star_valid"], "K": r["planner_config"].get("K"),
                     "n_attempts": r["n_attempts"], "n_valid": r["n_valid"], "search_time": r["search_time"],
                     "time_to_first_valid": r["time_to_first_valid"], "budget": r["budget"],
                     "budget_source": r.get("budget_source"),
                     "reported": r.get("n_reported_success", d.get("n_reported_success")),
                     "false_pos": r.get("n_false_positive", d.get("n_false_positive")),
                     "attempts_per_s": r.get("attempts_per_s", d.get("samples_per_s"))})
    df = pd.DataFrame(rows)
    check_rng_schemes(df)
    return df


def summarize_find(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize find trials per (method, config_label, beta, eps).

    Parameters
    ----------
    df : pandas.DataFrame
        From `find_frame`.

    Returns
    -------
    pandas.DataFrame
        Pooled per-attempt probability p_hat = valid / attempts with its
        Wilson interval (p, lo, hi); its 95% upper bound 3 / attempts after
        zero hits (upper95); P(success within K) = 1 - (1 - p_hat)^K for
        finders with a K option; trials with at least one valid attempt;
        false positives (IK); median search time and budget; median attempts
        per second.
    """
    out = []
    for (m, c, b, e), g in df.groupby(["method", "config_label", "beta", "eps"]):
        k, n = int(g["n_valid"].sum()), int(g["n_attempts"].sum())
        p, lo, hi = wilson(k, n)
        K = g["K"].dropna()
        K = int(K.iloc[0]) if len(K) else None
        out.append({"method": m, "config_label": c, "display": display_label(m, c), "beta": b, "eps": e,
                    "trials": len(g), "K": K, "valid": k, "attempts": n, "p": p, "lo": lo, "hi": hi,
                    "upper95": (3.0 / n) if (n and k == 0) else np.nan,
                    "p_within_K": (1 - (1 - p) ** K) if K else np.nan,
                    "trials_solved": int((g["n_valid"] > 0).sum()),
                    "false_pos": int(g["false_pos"].sum()) if g["false_pos"].notna().any() else np.nan,
                    "time_median": float(g["search_time"].median()),
                    "budget_median": float(g["budget"].median()) if g["budget"].notna().any() else np.nan,
                    "attempts_per_s": float(g["attempts_per_s"].median()) if g["attempts_per_s"].notna().any() else np.nan})
    return pd.DataFrame(out).sort_values(["config_label", "beta", "eps"], ascending=[True, True, False])
