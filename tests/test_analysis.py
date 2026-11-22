"""Wilson intervals, McNemar, and the summary/plot on a synthetic results file."""

import tempfile
from pathlib import Path

import numpy as np

from hitw.analysis import mcnemar_exact, paired_outcomes, plot_success_vs_eps, results_frame, summarize, wilson
from hitw.evaluate import append_result


def test_wilson_known_values():
    p, lo, hi = wilson(5, 10)
    assert abs(p - 0.5) < 1e-12 and abs(lo - 0.2366) < 1e-3 and abs(hi - 0.7634) < 1e-3
    p, lo, hi = wilson(0, 10)
    assert p == 0 and lo == 0 and abs(hi - 0.2775) < 1e-3
    p, lo, hi = wilson(100, 100)
    assert abs(hi - 1.0) < 1e-9 and abs(lo - 0.9630) < 1e-3
    p, lo, hi = wilson(50, 100)
    assert abs(hi - lo - 0.192) < 1e-3        # ~10-point half-width at p = 0.5, N = 100


def test_mcnemar():
    assert mcnemar_exact(10, 0, 0, 5) == 1.0
    assert abs(mcnemar_exact(10, 8, 0, 2) - 2 * 0.5 ** 8) < 1e-12
    assert mcnemar_exact(0, 3, 3, 0) == 1.0


def _fake(method, seed, eps, ok, beta=0.0, n_walls=1):
    return {"method": method, "seed": seed, "beta": beta, "eps": eps, "n_walls": n_walls,
            "status": "success" if ok else "infeasible",
            "failure_label": None if ok else "reported_feasible_but_collides",
            "solve_time": 1.0 + seed * 0.1, "n_evals": 10, "min_signed_distance": 0.001 if ok else -0.002}


def test_summary_and_plot():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.jsonl"
        for eps in (0.02, 0.002):
            for seed in range(10):
                append_result(p, _fake("KTO", seed, eps, ok=(eps > 0.01 or seed < 3)))
                append_result(p, _fake("GCS", seed, eps, ok=(seed < 6)))
        df = results_frame(p)
        s = summarize(df)
        row = s[(s["method"] == "KTO") & (s["eps"] == 0.002)].iloc[0]
        assert row["n"] == 10 and row["k"] == 3 and abs(row["p"] - 0.3) < 1e-12
        assert row["fail_reported_feasible_but_collides"] == 7
        po = paired_outcomes(df, "KTO", "GCS", 0.0, 1, 0.002)
        assert (po["both"], po["only_a"], po["only_b"], po["neither"]) == (3, 0, 3, 4)
        out = plot_success_vs_eps(df, Path(d) / "plots" / "success.png")
        assert out.exists() and out.stat().st_size > 10_000
