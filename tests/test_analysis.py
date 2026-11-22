"""Wilson intervals, McNemar, and the summaries on synthetic results files."""

import tempfile
from pathlib import Path

import numpy as np

from hitw.analysis import (display_label, find_frame, mcnemar_exact, paired_table, results_frame, summarize,
                           summarize_find, wilson)
from hitw.harness import append_result


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


def _fake(method, seed, eps, ok, beta=0.0, n_walls=1, label=None, scheme="v2"):
    return {"task": "connect", "method": method, "config_label": label or method, "planner_config": {},
            "rng_scheme": scheme, "seed": seed, "beta": beta, "eps": eps, "n_walls": n_walls,
            "status": "success" if ok else "infeasible",
            "failure_label": None if ok else "reported_feasible_but_collides",
            "solve_time": 1.0 + seed * 0.1, "n_evals": 10, "min_signed_distance": 0.001 if ok else -0.002}


def test_summary_and_paired_table():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.jsonl"
        for eps in (0.02, 0.002):
            for seed in range(10):
                append_result(p, _fake("kto", seed, eps, ok=(eps > 0.01 or seed < 3)))
                append_result(p, _fake("gcs-box", seed, eps, ok=(seed < 6)))
                append_result(p, _fake("iris-gcs", seed, eps, ok=True, label="iris-gcs[iris=np2]"))
        df = results_frame(p)
        s = summarize(df)
        row = s[(s["config_label"] == "kto") & (s["eps"] == 0.002)].iloc[0]
        assert row["n"] == 10 and row["k"] == 3 and abs(row["p"] - 0.3) < 1e-12
        assert row["fail_reported_feasible_but_collides"] == 7
        assert s[s["config_label"] == "iris-gcs[iris=np2]"].iloc[0]["display"] == "IRIS + GCS [iris=np2]"
        pt = paired_table(df)
        po = pt[(pt["A"] == "gcs-box") & (pt["B"] == "kto") & (pt["eps"] == 0.002)].iloc[0]
        assert (po["both"], po["only_A"], po["only_B"], po["neither"]) == (3, 3, 0, 4)


def test_display_labels():
    assert display_label("kto", "kto") == "KTO"
    assert display_label("iris-gcs", "iris-gcs[margin=default]") == "IRIS + GCS [margin=default]"
    assert display_label("my-planner", "my-planner[x=1]") == "my-planner [x=1]"      # not registered here


def test_mixed_rng_schemes_are_refused():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.jsonl"
        append_result(p, _fake("kto", 0, 0.02, ok=True, scheme="v1"))
        append_result(p, _fake("kto", 1, 0.02, ok=True, scheme="v2"))
        append_result(p, _fake("gcs-box", 0, 0.02, ok=True, scheme="v1"))
        try:
            results_frame(p)
            raise AssertionError("mixed schemes must be refused")
        except ValueError as e:
            assert "kto" in str(e) and "gcs-box" not in str(e)


def _fake_find(method, seed, eps, n, k, beta=0.0, K=None):
    detail = {"n_reported_success": k, "n_false_positive": 0} if method == "ik-restarts" else {"samples_per_s": 1000.0}
    return {"task": "find", "method": method, "config_label": method, "rng_scheme": "v1",
            "planner_config": {"K": K} if K else {}, "seed": seed, "beta": beta, "eps": eps, "n_walls": 1,
            "sequence": ["L"], "q_star_valid": True, "n_attempts": n, "n_valid": k, "search_time": 1.0,
            "time_to_first_valid": None, "budget": None if K else 1.0, "detail": detail}


def test_find_summary():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.jsonl"
        for eps in (0.02, 0.002):
            for seed in range(4):
                append_result(p, _fake_find("ik-restarts", seed, eps, 10, 1 if seed < 2 else 0, K=10))
                append_result(p, _fake_find("uniform-sampling", seed, eps, 1000, 0))
        s = summarize_find(find_frame(p))
        ik = s[(s["method"] == "ik-restarts") & (s["eps"] == 0.02)].iloc[0]
        assert (ik["valid"], ik["attempts"], ik["trials_solved"]) == (2, 40, 2)
        assert abs(ik["p_within_K"] - (1 - (1 - 0.05) ** 10)) < 1e-12
        us = s[(s["method"] == "uniform-sampling") & (s["eps"] == 0.02)].iloc[0]
        assert us["valid"] == 0 and abs(us["upper95"] - 3 / 4000) < 1e-15 and us["budget_median"] == 1.0
