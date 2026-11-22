"""run_trial records, results.jsonl round trip, and the sweep driver's kill timer."""

import json
import tempfile
from pathlib import Path

import numpy as np

from hitw.evaluate import FAILURE_LABELS, STATUSES
from hitw.harness import (append_result, done_keys, load_results, run_trial,
                           trial_key, trial_sequence)
from hitw.sweep import Trial, grid, run_sweep

REQUIRED = ["task", "planner_config", "planner_source", "rng_scheme", "method", "seed", "beta", "eps", "n_walls", "roll_draw", "drake_version", "status",
            "failure_label", "solve_time", "region_build_time", "region_volume", "iterations",
            "min_signed_distance", "path_length", "peak_joint_vel", "time_to_success", "n_evals"]


def test_trial_sequence_shared_across_eps_and_methods():
    a = trial_sequence(3, 0.1, 0.02, 2)
    b = trial_sequence(3, 0.1, 0.002, 2)
    assert [c.name for c in a.crossings] == [c.name for c in b.crossings]
    assert np.allclose(a.q_stars(), b.q_stars())
    c = trial_sequence(4, 0.1, 0.02, 2)
    assert not np.allclose(a.q_stars(), c.q_stars())


def test_run_trial_record():
    r = run_trial("kto", seed=0, beta=0.0, eps=0.02, n_walls=1, recheck_dt=5e-3)
    for k in REQUIRED:
        assert k in r, k
    assert r["status"] in STATUSES and r["failure_label"] in FAILURE_LABELS
    assert r["status"] == "success" and r["failure_label"] is None
    assert r["time_to_success"] == r["solve_time"] and r["n_evals"] > 0          # no region build
    assert r["budget"] is None and r["startup_time"] > 0 and not r["budget_overrun"]
    assert r["min_signed_distance"] >= 0 and r["drake_version"] == "1.58.0"
    assert (r["task"], r["method"], r["config_label"], r["rng_scheme"]) == ("connect", "kto", "kto", "v2")
    assert len(r["roll_draw"]) == 1 and len(r["roll_draw"][0]) == 4 and all(v == 0 for v in r["roll_draw"][0])
    json.dumps(r)   # JSON-ready
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.jsonl"
        append_result(p, r)
        append_result(p, dict(r, seed=1))
        rs = load_results(p)
        assert len(rs) == 2 and trial_key(rs[0]) == ("kto", "kto", 0, 0.0, 0.02, 1)
        assert done_keys(p) == {("kto", "kto", 0, 0.0, 0.02, 1), ("kto", "kto", 1, 0.0, 0.02, 1)}
        old = {k: v for k, v in r.items() if k not in ("task", "config_label", "planner_config")}
        append_result(p, dict(old, method="KTO", seed=2))                # the earlier record format
        try:
            load_results(p)
            raise AssertionError("an earlier-format record must be refused")
        except ValueError as e:
            assert "migrate_results.py" in str(e) and ":3:" in str(e)


def test_sweep_kill_timer_and_resume():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.jsonl"
        trials = grid(["kto"], [0], [0.0], [0.02], [1])
        assert trials[0].method == "kto" and trials[0].config_label == "kto"
        stats = run_sweep(trials, p, cap=0.01, startup_grace=0.3, workers=1, verbose=False)
        rs = load_results(p)
        assert stats.killed == 1 and len(rs) == 1
        assert rs[0]["status"] == "timeout" and rs[0]["failure_label"] == "timeout" and rs[0]["killed"]
        assert rs[0]["time_to_success"] is None and rs[0]["config_label"] == "kto" and abs(rs[0]["kill_timer"] - 0.31) < 1e-12
        stats2 = run_sweep(trials, p, cap=0.5, workers=1, verbose=False)
        assert stats2.skipped == 1 and stats2.run == 0


def test_sweep_runs_trials_in_subprocesses():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.jsonl"
        trials = grid(["gcs-box", "gcs-box[max_rounded_paths=3]"], [0], [0.0], [0.02], [1])
        stats = run_sweep(trials, p, cap=120.0, workers=2, verbose=False)
        rs = load_results(p)
        assert stats.run == 2 and stats.killed == 0 and len(rs) == 2
        assert {r["config_label"] for r in rs} == {"gcs-box", "gcs-box[max_rounded_paths=3]"}
        assert all(r["status"] == "success" and not r["killed"] for r in rs)
