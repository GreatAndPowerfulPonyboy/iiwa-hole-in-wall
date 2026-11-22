"""The find task: what a finder sees, harness-side validation, and the two finders."""

import dataclasses
import json
import tempfile
from pathlib import Path

import numpy as np

from hitw.api import Budget, Services, WallView
from hitw.find import is_valid, validate
from hitw.harness import find_trial, load_results, trial_sequence
from hitw.scene import make_scene
from hitw.sequences import LETTER_POSES
from hitw.sweep import grid, run_sweep

BAD = '''
from hitw.api import planner
@planner("{id}", task="find")
def bad(wall, svc, budget, rng, cfg):
    yield [0.0, 0.0, 0.0]
'''


def test_validity_matches_success_standard():
    seq = trial_sequence(0, 0.0, 0.02, 1)
    check = make_scene(seq.walls, beta=0.0)
    q_star = seq.crossings[0].q_star
    ok, d = is_valid(check, q_star, 0.0)
    assert ok and abs(d - 0.02) < 1e-3, d                        # q* clears the wall by eps
    other = next(q for n, q in LETTER_POSES.items() if n != seq.crossings[0].name)
    assert not is_valid(check, other, 0.0)[0]                     # another letter does not fit
    assert not is_valid(make_scene(seq.walls, beta=0.1), np.array([0.2, 0, 0, 0, 0, 0, 0]), 0.1)[0]   # band


def test_validate_gives_the_same_verdict_as_is_valid():
    for beta in (0.0, 0.1):
        seq = trial_sequence(1, beta, 0.02, 1)
        check = make_scene(seq.walls, beta=beta)
        rng = np.random.default_rng(0)
        lo, hi = check.free_limits()
        qs = [seq.crossings[0].q_star] + [check.q_full(rng.uniform(lo, hi)) for _ in range(300)]
        qs += [seq.crossings[0].q_star + rng.normal(scale=2e-3, size=7) * (beta > 0 or np.isin(range(7), [1, 3, 5]))
               for _ in range(100)]
        for q in qs:
            v, d = validate(check, q, beta)
            w, e = is_valid(check, q, beta)
            assert v == w and (d is None or d == e)


def test_the_finder_cannot_see_the_hidden_configuration():
    seq = trial_sequence(0, 0.1, 0.002, 1)
    wall = seq.walls[0]
    view = WallView.of(wall, 0.1, 0.002, seq.speed)
    assert {f.name for f in dataclasses.fields(view)} == {"boxes", "thickness", "width", "z0", "height", "t_cross",
                                                          "speed", "beta", "eps"}
    svc = Services.for_wall(view, drake_seed=0)
    assert svc.seq is None
    for scene in (svc.scene, svc.welded_scene(0.0)):
        spec = scene.walls[0]
        assert np.isnan(spec.q_star).all() and spec.silhouette is None and spec.opening is None
        assert np.isnan(spec.arm_depth) and spec.boxes == wall.boxes
    try:
        svc.swept_window_scene(0)
        raise AssertionError("no crossings in a find trial")
    except ValueError:
        pass
    q = seq.crossings[0].q_star                                   # same geometry as the real wall
    real = make_scene([wall], beta=0.1)
    from hitw.scene import min_distance
    assert svc.min_distance(q, view.t_cross, cap=0.1) == min_distance(real, q, 0.0, cap=0.1)


def test_ik_restarts_record():
    r = find_trial("ik-restarts[K=4]", seed=0, beta=0.0, eps=0.02)
    assert (r["task"], r["method"], r["config_label"], r["rng_scheme"]) == ("find", "ik-restarts", "ik-restarts[K=4]", "v2")
    assert r["n_attempts"] == 4 and len(r["attempts"]) == 4 and r["stopped"] == "exhausted" and r["q_star_valid"]
    assert r["n_valid"] == sum(a["valid"] for a in r["attempts"]) == len(r["valid_dist_to_q_star"])
    assert r["n_false_positive"] == sum(a["reported_success"] and not a["valid"] for a in r["attempts"])
    assert all(set(a) >= {"k", "info", "reported_success", "solve_time", "valid", "min_distance", "q"} for a in r["attempts"])
    assert r["budget"] is None and r["budget_source"] == "unlimited" and r["startup_time"] > 0
    json.dumps(r)
    stopped = find_trial("ik-restarts", seed=0, beta=0.0, eps=0.02, cap=0.0)
    assert stopped["n_attempts"] == 0 and stopped["stopped"] == "budget" and stopped["status"] == "infeasible"


def test_uniform_sampling_under_a_budget():
    r = find_trial("uniform-sampling", seed=0, beta=0.0, eps=0.02, cap=0.5)
    assert r["n_attempts"] > 100 and r["stopped"] == "budget" and not r["budget_overrun"]
    assert 0.5 <= r["search_time"] < 0.6 and "attempts" not in r
    assert (r["first_valid_attempt"] is None) == (r["n_valid"] == 0)
    capped = find_trial("uniform-sampling[max_samples=50]", seed=0, beta=0.0, eps=0.02, cap=5.0)
    assert capped["n_attempts"] == 50 and capped["stopped"] == "exhausted"


def test_a_candidate_of_the_wrong_shape_is_an_error():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "bad.py"
        f.write_text(BAD.format(id="bad-finder"))
        try:
            find_trial(f"{f}:bad", seed=0, beta=0.0, eps=0.02)
            raise AssertionError("a 3-vector candidate must be refused")
        except TypeError as e:
            assert "(7,)" in str(e)


def test_equal_compute_sweep():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.jsonl"
        trials = grid(["ik-restarts[K=2]", "uniform-sampling"], [1], [0.0], [0.02], [1],
                      budget_from={"uniform-sampling": "ik-restarts[K=2]"})
        us_trial = next(t for t in trials if t.method == "uniform-sampling")
        assert us_trial.budget_from == "ik-restarts[K=2]"
        stats = run_sweep(trials, p, cap=120.0, workers=2, verbose=False)
        ik, us = sorted(load_results(p), key=lambda r: r["method"])
        assert stats.run == 2 and us["budget"] == ik["search_time"]
        assert us["budget_source"] == "equal:ik-restarts[K=2]" and ik["budget_source"] == "cap"
        assert ik["planner_source"]["path"] == "planners/ik_restarts.py"
        try:
            grid(["uniform-sampling"], [1], [0.0], [0.02], [1], budget_from={"uniform-sampling": "ik-restarts"})
            raise AssertionError("the reference must be in the grid")
        except ValueError as e:
            assert "find runs" in str(e)
        q = Path(d) / "q.jsonl"
        lone = grid(["ik-restarts[K=2]", "uniform-sampling"], [1], [0.0], [0.02], [1],
                    budget_from={"uniform-sampling": "ik-restarts[K=2]"})
        run_sweep([t for t in lone if t.method == "uniform-sampling"] + [], q, verbose=False)   # reference absent
        assert load_results(q) == []
