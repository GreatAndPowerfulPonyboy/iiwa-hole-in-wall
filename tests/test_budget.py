"""Services, planner budgets, timing fields, "unsupported", and the kill timer."""

import tempfile
import time
from pathlib import Path

import numpy as np

from hitw.analysis import summarize
from hitw.api import Budget, Services
from hitw.harness import load_results, planner_rng, run_trial, trial_sequence
from hitw.scene import in_collision, make_scene, min_distance
from hitw.sweep import grid, run_sweep

PLANNERS = '''
import time
from hitw.api import PlanOutcome, planner
from hitw.sequences import reference_path


@planner("{prefix}-reference", supports_beta_positive=False)
def reference(seq, svc, budget, rng, cfg):
    ref = reference_path(seq)
    return PlanOutcome(status="success", q_of_t=lambda t: ref.value(t).ravel(), solve_time=1.0,
                       region_build_time=2.0)


@planner("{prefix}-sleeper")
def sleeper(seq, svc, budget, rng, cfg):
    time.sleep(60.0)
    return PlanOutcome(status="infeasible", q_of_t=None, solve_time=60.0)


@planner("{prefix}-nap")
def nap(seq, svc, budget, rng, cfg):
    time.sleep(0.2)
    return PlanOutcome(status="infeasible", q_of_t=None, solve_time=0.2)
'''


def _planners(d: Path, prefix: str) -> Path:
    f = d / f"{prefix}_planners.py"
    f.write_text(PLANNERS.format(prefix=prefix))
    return f


def test_services_build_cache_and_query_the_trial_scenes():
    seq = trial_sequence(0, 0.0, 0.02, 2)
    svc = Services(seq, drake_seed=3)
    assert svc.drake_seed == 3 and svc.scene.wall_joint is not None
    w0 = svc.swept_window_scene(0)
    assert w0 is svc.swept_window_scene(0) and w0 is not svc.swept_window_scene(1)
    c, w = seq.crossings[0], seq.walls[0]
    assert abs(w0.walls[0].thickness - (w.thickness + 2 * seq.speed * c.delta)) < 1e-12
    assert w0.fixed_wall_y == 0.0 and w0.plant.num_positions() == 3
    absent = svc.wall_absent_scene()
    assert absent is svc.wall_absent_scene() and absent.walls == [] and absent.plant.num_positions() == 3
    welded = svc.welded_scene(0.0)
    assert welded is svc.welded_scene(0.0) and len(welded.walls) == 2 and welded.fixed_wall_y == 0.0
    assert svc.checker(absent) is svc.checker(absent)
    try:
        svc.checker(svc.scene)
        raise AssertionError("a checker needs a welded scene")
    except ValueError:
        pass
    y = seq.wall_y_of_t()
    rng = np.random.default_rng(0)
    lo, hi = svc.scene.free_limits()
    for t in (0.0, c.t, c.t + 0.05, seq.crossings[1].t):
        for q in [c.q_star, seq.q0] + [svc.scene.q_full(rng.uniform(lo, hi)) for _ in range(3)]:
            assert svc.in_collision(q, t) == in_collision(svc.scene, q, y(t))
            assert svc.min_distance(q, t, cap=0.1) == min_distance(svc.scene, q, y(t), cap=0.1)


def test_budget_stops_rrt_iris_and_kto():
    from tests.test_plan_rrt import _pillar_scene
    from planners.rrt_connect import _Segment
    from hitw.sequences import LETTER_POSES
    scene = _pillar_scene()
    seg = _Segment(scene, lambda t: 0.0, scene.velocity_limits(), 0.01, np.random.default_rng(0))
    qa, qb = scene.q_free(LETTER_POSES["I"]), scene.q_free(LETTER_POSES["L"])
    knots, n, status = seg.plan(qa, 0.0, qb, 6.0, 2000, deadline=time.perf_counter())
    assert (knots, n, status) == (None, 0, "timeout")

    from planners.gcs import IrisGcsConfig, plan_iris_gcs
    seq = trial_sequence(0, 0.0, 0.02, 2)
    svc = Services(seq, drake_seed=0)
    out = plan_iris_gcs(seq, svc, Budget(0.0), planner_rng(0), IrisGcsConfig(iris="np2"))
    assert out.status == "timeout" and out.extras["solution_result"] == "budget_expired"
    assert len(out.region_volume["transit_build"]) == 1 and out.region_build_time > 0   # stopped after one region

    from planners.kto import KtoConfig, plan_kto
    out = plan_kto(seq, svc, Budget(0.0), planner_rng(0), KtoConfig())
    assert out.status == "timeout" and out.extras["attempts"] == 0 and out.q_of_t is None


def test_timing_fields_and_unsupported():
    with tempfile.TemporaryDirectory() as d:
        f = _planners(Path(d), "timing")
        r = run_trial(f"{f}:reference", 0, 0.0, 0.02, 1, recheck_dt=5e-3, cap=100.0)
        assert r["status"] == "success", r["failure_label"]
        assert r["time_to_success"] == 3.0                                        # solve + region build
        assert r["budget"] == 100.0 and not r["budget_overrun"] and r["startup_time"] > 0
        assert 0 <= r["planner_wall_clock"] < 1.0
        u = run_trial(f"{f}:reference", 0, 0.1, 0.02, 1)
        assert u["status"] == "unsupported" and u["failure_label"] is None and "sequence" not in u
        late = run_trial(f"{f}:nap", 0, 0.0, 0.02, 1, cap=0.05)
        assert late["budget_overrun"] and late["planner_wall_clock"] >= 0.2
        p = Path(d) / "r.jsonl"
        from hitw.harness import append_result
        for rec in (r, u):
            append_result(p, rec)
        from hitw.analysis import results_frame
        s = summarize(results_frame(p))
        assert set(s["beta"]) == {0.0, 0.1}
        row = s[s["beta"] == 0.1].iloc[0]
        assert row["n"] == 0 and row["n_unsupported"] == 1 and np.isnan(row["p"])
        assert s[s["beta"] == 0.0].iloc[0]["n"] == 1


def test_kill_timer_records_the_budget_clock_not_the_cap():
    with tempfile.TemporaryDirectory() as d:
        f = _planners(Path(d), "kill")
        p = Path(d) / "r.jsonl"
        t0 = time.perf_counter()
        stats = run_sweep(grid([f"{f}:sleeper"], [0], [0.0], [0.02], [1]), p, cap=0.5, startup_grace=4.0,
                          workers=1, verbose=False)
        wall = time.perf_counter() - t0
        (r,) = load_results(p)
        assert stats.killed == 1 and r["killed"] and r["status"] == "timeout" and r["kill_timer"] == 4.5
        assert r["startup_time"] > 0 and 0 < r["solve_time"] < 4.5 and r["budget_overrun"]
        assert abs(r["startup_time"] + r["solve_time"] - 4.5) < 1.0 and wall < 10
        q = Path(d) / "q.jsonl"
        run_sweep(grid(["kto"], [0], [0.0], [0.02], [1]), q, cap=0.01, startup_grace=0.2, workers=1, verbose=False)
        (k,) = load_results(q)
        assert k["killed"] and k["solve_time"] is None and k["startup_time"] is None and not k["budget_overrun"]
