"""Stage B: IK restarts and rejection sampling on one wall at 20 mm."""

import numpy as np

from hitw.evaluate import trial_sequence
from hitw.scene import make_scene
from hitw.sequences import LETTER_POSES
from hitw.stage_b import StageBTrial, b1_ik, b2_sampling, is_valid, run_stage_b_trial


def test_validity_matches_success_standard():
    seq = trial_sequence(0, 0.0, 0.02, 1)
    check = make_scene(seq.walls, beta=0.0)
    q_star = seq.crossings[0].q_star
    ok, d = is_valid(check, q_star, 0.0)
    assert ok and abs(d - 0.02) < 1e-3, d                        # q* clears the wall by eps
    other = next(q for n, q in LETTER_POSES.items() if n != seq.crossings[0].name)
    assert not is_valid(check, other, 0.0)[0]                     # another letter does not fit
    assert not is_valid(make_scene(seq.walls, beta=0.1), np.array([0.2, 0, 0, 0, 0, 0, 0]), 0.1)[0]   # band


def test_b1_finds_the_channel_from_a_warm_guess_and_records_restarts():
    seq = trial_sequence(0, 0.0, 0.02, 1)
    check = make_scene(seq.walls, beta=0.0)
    ik = make_scene(seq.walls, beta=0.0, fixed_wall_y=0.0)
    out = b1_ik(ik, check, 0.0, 0.01, np.random.default_rng(0), K=4)
    assert out["K"] == 4 and len(out["restarts"]) == 4 and 0 <= out["p_hat"] <= 1
    for r in out["restarts"]:
        assert r["valid"] == (r["min_distance"] >= 0) if not np.isnan(r["min_distance"]) else True
    print(f"  B1 K=4 at 20 mm: valid {out['n_valid']}, reported {out['n_reported_success']}, "
          f"false positives {out['n_false_positive']}, median restart {out['restart_time_median']:.2f}s")


def test_b2_budget_and_rate_fields():
    seq = trial_sequence(0, 0.0, 0.02, 1)
    check = make_scene(seq.walls, beta=0.0)
    out = b2_sampling(check, 0.0, np.random.default_rng(0), budget_s=1.0, max_samples=10_000)
    assert out["n_samples"] > 100 and out["elapsed"] <= 1.5
    assert (out["hit_rate_upper95"] is None) == (out["n_hits"] > 0)
    print(f"  B2 1 s budget: {out['n_samples']} samples ({out['samples_per_s']:.0f}/s), hits {out['n_hits']}")


def test_run_stage_b_trial_record():
    import json
    r = run_stage_b_trial(StageBTrial(seed=1, beta=0.1, eps=0.02, K=10, b2_budget=0.5))
    assert r["q_star_valid"] and r["b1"]["K"] == 10 and r["b2"]["budget_s"] == 0.5
    assert r["status"] in ("success", "infeasible") and len(r["roll_draw"][0]) == 4
    assert len(r["b1"]["valid_dist_to_q_star"]) == r["b1"]["n_valid"]       # the per-solution bookkeeping
    json.dumps(r)
    print(f"  trial: status {r['status']}, B1 p_hat {r['b1']['p_hat']:.2f}, B2 {r['b2']['n_samples']} samples, {r['total_time']:.1f}s")
