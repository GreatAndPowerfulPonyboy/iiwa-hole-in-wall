"""gcs-box and iris-gcs on a 1-wall sequence at beta = 0."""

import numpy as np

from hitw.evaluate import classify, recheck
from hitw.api import Services
from planners.gcs import (IrisGcsConfig, gcs_q_of_t, gcs_wall_error, grow_region, region_volume, solve_gcs_box,
                           solve_iris_gcs, y_schedule)
from hitw.scene import make_scene
from hitw.sequences import LETTER_POSES, make_sequence


def _seq(eps=0.02, poses=("L",), beta=0.0, rng=None):
    return make_sequence([LETTER_POSES[p] for p in poses], beta=beta, eps=eps, rng=rng, names=list(poses))


def test_y_schedule_is_contiguous():
    seq = _seq(poses=("L", "SLASH"))
    s = y_schedule(seq)
    assert s["transits"][0][0] == s["y0"] and s["transits"][0][1] == s["windows"][0][0]
    assert s["transits"][1] == (s["windows"][0][1], s["windows"][1][0])
    assert all(a < b for a, b in s["windows"] + s["transits"])


def test_gcs_box_chain_one_wall():
    seq = _seq()
    scene = make_scene(seq.walls, beta=0.0)
    res = solve_gcs_box(seq, scene, seed=0)
    print(f"  gcs-box: {res.status} {res.solver_name} {res.solution_result} solve={res.solve_time:.2f}s")
    assert res.status == "success", res
    assert gcs_wall_error(res.traj, seq, scene) < 1e-4
    rc = recheck(gcs_q_of_t(res.traj, scene, seq), seq, scene, dt=2e-3)
    print(f"  recheck min d={1e3*rc.min_signed_distance:.2f} mm, window excess {rc.window_excess:+.1e}, peak v {rc.peak_joint_vel:.2f}")
    assert rc.collision_free and rc.window_excess < 1e-6
    assert rc.peak_joint_vel <= scene.velocity_limits().max() * 1.05


def test_iris_regions_and_volumes():
    seq = _seq()
    scene = make_scene(seq.walls, beta=0.0)
    svc = Services(seq, scene, drake_seed=0)
    w_scene = svc.swept_window_scene(0)
    assert svc.swept_window_scene(0) is w_scene
    assert w_scene.plant.num_positions() == 3
    assert abs(w_scene.walls[0].thickness - (seq.walls[0].thickness + 2 * seq.speed * seq.crossings[0].delta)) < 1e-12
    q_star = scene.q_free(seq.crossings[0].q_star)
    region, dt = grow_region(svc.checker(w_scene), w_scene, q_star, variant="zo", seed=0, threads=2)
    vol = region_volume(region)
    print(f"  IrisZo window region: {dt:.2f}s, ellipsoid vol {vol['ellipsoid']:.3e}, sampled {vol['sampled']:.3e} "
          f"(acc {vol['sampled_rel_accuracy']:.3f}, n {vol['sampled_n']})")
    assert region.PointInSet(q_star) and vol["ellipsoid"] > 0
    t_scene = svc.wall_absent_scene()
    q0 = scene.q_free(seq.q0)
    region_t, dt_t = grow_region(svc.checker(t_scene), t_scene, (q0 + q_star) / 2, variant="zo", seed=0,
                                 threads=2, containment=np.array([q0, q_star]))
    assert region_t.PointInSet(q0) and region_t.PointInSet(q_star)
    assert region_volume(region_t)["ellipsoid"] > vol["ellipsoid"]


def test_iris_gcs_one_wall():
    seq = _seq()
    scene = make_scene(seq.walls, beta=0.0)
    res = solve_iris_gcs(seq, Services(seq, scene, drake_seed=0), cfg=IrisGcsConfig(iris="zo", threads=2))
    print(f"  iris-gcs: {res.status} build={res.region_build_time:.2f}s solve={res.solve_time:.2f}s "
          f"window vol {res.region_volume['windows'][0]['ellipsoid']:.2e}")
    assert res.status == "success", res
    rc = recheck(gcs_q_of_t(res.traj, scene, seq), seq, scene, dt=2e-3)
    assert rc.collision_free and rc.window_excess < 1e-6
    assert classify(res.status, rc)[0] in ("success", "infeasible")


def test_window_margin_scales_with_eps():
    from planners.gcs import DEFAULT_MARGIN, window_margin
    for eps in (0.02, 0.005, 0.002):
        seq = _seq(eps=eps)
        m = window_margin(seq, 0, "scaled")
        assert m == min(DEFAULT_MARGIN, 0.5 * seq.crossings[0].tol)
        assert window_margin(seq, 0, "default") == DEFAULT_MARGIN
        assert window_margin(seq, 0, 1e-3) == 1e-3
    assert window_margin(_seq(eps=0.02), 0, "scaled") < DEFAULT_MARGIN     # 20 mm: tol/2 = 4.4e-3 at beta = 0


def test_iris_gcs_at_2mm_scaled_margin_succeeds_and_default_margin_is_reported():
    """At eps = 2 mm the free radius around q* is eps / rho = 1.8e-3 rad, below
    Drake's default 1e-2 margin: IRIS rejects the seed.  With margin = tol / 2
    all three variants grow a region containing the tolerance box."""
    seq = _seq(eps=0.002)
    scene = make_scene(seq.walls, beta=0.0)
    res = solve_iris_gcs(seq, Services(seq, scene, drake_seed=0), cfg=IrisGcsConfig(iris="zo", threads=2, margin="default"))
    assert res.status == "infeasible" and res.solution_result == "iris_seed_failed", res
    assert res.traj is None and res.region_build_time > 0 and "window_1" in res.extras["error"]
    assert res.region_volume["window_margin"] == [0.01] and len(res.region_volume["windows"]) == 0
    print(f"  default margin: {res.solution_result} after {res.region_build_time:.1f}s build")
    for variant in ("zo", "np2"):
        res = solve_iris_gcs(seq, Services(seq, scene, drake_seed=0), cfg=IrisGcsConfig(iris=variant, threads=2, margin="scaled"))
        w = res.region_volume["windows"][0]
        print(f"  {variant} scaled margin {res.region_volume['window_margin'][0]:.2e}: {res.status} build={res.region_build_time:.1f}s "
              f"ellipsoid vol {w['ellipsoid']:.2e} box_in={w['contains_tolerance_box']}")
        assert res.status == "success", res
        assert w["contains_q_star"] and w["contains_tolerance_box"]
        rc = recheck(gcs_q_of_t(res.traj, scene, seq), seq, scene, dt=2e-3)
        assert rc.collision_free and rc.window_excess < 1e-6, rc
