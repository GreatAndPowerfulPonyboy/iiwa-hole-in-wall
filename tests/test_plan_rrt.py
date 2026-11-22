"""M4 (time-augmented RRT-Connect) on 1- and 2-wall sequences."""

import numpy as np

from hitw.evaluate import recheck
from hitw.plan_rrt import STEP, _Segment, _Tree, plan_rrt, rrt_q_of_t
from hitw.scene import make_scene
from hitw.sequences import LETTER_POSES, make_sequence


def test_extend_assigns_max_speed_times_and_respects_velocity():
    seq = make_sequence([LETTER_POSES["L"]], beta=0.0, eps=0.02)
    scene = make_scene(seq.walls, beta=0.0)
    seg = _Segment(scene, lambda t: -5.0, scene.velocity_limits(), 0.01, np.random.default_rng(0))
    seg.t_lo, seg.t_hi = 0.0, 6.0
    fwd = _Tree(np.zeros(3), 0.0, True, seg.vmax)
    status, idx = seg.extend(fwd, np.array([2.0, -2.0, 1.0]))
    assert status == "Advanced" and fwd.ts[idx] > 0
    assert abs(np.linalg.norm(fwd.qs[idx]) - STEP) < 1e-9                        # one step of STEP rad
    assert abs(fwd.ts[idx] - fwd.needed(np.zeros(3), fwd.qs[idx])) < 1e-12       # earliest arrival
    assert seg.velocity_ok(np.zeros(3), 0.0, fwd.qs[idx], fwd.ts[idx])
    bwd = _Tree(np.zeros(3), 6.0, False, seg.vmax)
    status, idx = seg.extend(bwd, np.array([2.0, -2.0, 1.0]))
    assert status == "Advanced" and bwd.ts[idx] < 6.0                             # latest departure
    # a target state that no node can reach in time is Trapped, not truncated
    assert seg.extend(fwd, np.array([2.0, 0.0, 0.0]), 0.1)[0] == "Trapped"
    assert not seg.velocity_ok(np.zeros(3), 0.0, np.array([2.0, 0, 0]), 0.1)


def _pillar_scene():
    """A static pillar in the arm plane that blocks the straight line I -> L
    while both poses stay clear (searched for in milestone 4)."""
    from shapely.geometry import box as shapely_box
    from hitw.walls import WallSpec
    spec = WallSpec(q_star=np.zeros(7), eps=0.0, strip=0.01, thickness=0.05, width=0.08, z0=0.85, height=0.3,
                    boxes=[(0.25, 1.0, 0.08, 0.3)], silhouette=shapely_box(0, 0, 0, 0),
                    opening=shapely_box(0, 0, 0, 0), arm_depth=0.1146)
    return make_scene([spec], beta=0.0)


def test_rrt_tree_search_around_pillar():
    """The trial sequences never block a straight transit (transit_feasible),
    so this is the test that exercises the tree search and the shortcut."""
    from hitw.scene import min_distance
    scene = _pillar_scene()
    qa, qb = scene.q_free(LETTER_POSES["I"]), scene.q_free(LETTER_POSES["L"])
    assert min_distance(scene, LETTER_POSES["I"], 0.0) > 0.02 and min_distance(scene, LETTER_POSES["L"], 0.0) > 0.02
    for seed in range(3):
        seg = _Segment(scene, lambda t: 0.0, scene.velocity_limits(), 0.01, np.random.default_rng(seed))
        knots, n, status = seg.plan(qa, 0.0, qb, 6.0, 2000)
        assert status == "success" and n > 0, (status, n)                         # the direct edge is blocked
        short = seg.shortcut(knots, 200)
        ts = [t for _, t in short]
        assert len(short) <= len(knots) and all(b > a for a, b in zip(ts[:-1], ts[1:]))
        assert all(seg.velocity_ok(qa_, ta_, qb_, tb_) for (qa_, ta_), (qb_, tb_) in zip(short[:-1], short[1:]))
        q_of_t = rrt_q_of_t([(scene.q_full(q), t) for q, t in short], 6.0)
        d = min(min_distance(scene, q_of_t(t), 0.0, cap=0.1) for t in np.arange(0.0, 6.0, 5e-3))
        print(f"  seed {seed}: {n} samples, {len(knots)} -> {len(short)} knots, dense min d {1e3*d:.1f} mm")
        assert d >= 0.0


def test_rrt_one_and_two_walls():
    for poses in (["L"], ["L", "SLASH"]):
        seq = make_sequence([LETTER_POSES[p] for p in poses], beta=0.0, eps=0.02, names=poses)
        scene = make_scene(seq.walls, beta=0.0)
        res = plan_rrt(seq, scene, seed=0)
        print(f"  M4 {poses}: {res.status} samples={res.iterations} checks={res.extras['checks']} "
              f"res={res.extras['edge_resolution']:.4f} solve={res.solve_time:.2f}s segs={res.extras['segments']}")
        assert res.status == "success", res
        ts = [t for _, t in res.knots]
        assert all(b > a for a, b in zip(ts[:-1], ts[1:]))
        rc = recheck(rrt_q_of_t(res.knots, seq.T), seq, scene, dt=2e-3)
        print(f"  recheck min d={1e3*rc.min_signed_distance:.2f} mm, window excess {rc.window_excess:+.1e}, peak v {rc.peak_joint_vel:.2f}")
        assert rc.passed, rc
        assert rc.peak_joint_vel <= scene.velocity_limits().max() * 1.01
