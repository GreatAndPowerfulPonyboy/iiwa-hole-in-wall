"""kto on a 1-wall sequence at beta = 0, plus the dense re-check."""

import numpy as np

from hitw.evaluate import classify, recheck
from planners.kto import greville, plan_trajopt, plant_q_of_t
from hitw.scene import make_scene
from hitw.sequences import LETTER_POSES, make_sequence


def test_kto_one_wall_beta0_passes_recheck():
    seq = make_sequence([LETTER_POSES["L"]], beta=0.0, eps=0.02)
    scene = make_scene(seq.walls, beta=0.0)
    res = plan_trajopt(seq, scene, np.random.default_rng(0))
    print(f"  status={res.status} info={res.solver_info} solve_time={res.solve_time:.2f}s evals={res.n_evals}")
    assert res.status == "success", res
    assert res.n_evals > 0 and res.extras["evals_per_attempt"] == [res.n_evals]
    traj = res.traj
    # the wall row reproduces wall_y(t) exactly, and the duration is T
    assert abs(traj.end_time() - seq.T) < 1e-9
    y = seq.wall_y_of_t()
    wi = scene.wall_joint.position_start()
    for t in np.linspace(0, seq.T, 41):
        assert abs(traj.value(t).ravel()[wi] - y(t)) < 1e-9
    q_of_t = plant_q_of_t(traj, scene)
    assert np.allclose(q_of_t(0.0), seq.q0, atol=1e-6)
    rc = recheck(q_of_t, seq, scene, dt=1e-3)
    print(f"  recheck: min d = {1e3*rc.min_signed_distance:.2f} mm at t = {rc.t_min:.3f}s, "
          f"window excess {rc.window_excess:+.2e} rad, peak v {rc.peak_joint_vel:.2f} rad/s, "
          f"{rc.n_samples} samples in {rc.elapsed:.1f}s")
    assert rc.passed, rc
    assert rc.peak_joint_vel <= scene.velocity_limits().max() * 1.01
    assert classify(res.status, rc) == ("success", None)


def test_greville_reproduces_linear():
    from pydrake.all import BsplineBasis, BsplineTrajectory, KnotVectorType
    b = BsplineBasis(4, 12, KnotVectorType.kClampedUniform, 0.0, 1.0)
    g = greville(b)
    tr = BsplineTrajectory(b, np.asfortranarray((2 * g - 0.5).reshape(1, -1)))
    for s in np.linspace(0, 1, 17):
        assert abs(tr.value(s)[0, 0] - (2 * s - 0.5)) < 1e-12


def test_recheck_catches_collision_and_window_miss():
    seq = make_sequence([LETTER_POSES["L"]], beta=0.0, eps=0.02)
    scene = make_scene(seq.walls, beta=0.0)
    c = seq.crossings[0]
    # staying at q0 = home: upright arm, the L cutout does not fit -> collides and misses the window
    rc = recheck(lambda t: seq.q0, seq, scene, dt=5e-3)
    assert not rc.collision_free and not rc.windows_ok
    assert classify("success", rc) == ("infeasible", "reported_feasible_but_collides")
    # staying at q* exactly: collision-free, window honoured
    rc2 = recheck(lambda t: c.q_star, seq, scene, dt=5e-3)
    assert rc2.passed and rc2.min_signed_distance >= 0
    # a 2-norm deviation just above tol inside the window is a window miss
    dq = np.zeros(7); dq[1] = 1.5 * c.tol
    rc3 = recheck(lambda t: c.q_star + dq, seq, scene, dt=5e-3)
    assert not rc3.windows_ok and rc3.window_excess > 0
    assert classify("success", rc3)[1] in ("collision_free_but_misses_window", "reported_feasible_but_collides")
    assert classify("iteration_limit", None) == ("iteration_limit", "timeout")
    assert classify("infeasible", None) == ("infeasible", "infeasible_reported")
