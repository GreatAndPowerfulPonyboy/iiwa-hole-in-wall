"""Scene construction for beta = 0 (welds) and beta > 0 (banded limits)."""

import numpy as np

from hitw import params
from hitw.kinematics import bare_arm
from hitw.scene import in_collision, make_scene, min_distance, set_state, signed_distance_pairs
from hitw.walls import make_wall

Q_STAR = np.array([0, 0.5, 0, -1.9, 0, 0.65, 0])


def test_beta_zero_welds():
    s = make_scene([], beta=0.0)
    assert s.num_free == 3 and s.free_joints == (1, 3, 5)
    assert s.plant.num_positions() == 4            # 3 arm + 1 wall
    assert s.plant.num_positions(s.iiwa) == 3
    q = s.q_full([0.5, -1.9, 0.65])
    assert np.allclose(q, Q_STAR) and np.allclose(s.q_free(q), [0.5, -1.9, 0.65])
    lo, hi = s.free_limits()
    assert np.allclose(hi, [2.094, 2.094, 2.094], atol=1e-3)
    assert abs(s.rho - np.linalg.norm([0.965, 0.566, 0.166])) < 1e-2
    set_state(s, Q_STAR, 0.3)
    assert np.allclose(s.plant.GetPositions(s.plant_context, s.iiwa), [0.5, -1.9, 0.65])
    assert abs(s.wall_joint.get_translation(s.plant_context) - 0.3) < 1e-12
    # frozen joints really are welded: the arm pose matches the bare arm at Q_STAR
    p7 = s.plant.EvalBodyPoseInWorld(s.plant_context, s.plant.GetBodyByName("iiwa_link_7")).translation()
    arm = bare_arm(); arm.set_q(Q_STAR)
    p7_ref = arm.plant.EvalBodyPoseInWorld(arm.plant_context, arm.plant.GetBodyByName("iiwa_link_7")).translation()
    assert np.allclose(p7, p7_ref, atol=1e-12)


def test_beta_positive_limits():
    s = make_scene([], beta=0.1)
    assert s.num_free == 7 and s.plant.num_positions() == 8
    lo, hi = s.free_limits()
    for j in params.ROLL_JOINTS:
        assert lo[j] == -0.1 and hi[j] == 0.1
    for j in params.PITCH_JOINTS:
        assert abs(hi[j] - 2.094) < 1e-3
    try:
        s.check_q([0.2, 0, 0, 0, 0, 0, 0])
        assert False, "check_q should reject a roll beyond beta"
    except ValueError:
        pass


def test_fixed_wall_is_welded():
    w = make_wall(Q_STAR, 0.02)
    s = make_scene([w], beta=0.0, fixed_wall_y=0.0)
    assert s.wall_joint is None and s.plant.num_positions() == 3
    assert min_distance(s, Q_STAR, 0.0) >= 0.0


def test_distance_semantics():
    w = make_wall(Q_STAR, 0.02)
    s = make_scene([w], beta=0.0)
    far = min_distance(s, Q_STAR, 5.0)              # wall far: self-collision only
    pairs = signed_distance_pairs(s, Q_STAR, 5.0)
    assert all("wall" not in a and "wall" not in b for _, a, b in pairs)
    assert far > 0
    near = min_distance(s, Q_STAR, 0.0)
    assert 0 <= near <= far
    # a clearly wrong configuration collides with the wall at the crossing
    q_bad = Q_STAR + np.array([0, 0.4, 0, 0, 0, 0, 0])
    assert min_distance(s, q_bad, 0.0) < 0 and in_collision(s, q_bad, 0.0)
    assert not in_collision(s, Q_STAR, 0.0)
    # boolean and signed queries agree on random configurations
    rng = np.random.default_rng(1)
    lo, hi = s.free_limits()
    agree = 0
    for _ in range(100):
        q = s.q_full(rng.uniform(lo, hi))
        agree += in_collision(s, q, 0.0) == (min_distance(s, q, 0.0) < 0)
    assert agree == 100, agree
