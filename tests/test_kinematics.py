"""Joint radii, motion bound rho, arm depth."""

import numpy as np

from hitw import params
from hitw.kinematics import arm_depth, bare_arm, joint_radii, motion_bound

M0_RADII = np.array([0.965, 0.965, 0.566, 0.566, 0.166, 0.166, 0.073])


def test_joint_radii_match_milestone0():
    r = joint_radii()
    assert np.allclose(r, M0_RADII, atol=1e-3), r
    assert abs(motion_bound(r) - 1.602) < 1e-3
    assert motion_bound(r, params.PITCH_JOINTS) < motion_bound(r)


def test_motion_bound_dominates_sampled_point_travel():
    """No corner of any arm box moves farther than rho ||dq||_2, nor farther
    than sum_j r_j |dq_j|, over random pairs of configurations."""
    arm = bare_arm()
    r = joint_radii()
    rho = motion_bound(r)
    lo, hi = arm.plant.GetPositionLowerLimits(), arm.plant.GetPositionUpperLimits()
    rng = np.random.default_rng(0)
    worst = 0.0
    for scale in (1.0, 0.1, 0.01):
        for _ in range(150):
            q = rng.uniform(lo, hi)
            dq = rng.normal(size=7)
            dq *= scale * rng.uniform(0.2, 1.0) / np.linalg.norm(dq)
            q2 = np.clip(q + dq, lo, hi)
            dq = q2 - q
            travel = np.linalg.norm(arm.all_corners(q2) - arm.all_corners(q), axis=1).max()
            per_joint = float(r @ np.abs(dq))
            l2 = rho * np.linalg.norm(dq)
            assert travel <= per_joint + 1e-9, (travel, per_joint, dq)
            assert travel <= l2 + 1e-9, (travel, l2, dq)
            worst = max(worst, travel / l2)
    print(f"  tightest ratio travel / (rho ||dq||): {worst:.3f}")


def test_arm_depth():
    d0 = arm_depth(np.zeros(7))
    assert abs(d0 - 2 * 0.1146) < 1e-3, d0
    assert abs(arm_depth([0, 0.5, 0, -1.9, 0, 0.65, 0]) - d0) < 1e-9   # planar: same depth
    assert arm_depth([0.1, 0.5, 0.1, -1.9, 0.1, 0.65, 0.1]) > d0 + 0.05
