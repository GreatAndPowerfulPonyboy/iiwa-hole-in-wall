"""The clearance guarantee of make_wall, at q_star and within tolerance."""

import numpy as np
from pydrake.all import MultibodyPlant, Parser

from hitw import params
from hitw.kinematics import bare_arm
from hitw.scene import make_scene, min_distance, signed_distance_pairs
from hitw.walls import make_wall, planar_clearance, wall_boxes_polygon, wall_sdf

Q_COMPACT = np.array([0, 0.5, 0, -1.9, 0, 0.65, 0])
Q_REACH = np.array([0, 1.2, 0, -1.0, 0, 0.5, 0])
EPS_TESTED = (0.02, 0.005, 0.002)


def _wall_pairs(scene, q, wall_y, cap=0.5):
    return [d for d, a, b in signed_distance_pairs(scene, q, wall_y, cap) if "wall" in (a, b)]


def test_boxes_are_material():
    for q in (Q_COMPACT, Q_REACH):
        for eps in EPS_TESTED:
            w = make_wall(q, eps)
            mat = wall_boxes_polygon(w)
            assert mat.intersection(w.opening).area < 1e-12
            minx, minz, maxx, maxz = mat.bounds
            assert minx >= -w.width / 2 - 1e-9 and maxx <= w.width / 2 + 1e-9
            assert minz >= w.z0 - 1e-9 and maxz <= w.z0 + w.height + 1e-9
            # the slab minus the opening is covered up to the strip rounding slack
            slab_area = w.width * w.height
            assert mat.area > slab_area - w.opening.area - w.strip * w.opening.length
            print(f"  eps={1e3*eps:5.1f} mm: {w.n_boxes} boxes, opening area {w.opening.area:.4f} m^2")


def test_sampled_arm_points_clear_by_eps():
    rng = np.random.default_rng(0)
    arm = bare_arm()
    for q in (Q_COMPACT, Q_REACH):
        pts = arm.sample_points(q, 400, rng)
        for eps in EPS_TESTED:
            w = make_wall(q, eps)
            c = planar_clearance(w, pts)
            assert c.min() >= eps - 1e-9, (c.min(), eps)
            print(f"  eps={1e3*eps:5.1f} mm: min planar clearance of {len(pts)} arm points = {1e3*c.min():.3f} mm")


def test_realized_clearance_beta0():
    for q in (Q_COMPACT, Q_REACH):
        for eps in EPS_TESTED:
            w = make_wall(q, eps)
            s = make_scene([w], beta=0.0)
            d_wall = min(_wall_pairs(s, q, 0.0))
            d_all = min_distance(s, q, 0.0)
            assert d_wall >= eps - 1e-9, (d_wall, eps)
            assert d_all >= 0
            # the base link pair is visible (the wall is not anchored)
            names = {(a, b) for _, a, b in signed_distance_pairs(s, q, 0.0, 0.5)}
            assert any("iiwa_link_0" in ab and "wall" in ab for ab in names)
            # the tightest point over the whole crossing is in-plane: sweep wall_y
            d_sweep = min(min(_wall_pairs(s, q, y)) for y in np.linspace(-0.25, 0.25, 101))
            assert d_sweep >= eps - 1e-9
            print(f"  eps={1e3*eps:5.1f} mm: realized wall clearance {1e3*d_wall:.3f} mm "
                  f"(sweep min {1e3*d_sweep:.3f} mm), min over all pairs {1e3*d_all:.1f} mm")


def test_realized_clearance_beta_positive():
    rng = np.random.default_rng(17)
    beta = 0.1
    for q_pitch in (Q_COMPACT, Q_REACH):
        q = q_pitch.copy()
        q[list(params.ROLL_JOINTS)] = rng.uniform(-beta, beta, size=4)
        for eps in EPS_TESTED:
            w = make_wall(q, eps)
            s = make_scene([w], beta=beta)
            d_wall = min(_wall_pairs(s, q, 0.0))
            assert d_wall >= eps - 1e-9, (d_wall, eps)
            print(f"  beta={beta} rolls={np.round(q[list(params.ROLL_JOINTS)],3)} eps={1e3*eps:5.1f} mm: "
                  f"realized {1e3*d_wall:.3f} mm, depth {w.arm_depth:.3f} m, {w.n_boxes} boxes")


def test_within_tolerance_keeps_half_eps():
    """Inside the window tolerance eps / (2 rho) the wall clearance stays >= eps / 2."""
    rng = np.random.default_rng(3)
    for beta in (0.0, 0.1):
        q = Q_COMPACT.copy()
        if beta > 0:
            q[list(params.ROLL_JOINTS)] = rng.uniform(-beta, beta, size=4)
        for eps in (0.02, 0.002):
            w = make_wall(q, eps)
            s = make_scene([w], beta=beta)
            tol = params.window_tolerance(eps, s.rho)
            worst = np.inf
            for _ in range(60):
                dq = rng.normal(size=s.num_free)
                dq *= tol / np.linalg.norm(dq)
                q2 = s.q_full(s.q_free(q) + dq)
                for y in (0.0, 0.03, -0.03):
                    worst = min(worst, min(_wall_pairs(s, q2, y)))
            assert worst >= eps / 2 - 1e-9, (worst, eps / 2)
            print(f"  beta={beta} eps={1e3*eps:4.1f} mm tol={tol:.5f} rad: "
                  f"worst clearance at ||dq||=tol is {1e3*worst:.3f} mm (bound {1e3*eps/2:.3f} mm)")


def test_wall_sdf_parses():
    w = make_wall(Q_COMPACT, 0.01)
    plant = MultibodyPlant(0.0)
    Parser(plant).AddModelsFromString(wall_sdf(w), "sdf")
    plant.Finalize()
    assert plant.num_positions() == 1
