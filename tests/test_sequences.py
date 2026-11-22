"""Windows, tolerances, roll draws, the reference path and the transit test."""

import numpy as np

from hitw import params
from hitw.scene import make_scene
from hitw.sequences import LETTER_POSES, make_sequence, random_sequence, reference_path, toppra_time, transit_feasible

Q_L = LETTER_POSES["L"]


def test_make_sequence_windows_and_rolls():
    seq = make_sequence([Q_L, LETTER_POSES["I"]], beta=0.0, eps=0.01)
    assert seq.n_walls == 2 and seq.T == params.T_FIRST + params.WINDOW_SPACING + params.T_TAIL
    c0, c1 = seq.crossings
    assert c0.t == params.T_FIRST and c1.t == params.T_FIRST + params.WINDOW_SPACING
    assert abs(c0.delta - (params.WALL_THICKNESS + seq.walls[0].arm_depth) / (2 * params.WALL_SPEED)) < 1e-12
    assert abs(c0.tol - 0.01 / (2 * seq.rho)) < 1e-12
    assert np.all(c0.roll_draw == 0)
    y = seq.wall_y_of_t()
    assert abs(y(c0.t)) < 1e-12 and abs(y(c1.t) + seq.walls[1].y_offset) < 1e-12   # slab 1 centred at t_1
    rng = np.random.default_rng(5)
    seq_b = make_sequence([Q_L], beta=0.1, eps=0.01, rng=rng)
    rolls = seq_b.crossings[0].roll_draw
    assert rolls.shape == (4,) and np.all(np.abs(rolls) <= 0.1) and np.any(rolls != 0)
    assert np.allclose(seq_b.crossings[0].q_star[list(params.ROLL_JOINTS)], rolls)
    assert seq_b.rho > seq.rho
    try:
        make_sequence([Q_L], beta=0.1, eps=0.01)
        assert False
    except ValueError:
        pass


def test_reference_path_holds_windows():
    seq = make_sequence([Q_L, LETTER_POSES["Z"]], beta=0.0, eps=0.005)
    ref = reference_path(seq)
    assert np.allclose(ref.value(0.0).ravel(), seq.q0)
    for c in seq.crossings:
        for t in np.linspace(c.t - c.delta, c.t + c.delta, 7):
            assert np.allclose(ref.value(t).ravel(), c.q_star, atol=1e-12)
    assert np.allclose(ref.value(seq.T).ravel(), seq.crossings[-1].q_star)


def test_toppra_time_and_transit():
    t = toppra_time(np.zeros(7), Q_L)
    assert t is not None and 1.0 < t < 3.0, t
    assert toppra_time(Q_L, Q_L) is not None
    seq = make_sequence([Q_L, LETTER_POSES["SEVEN"]], beta=0.0, eps=0.01)
    scene = make_scene(seq.walls, beta=0.0)
    ok, info = transit_feasible(seq, scene)
    assert ok, info
    assert len(info["legs"]) == 2 and all(l["toppra_time"] < l["available"] for l in info["legs"])
    tight = make_sequence([Q_L, LETTER_POSES["SEVEN"]], beta=0.0, eps=0.01, spacing=0.8)
    ok2, info2 = transit_feasible(tight, scene)
    assert not ok2, info2


def test_random_sequence_reproducible():
    scene = make_scene([], beta=0.1)
    a = random_sequence(2, np.random.default_rng(3), 0.1, 0.01, scene)
    b = random_sequence(2, np.random.default_rng(3), 0.1, 0.01, scene)
    assert a is not None and b is not None
    assert [c.name for c in a.crossings] == [c.name for c in b.crossings]
    assert np.allclose(a.q_stars(), b.q_stars())
    assert a.crossings[0].name != a.crossings[1].name
