"""Crossing sequences: given crossing configurations q*_i with times t_i, the
walls built from them, the time windows and the joint-space tolerance inside
each window.

Window: the wall overlaps the arm's y-extent for |t - t_i| <= delta_i with
delta_i = (t + d_i) / (2 v), d_i the arm depth at q*_i.  Inside the window
the configuration must stay within tol_i = eps / (2 rho) of q*_i in the
2-norm over the free joints, which keeps the wall clearance >= eps / 2
(kinematics.motion_bound).

For beta > 0 the roll joints of every q*_i are drawn uniformly in
[-beta, beta] from the trial rng and recorded; the wall silhouette is built
from the full q*_i, so the draws shape the cutouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydrake.all import CalcGridPointsOptions, PathParameterizedTrajectory, PiecewisePolynomial, Toppra

from . import params
from .kinematics import bare_arm, joint_radii, motion_bound
from .scene import Scene, in_collision, place_walls
from .walls import WallSpec, make_wall

# Planar letter-like crossing poses (rolls = 0), produced by
# scripts/gen_letter_poses.py (IK on the beta = 0 arm with a self-clearance
# bound) and pasted here so every trial uses the same numbers.
LETTER_POSES: dict[str, np.ndarray] = {
    # upright
    "I": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    # upper arm vertical, forearm horizontal; self clearance 54.5 mm
    "L": np.array([0.0, 0.0503, 0.0, -1.5190, 0.0, 0.0013, 0.0]),
    # arm leaning forward, forearm hanging down; self clearance 37.7 mm
    "SEVEN": np.array([0.0, 0.9499, 0.0, -2.0045, 0.0, 0.1772, 0.0]),
    # forearm folded back over the shoulder; self clearance 47.8 mm
    "Z": np.array([0.0, 0.4506, 0.0, 1.3656, 0.0, 0.1195, 0.0]),
    # arm leaning forward, forearm turned back up; self clearance 54.4 mm
    "U": np.array([0.0, 1.4966, 0.0, 1.4925, 0.0, -0.0036, 0.0]),
    # gentle diagonal reach; self clearance 48.1 mm
    "SLASH": np.array([0.0, 0.4596, 0.0, -0.4503, 0.0, -0.1145, 0.0]),
}

Q_HOME = np.zeros(params.NUM_JOINTS)


@dataclass
class Crossing:
    """One crossing: the configuration to match while the wall passes.

    Attributes
    ----------
    q_star : ndarray, shape (7,)
        Crossing configuration in radians.
    t : float
        Crossing time in seconds (the slab is centred on the arm plane).
    delta : float
        Window half-width in seconds: the wall overlaps the arm for
        ``|t' - t| <= delta``.
    tol : float
        Joint-space 2-norm tolerance in radians around q_star inside the window.
    roll_draw : ndarray, shape (4,)
        The roll values drawn for this crossing (zeros at beta = 0).
    name : str
        Name of the pose.
    """

    q_star: np.ndarray          # full 7-vector
    t: float                    # crossing time, s
    delta: float                # window half-width, s
    tol: float                  # joint-space 2-norm tolerance inside the window, rad
    roll_draw: np.ndarray       # the four roll values drawn (zeros when beta = 0)
    name: str = ""

    @property
    def window(self) -> tuple[float, float]:
        """(start, end) of the window in seconds."""
        return (self.t - self.delta, self.t + self.delta)


@dataclass
class Sequence:
    """A connect problem: start configuration, crossings, walls and timing.

    Attributes
    ----------
    beta : float
        Roll band in radians.
    eps : float
        Clearance in meters.
    q0 : ndarray, shape (7,)
        Start configuration in radians (zero velocity at both ends).
    crossings : list of Crossing
        In time order.
    walls : list of WallSpec
        One per crossing, placed on the carrier.
    T : float
        Trajectory duration in seconds; q_of_t must cover [0, T].
    speed : float
        Wall speed in m/s.
    rho : float
        Motion bound of the free joints in m/rad.
    """

    beta: float
    eps: float
    q0: np.ndarray
    crossings: list[Crossing]
    walls: list[WallSpec]
    T: float                    # total trajectory duration, s
    speed: float = params.WALL_SPEED
    rho: float = 0.0
    extras: dict = field(default_factory=dict)

    @property
    def n_walls(self) -> int:
        """Number of walls."""
        return len(self.crossings)

    def wall_y_of_t(self):
        """Return the carrier translation ``wall_y(t)`` in meters as a function of time in seconds."""
        t1 = self.crossings[0].t
        return lambda t: self.speed * (t - t1)

    def q_stars(self) -> np.ndarray:
        """Return the crossing configurations, shape (n_walls, 7)."""
        return np.array([c.q_star for c in self.crossings])


def draw_rolls(q_pitch: np.ndarray, beta: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (q, rolls): `q_pitch` with the four roll joints drawn uniformly in [-beta, beta]."""
    q = np.asarray(q_pitch, dtype=float).ravel().copy()
    rolls = rng.uniform(-beta, beta, size=len(params.ROLL_JOINTS)) if beta > 0 else np.zeros(4)
    q[list(params.ROLL_JOINTS)] = rolls
    return q, rolls


def make_sequence(
    q_stars: list[np.ndarray],
    beta: float,
    eps: float,
    rng: np.random.Generator | None = None,
    *,
    q0: np.ndarray = Q_HOME,
    t_first: float = params.T_FIRST,
    spacing: float = params.WINDOW_SPACING,
    tail: float = params.T_TAIL,
    names: list[str] | None = None,
    strip: float = params.STRIP_HEIGHT,
) -> Sequence:
    """Build a connect problem from crossing poses.

    Parameters
    ----------
    q_stars : list of ndarray, shape (7,)
        Pitch poses in radians; their roll joints are redrawn.
    beta : float
        Roll band in radians.
    eps : float
        Clearance in meters.
    rng : numpy.random.Generator, optional
        Source of the roll draws; required when beta > 0.
    q0 : ndarray, shape (7,)
        Start configuration.
    t_first, spacing, tail : float
        First crossing time, time between crossings and time after the
        last one, in seconds.
    names : list of str, optional
        Pose names.
    strip : float
        Strip height in meters for cutting the walls.

    Returns
    -------
    Sequence

    Raises
    ------
    ValueError
        If beta > 0 and `rng` is None.
    """
    if beta > 0 and rng is None:
        raise ValueError("beta > 0 needs an rng for the roll draws")
    free = params.PITCH_JOINTS if beta <= 0 else tuple(range(params.NUM_JOINTS))
    rho = motion_bound(joint_radii(), free)
    crossings, walls = [], []
    for i, q_pitch in enumerate(q_stars):
        q, rolls = draw_rolls(q_pitch, beta, rng)
        wall = make_wall(q, eps, strip=strip)
        t = t_first + i * spacing
        crossings.append(Crossing(
            q_star=q, t=t, delta=params.window_half_width(wall.arm_depth),
            tol=params.window_tolerance(eps, rho), roll_draw=rolls,
            name=names[i] if names else f"q{i}"))
        walls.append(wall)
    place_walls(walls, [c.t for c in crossings])
    T = crossings[-1].t + tail
    return Sequence(beta=beta, eps=eps, q0=np.asarray(q0, dtype=float).ravel().copy(),
                    crossings=crossings, walls=walls, T=T, rho=rho)


def reference_path(seq: Sequence):
    """Return the piecewise-linear reference path used as a warm start.

    Parameters
    ----------
    seq : Sequence

    Returns
    -------
    pydrake.trajectories.PiecewisePolynomial
        q(t) through q0 and the crossings, held at q*_i over each window
        and at the last q* until T; ``.value(t)`` is shape (7, 1).

    Raises
    ------
    ValueError
        If the windows overlap or leave no transit time.
    """
    times = [0.0]
    knots = [seq.q0]
    for c in seq.crossings:
        times += [c.t - c.delta, c.t + c.delta]
        knots += [c.q_star, c.q_star]
    times.append(seq.T)
    knots.append(seq.crossings[-1].q_star)
    times = np.array(times)
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"windows overlap or leave no transit time: {times}")
    return PiecewisePolynomial.FirstOrderHold(times, np.column_stack(knots))


def in_window(seq: Sequence, t: float) -> Crossing | None:
    """Return the crossing whose window contains time `t` in seconds, or None."""
    for c in seq.crossings:
        if c.t - c.delta <= t <= c.t + c.delta:
            return c
    return None


# --- straight-line transit feasibility (sequence acceptance test) ----------

def toppra_time(q_a: np.ndarray, q_b: np.ndarray, scale: float = params.LIMIT_SCALE) -> float | None:
    """Return the minimum time in seconds to travel the straight line from q_a to q_b.

    Under the imposed velocity and acceleration limits (Toppra on the bare
    7-DOF arm); None if Toppra fails.
    """
    arm = bare_arm()
    path = PiecewisePolynomial.FirstOrderHold([0.0, 1.0], np.column_stack([q_a, q_b]))
    grid = Toppra.CalcGridPoints(path, CalcGridPointsOptions())
    tp = Toppra(path, arm.plant, grid)
    v = scale * arm.plant.GetVelocityUpperLimits()
    a = scale * arm.plant.GetAccelerationUpperLimits()
    tp.AddJointVelocityLimit(-v, v)
    tp.AddJointAccelerationLimit(-a, a)
    s_of_t = tp.SolvePathParameterization()
    if s_of_t is None:
        return None
    return float(PathParameterizedTrajectory(path, s_of_t).end_time())


def edge_is_free(scene: Scene, q_a: np.ndarray, q_b: np.ndarray, wall_y: float, step: float) -> bool:
    """Return True if the straight edge q_a -> q_b is collision free at wall_y, checked every `step` rad (2-norm)."""
    n = max(1, int(np.ceil(np.linalg.norm(scene.q_free(q_b) - scene.q_free(q_a)) / step)))
    for k in range(n + 1):
        if in_collision(scene, q_a + (q_b - q_a) * k / n, wall_y):
            return False
    return True


def transit_feasible(seq: Sequence, scene: Scene, edge_step: float = 0.01) -> tuple[bool, dict]:
    """Test whether straight-line transits solve the sequence (sufficient, not necessary).

    Every straight-line edge q0 -> q*_1 -> ... must be self-collision free
    with the wall far away, and its Toppra travel time plus the two adjacent
    window half-widths must fit the time available.

    Parameters
    ----------
    seq : Sequence
    scene : Scene
        A scene of the same beta.
    edge_step : float
        Edge-check resolution in radians.

    Returns
    -------
    ok : bool
    info : dict
        Per leg: edge free, Toppra time and time available.

    Notes
    -----
    Curved paths may solve sequences this rejects, so a generator using it
    is biased toward easy sequences.
    """
    legs = [(seq.q0, 0.0, 0.0)] + [(c.q_star, c.t, c.delta) for c in seq.crossings]
    info = {"legs": []}
    ok = True
    far = -5.0
    for (qa, ta, da), (qb, tb, db) in zip(legs[:-1], legs[1:]):
        free = edge_is_free(scene, qa, qb, far, edge_step)
        tt = toppra_time(qa, qb)
        avail = (tb - db) - (ta + da)
        fits = tt is not None and tt <= avail
        info["legs"].append({"edge_free": free, "toppra_time": tt, "available": avail})
        ok &= free and fits
    return ok, info


def random_sequence(
    n: int,
    rng: np.random.Generator,
    beta: float,
    eps: float,
    scene: Scene,
    *,
    pose_pool: dict[str, np.ndarray] | None = None,
    max_tries: int = 100,
    **kwargs,
) -> Sequence | None:
    """Draw a random connect problem.

    Parameters
    ----------
    n : int
        Number of crossings.
    rng : numpy.random.Generator
        Source of the pose choices and roll draws.
    beta : float
        Roll band in radians.
    eps : float
        Clearance in meters.
    scene : Scene
        A scene of the same beta, for the transit test.
    pose_pool : dict, optional
        Name to pose; defaults to LETTER_POSES.  No pose repeats
        immediately.
    max_tries : int
        Draws before giving up.
    **kwargs
        Passed to `make_sequence`.

    Returns
    -------
    Sequence or None
        The first draw that passes `transit_feasible`, or None.
    """
    pool = pose_pool or LETTER_POSES
    names = list(pool)
    for _ in range(max_tries):
        chosen = []
        while len(chosen) < n:
            cand = names[rng.integers(len(names))]
            if not chosen or cand != chosen[-1]:
                chosen.append(cand)
        seq = make_sequence([pool[c] for c in chosen], beta, eps, rng, names=chosen, **kwargs)
        ok, info = transit_feasible(seq, scene)
        if ok:
            seq.extras["transit"] = info
            return seq
    return None
