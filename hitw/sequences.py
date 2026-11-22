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
    q_star: np.ndarray          # full 7-vector
    t: float                    # crossing time, s
    delta: float                # window half-width, s
    tol: float                  # joint-space 2-norm tolerance inside the window, rad
    roll_draw: np.ndarray       # the four roll values drawn (zeros when beta = 0)
    name: str = ""

    @property
    def window(self) -> tuple[float, float]:
        return (self.t - self.delta, self.t + self.delta)


@dataclass
class Sequence:
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
        return len(self.crossings)

    def wall_y_of_t(self):
        t1 = self.crossings[0].t
        return lambda t: self.speed * (t - t1)

    def q_stars(self) -> np.ndarray:
        return np.array([c.q_star for c in self.crossings])


def draw_rolls(q_pitch: np.ndarray, beta: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """q with the four roll joints redrawn uniformly in [-beta, beta]."""
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
    """Build the walls, windows and tolerances for the given pitch poses.  With
    beta > 0 an rng is required for the roll draws."""
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
    """Piecewise-linear q(t) through q0 and the crossings, held at q*_i over
    each window and at the last q* until T: the planners' warm start."""
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
    for c in seq.crossings:
        if c.t - c.delta <= t <= c.t + c.delta:
            return c
    return None


# --- straight-line transit feasibility (sequence acceptance test) ----------

def toppra_time(q_a: np.ndarray, q_b: np.ndarray, scale: float = params.LIMIT_SCALE) -> float | None:
    """Minimum travel time along the straight line from q_a to q_b under the
    imposed velocity and acceleration limits (Toppra on the bare 7-DOF arm),
    or None if Toppra fails."""
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
    """Straight-line edge check at joint-space resolution `step` (2-norm)."""
    n = max(1, int(np.ceil(np.linalg.norm(scene.q_free(q_b) - scene.q_free(q_a)) / step)))
    for k in range(n + 1):
        if in_collision(scene, q_a + (q_b - q_a) * k / n, wall_y):
            return False
    return True


def transit_feasible(seq: Sequence, scene: Scene, edge_step: float = 0.01) -> tuple[bool, dict]:
    """The sufficient (not necessary) acceptance test: every straight-line
    edge q0 -> q*_1 -> ... is self-collision free with the wall far away, and
    its Toppra travel time plus the two adjacent window half-widths fits the
    time available.  Curved paths may solve sequences this rejects, so a
    generator using it is biased toward easy sequences."""
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
    """n crossings drawn from pose_pool (default LETTER_POSES, no immediate
    repeats), rolls drawn from rng, accepted only if transit_feasible."""
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
