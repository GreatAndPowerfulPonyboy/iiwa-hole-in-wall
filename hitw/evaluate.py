"""The single success criterion: a dense re-check of the trajectory on a 1 ms
grid with the wall at its true position, using scene.min_distance (the
shared distance function).  A trajectory passes if the minimum signed
distance is >= 0 and every window is honoured.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .scene import Scene, min_distance
from .sequences import Sequence

RECHECK_DT = 1e-3
RECHECK_CAP = 0.1      # m; min_signed_distance is reported capped at this


@dataclass
class Recheck:
    """Result of the dense re-check.

    Attributes
    ----------
    passed : bool
        Collision free and every window honoured.
    collision_free : bool
        Minimum signed distance >= 0 on the grid.
    windows_ok : bool
        Within tol of q*_i throughout every window.
    min_signed_distance : float
        Meters, capped at the re-check cap.
    t_min : float
        Time of the minimum, seconds.
    window_excess : float
        Max over windows of ||q - q*||_2 - tol in radians (<= 0 is ok).
    path_length : float
        Sum of 2-norm steps over the free joints, radians.
    peak_joint_vel : float
        Largest per-joint finite-difference speed on the grid, rad/s.
    n_samples : int
        Grid points.
    elapsed : float
        Seconds the re-check took.
    """

    passed: bool
    collision_free: bool
    windows_ok: bool
    min_signed_distance: float
    t_min: float
    window_excess: float          # max over windows of ||q - q*||_2 - tol (<= 0 is ok)
    path_length: float            # sum of 2-norm steps over the free joints, rad
    peak_joint_vel: float         # rad/s, finite difference on the grid
    n_samples: int
    elapsed: float
    extras: dict = field(default_factory=dict)


def recheck(q_of_t: Callable[[float], np.ndarray], seq: Sequence, scene: Scene,
            dt: float = RECHECK_DT, cap: float = RECHECK_CAP) -> Recheck:
    """Re-check a trajectory densely with the wall at its true position.

    Parameters
    ----------
    q_of_t : callable
        ``q_of_t(t) -> ndarray, shape (7,)``, the full joint vector in
        radians for t in [0, seq.T] seconds.
    seq : Sequence
        The problem.
    scene : Scene
        The prismatic-wall scene of `seq`.
    dt : float
        Grid step in seconds.
    cap : float
        Distance cap in meters for `scene.min_distance`.

    Returns
    -------
    Recheck
    """
    t0 = time.perf_counter()
    wall_y = seq.wall_y_of_t()
    ts = np.arange(0.0, seq.T + dt / 2, dt)
    qs = np.array([q_of_t(t) for t in ts])
    d = np.array([min_distance(scene, q, wall_y(t), cap=cap) for q, t in zip(qs, ts)])
    k = int(np.argmin(d))
    free_q = qs[:, list(scene.free_joints)]
    excess = -np.inf
    for c in seq.crossings:
        mask = (ts >= c.t - c.delta) & (ts <= c.t + c.delta)
        dev = np.linalg.norm(free_q[mask] - scene.q_free(c.q_star), axis=1) if mask.any() else np.array([np.inf])
        excess = max(excess, float(dev.max() - c.tol))
    steps = np.linalg.norm(np.diff(free_q, axis=0), axis=1)
    peak_v = float((np.abs(np.diff(free_q, axis=0)) / dt).max()) if len(ts) > 1 else 0.0
    collision_free = bool(d[k] >= 0.0)
    windows_ok = bool(excess <= 0.0)
    return Recheck(
        passed=collision_free and windows_ok, collision_free=collision_free, windows_ok=windows_ok,
        min_signed_distance=float(d[k]), t_min=float(ts[k]), window_excess=float(excess),
        path_length=float(steps.sum()), peak_joint_vel=peak_v, n_samples=len(ts),
        elapsed=time.perf_counter() - t0, extras={"cap": cap, "dt": dt},
    )


def classify(solver_status: str, rc: Recheck | None) -> tuple[str, str | None]:
    """Classify a trial from the planner's reported status and the re-check.

    Parameters
    ----------
    solver_status : str
        What the planner reported, one of `STATUSES`.
    rc : Recheck or None
        The re-check, or None when there is no trajectory.

    Returns
    -------
    status : str
        One of `STATUSES`; "success" only if the planner reported success
        and the re-check passed.
    failure_label : str or None
        One of `FAILURE_LABELS`; a reported success that fails the re-check
        is ``reported_feasible_but_collides`` or
        ``collision_free_but_misses_window``.
    """
    if solver_status != "success":
        label = {"timeout": "timeout", "iteration_limit": "timeout"}.get(solver_status, "infeasible_reported")
        return solver_status, label
    if rc is None:
        return "infeasible", "infeasible_reported"
    if not rc.collision_free:
        return "infeasible", "reported_feasible_but_collides"
    if not rc.windows_ok:
        return "infeasible", "collision_free_but_misses_window"
    return "success", None


# "unsupported": the planner does not handle this beta; such trials are
# recorded but not counted in success rates.
STATUSES = ("success", "timeout", "iteration_limit", "infeasible", "unsupported")
FAILURE_LABELS = (None, "infeasible_reported", "reported_feasible_but_collides",
                  "collision_free_but_misses_window", "timeout")
