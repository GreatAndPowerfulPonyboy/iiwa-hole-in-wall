"""The find task's validity standard: when a candidate fits through the wall.

A candidate is valid when its roll joints are inside the beta band and the
smallest signed distance between the arm and anything (wall or another arm
link) is >= 0, on the prismatic scene with the slab centred on the arm
plane (scene.min_distance, the function the connect task is graded with).
The opening is the arm silhouette buffered by eps, so a configuration with
clearance >= eps exists only at the configuration the opening was cut for
(realized clearance eps + 0.07 mm); the valid set is the set of
configurations whose silhouette fits inside the opening, roughly a ball of
radius eps / rho around it, and its size is what eps controls.  The harness
checks every candidate a finder proposes with `validate`.
"""

from __future__ import annotations

import numpy as np

from . import params
from .scene import Scene, in_collision, min_distance


def validate(check_scene: Scene, q_full: np.ndarray, beta: float) -> tuple[bool, float | None]:
    """Check a candidate as `is_valid` does, skipping the distance query when it collides.

    Parameters
    ----------
    check_scene : Scene
        The prismatic scene of the wall.
    q_full : ndarray, shape (7,)
        Candidate joint vector in radians.
    beta : float
        Roll band in radians.

    Returns
    -------
    valid : bool
        Same verdict as `is_valid`.
    min_distance : float or None
        The minimum signed distance in meters (capped at 0.1) when the
        candidate is collision free; None when a roll joint is outside the
        band or the boolean check finds a penetration.

    Notes
    -----
    A penetrating pair has negative signed distance, so a candidate that
    `in_collision` rejects has min_distance < 0 and is invalid either way;
    the boolean check is about 20x cheaper.
    """
    q_full = np.asarray(q_full, dtype=float).ravel()
    if any(abs(q_full[j]) > beta + 1e-6 for j in params.ROLL_JOINTS):
        return False, None
    if in_collision(check_scene, q_full, 0.0):
        return False, None
    d = min_distance(check_scene, q_full, 0.0, cap=0.1)
    return bool(d >= 0.0), float(d)


def is_valid(check_scene: Scene, q_full: np.ndarray, beta: float) -> tuple[bool, float]:
    """Check a candidate crossing configuration against the find task's standard.

    Parameters
    ----------
    check_scene : Scene
        The prismatic scene of the wall.
    q_full : ndarray, shape (7,)
        Candidate joint vector in radians.
    beta : float
        Roll band in radians.

    Returns
    -------
    valid : bool
        Roll joints inside the band and minimum signed distance >= 0 with
        the slab centred on the arm plane.
    min_distance : float
        The minimum signed distance in meters (capped at 0.1), or NaN when a
        roll joint is outside the band.
    """
    q_full = np.asarray(q_full, dtype=float).ravel()
    for j in params.ROLL_JOINTS:
        if abs(q_full[j]) > beta + 1e-6:
            return False, float("nan")
    d = min_distance(check_scene, q_full, 0.0, cap=0.1)
    return bool(d >= 0.0), float(d)
