"""Bridge-test sampling for the find task.

Hsu, Jiang, Reif and Sun, "The bridge test for sampling narrow passages with
probabilistic roadmap planners", ICRA 2003.  A narrow passage is free space
squeezed between obstacles, so a short segment (a "bridge") whose two ends
are in collision and whose midpoint is free is likely to lie across one.
This finder draws such bridges and proposes their midpoints.

Check it:  python3 hitw check planners/bridge_test.py:find_bridge
"""

from dataclasses import dataclass

import numpy as np

from hitw.api import planner


@dataclass(frozen=True)
class BridgeConfig:
    """Options of the bridge test.

    Parameters
    ----------
    sigma : float
        Standard deviation in radians of the offset from a bridge's first
        end to its second, per joint.
    """

    sigma: float = 0.1


@planner("bridge-test", task="find", config=BridgeConfig, display_name="Bridge test")
def find_bridge(wall, svc, budget, rng, cfg):
    """Propose the free midpoints of bridges whose two ends collide.

    Parameters
    ----------
    wall : hitw.api.WallView
        The wall; it is centred on the arm at ``wall.t_cross``.
    svc : hitw.api.Services
        Collision checks against the wall (``svc.in_collision``).
    budget : hitw.api.Budget
        Checked between bridges, since many bridges are tried per candidate.
    rng : numpy.random.Generator
        Source of every random number.
    cfg : BridgeConfig
        Finder options.

    Yields
    ------
    ndarray, shape (7,)
        A bridge midpoint that is collision free, in radians.
    """
    scene, t = svc.scene, wall.t_cross
    lo, hi = scene.free_limits()
    while not budget.expired():
        a = rng.uniform(lo, hi)
        if not svc.in_collision(scene.q_full(a), t):
            continue
        b = np.clip(a + rng.normal(scale=cfg.sigma, size=a.shape), lo, hi)
        if not svc.in_collision(scene.q_full(b), t):
            continue
        mid = scene.q_full((a + b) / 2)
        if not svc.in_collision(mid, t):
            yield mid
