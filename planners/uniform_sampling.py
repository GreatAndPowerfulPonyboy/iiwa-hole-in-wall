"""uniform-sampling: a finder for the find task by rejection sampling.

Configurations drawn uniformly within the joint limits (roll joints within
the beta band), up to max_samples; the budget usually ends the search first.
Its hit rate measures the volume of the valid set, which shrinks with eps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hitw.api import Budget, Services, WallView, planner

MAX_SAMPLES = 1_000_000


@dataclass(frozen=True)
class UniformSamplingConfig:
    """Options of the uniform-sampling finder.

    Parameters
    ----------
    max_samples : int
        Sample cap; the budget usually ends the search first.
    """

    max_samples: int = MAX_SAMPLES


@planner("uniform-sampling", task="find", config=UniformSamplingConfig, display_name="Uniform sampling")
def find_uniform_sampling(wall: WallView, svc: Services, budget: Budget, rng: np.random.Generator,
                          cfg: UniformSamplingConfig):
    """Propose configurations drawn uniformly within the joint limits.

    Parameters
    ----------
    wall : WallView
        The wall.
    svc : Services
        The wall's services; samples are drawn in ``svc.scene``'s limits.
    budget : Budget
        Wall-clock budget; the harness stops asking for samples once it
        expires.
    rng : numpy.random.Generator
        Source of the samples.
    cfg : UniformSamplingConfig
        Finder options.

    Yields
    ------
    q : ndarray, shape (7,)
        One sample in radians (roll joints within the band).
    """
    scene = svc.scene
    lo, hi = scene.free_limits()
    for _ in range(cfg.max_samples):
        yield scene.q_full(rng.uniform(lo, hi))
