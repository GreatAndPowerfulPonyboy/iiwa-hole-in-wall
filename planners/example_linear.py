"""Example connect planner: straight joint-space lines between the crossing
configurations, holding each one through its window.
Check it:  python3 hitw check planners/example_linear.py:plan_linear
Run it:    python3 hitw run --planner planners/example_linear.py:plan_linear --seeds 0-2 --out results/linear.jsonl
"""

import time
from dataclasses import dataclass

import numpy as np

from hitw.api import PlanOutcome, planner


@dataclass(frozen=True)
class LinearConfig:
    """Options of the straight-line planner.

    Parameters
    ----------
    lead : float
        Seconds before each window opens at which the arm reaches q*_i and
        starts holding it; 0 arrives exactly as the window opens.
    """

    lead: float = 0.0


@planner("linear", config=LinearConfig, display_name="Straight line + dwell")
def plan_linear(seq, svc, budget, rng, cfg):
    """Plan straight joint-space lines between the crossings, holding each q*_i through its window.

    Parameters
    ----------
    seq : hitw.sequences.Sequence
        The crossings (q_star, t, delta), the start q0 and the duration T.
    svc : hitw.api.Services
        Trial services (unused: this planner does not look at the scene).
    budget : hitw.api.Budget
        Wall-clock budget (unused: planning takes microseconds).
    rng : numpy.random.Generator
        The planner's random stream (unused).
    cfg : LinearConfig
        Planner options.

    Returns
    -------
    hitw.api.PlanOutcome
        "success" with a piecewise-linear q(t) on [0, T], or "infeasible"
        when `cfg.lead` leaves no time for a transit.
    """
    t0 = time.perf_counter()
    times, knots = [0.0], [seq.q0]
    for c in seq.crossings:
        arrive = c.t - c.delta - cfg.lead
        if arrive <= times[-1]:
            return PlanOutcome(status="infeasible", q_of_t=None, solve_time=time.perf_counter() - t0,
                               extras={"reason": f"no transit time before the window at t = {c.t:g} s"})
        times += [arrive, c.t + c.delta]
        knots += [c.q_star, c.q_star]
    times.append(seq.T)
    knots.append(seq.crossings[-1].q_star)
    ts, qs = np.array(times), np.array(knots)

    def q_of_t(t):
        t = min(max(float(t), 0.0), seq.T)
        return np.array([np.interp(t, ts, qs[:, j]) for j in range(qs.shape[1])])

    return PlanOutcome(status="success", q_of_t=q_of_t, solve_time=time.perf_counter() - t0,
                       extras={"knot_times": times})
