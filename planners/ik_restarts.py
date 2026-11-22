"""ik-restarts: a finder for the find task by nonlinear inverse kinematics.

Drake's InverseKinematics on the wall welded at the arm plane, solved by
SNOPT from K uniformly random initial guesses under
AddMinimumDistanceLowerBoundConstraint(bound), bound = eps / 2 by default:
with bound = eps the feasible set is the 0.07 mm of realized slack around
the configuration the opening was cut for.  The constraint covers wall pairs
and arm self pairs.  One candidate per restart, with SNOPT's outcome
attached, so the records count SNOPT's false positives.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from pydrake.all import InverseKinematics, SnoptSolver, SolverOptions

from hitw.api import Budget, Services, WallView, planner

K_DEFAULT = 30
BOUND_FACTOR = 0.5
MAJOR_ITERATIONS = 200
INFLUENCE_OFFSET = 0.01     # m, MinimumDistanceLowerBoundConstraint influence_distance_offset


@dataclass(frozen=True)
class IkRestartsConfig:
    """Options of the IK-restarts finder.

    Parameters
    ----------
    K : int
        Restarts.
    bound_factor : float
        IK minimum-distance bound as a fraction of eps.
    major_iterations : int
        SNOPT major iterations limit per restart.
    """

    K: int = K_DEFAULT
    bound_factor: float = BOUND_FACTOR
    major_iterations: int = MAJOR_ITERATIONS


@planner("ik-restarts", task="find", config=IkRestartsConfig, display_name="IK restarts")
def find_ik_restarts(wall: WallView, svc: Services, budget: Budget, rng: np.random.Generator,
                     cfg: IkRestartsConfig):
    """Propose crossing configurations by IK from random restarts.

    Parameters
    ----------
    wall : WallView
        The wall.
    svc : Services
        The wall's services; the IK runs on ``svc.welded_scene(0.0)``.
    budget : Budget
        Wall-clock budget; the harness stops asking for restarts once it
        expires.
    rng : numpy.random.Generator
        Source of the initial guesses.
    cfg : IkRestartsConfig
        Finder options.

    Yields
    ------
    q : ndarray, shape (7,)
        The IK solution of one restart, in radians.
    info : dict
        k, reported_success (SNOPT says solved), info (SNOPT exit code),
        solve_time (s) and q_guess (the initial guess, free joints).
    """
    ik_scene = svc.welded_scene(0.0)
    lo, hi = ik_scene.free_limits()
    idx = ik_scene.arm_position_indices()
    opts = SolverOptions()
    opts.SetOption(SnoptSolver.id(), "Major iterations limit", int(cfg.major_iterations))
    solver = SnoptSolver()
    bound = cfg.bound_factor * wall.eps
    for k in range(cfg.K):
        q0_free = rng.uniform(lo, hi)
        ik = InverseKinematics(ik_scene.plant, ik_scene.plant_context)
        ik.AddMinimumDistanceLowerBoundConstraint(float(bound), INFLUENCE_OFFSET)
        prog = ik.prog()
        guess = np.zeros(ik_scene.plant.num_positions())
        guess[idx] = q0_free
        prog.SetInitialGuess(ik.q(), guess)
        t0 = time.perf_counter()
        result = solver.Solve(prog, None, opts)
        dt = time.perf_counter() - t0
        q_plant = result.GetSolution(ik.q())
        yield ik_scene.q_full(q_plant[idx]), {"k": k, "reported_success": bool(result.is_success()),
                                              "info": int(result.get_solver_details().info), "solve_time": dt,
                                              "q_guess": q0_free.tolist()}
