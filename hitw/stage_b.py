"""Stage B: only the wall is given.  How hard is it to find a valid crossing
configuration, as a function of the clearance eps?

A trial takes the wall of the 1-wall sequence of (seed, beta) at eps (so the
hidden q* and its roll draw are the ones Stage A used), places it centred on
the arm plane (carrier translation 0), and asks two searchers for any
configuration that fits through it.

Validity is the Stage A standard on the shared distance function: the
smallest signed distance between the arm and anything (wall or another arm
link) is >= 0, checked on the prismatic scene (scene.min_distance), with the
roll joints inside the beta band.  The opening is the arm silhouette buffered
by eps, so a configuration with clearance >= eps exists only at q* itself (the
realized clearance there is eps + 0.07 mm); the valid set is the set of
configurations whose silhouette fits inside the opening, roughly a ball of
radius eps / rho around q* in configuration space, and its size is what eps
controls.

B1: nonlinear IK (Drake InverseKinematics on the welded-wall scene, SNOPT)
    from K uniformly random initial guesses with
    AddMinimumDistanceLowerBoundConstraint(bound), bound = eps / 2 by default
    (bound = eps leaves 0.07 mm of feasible slack; see PROGRESS M5).  The
    constraint covers wall pairs and arm self pairs.  Every restart's outcome
    is recorded; p_hat = successes / K and P(success within K) =
    1 - (1 - p_hat)^K are computed in analysis so K need not be fixed.
B2: uniform rejection sampling in the joint limits with the same checker
    (scene.in_collision, the boolean form of min_distance), under a
    wall-clock budget equal to B1's total time in the same trial (equal
    compute, paired), capped at max_samples.  Zero hits in n samples give the
    95% upper bound ~3/n.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from pydrake.all import InverseKinematics, SnoptSolver, SolverOptions

from . import params
from .evaluate import DRAKE_VERSION, _commit, _jsonable, trial_sequence
from .scene import Scene, in_collision, make_scene, min_distance

K_DEFAULT = 30
MAJOR_ITERATIONS = 200
INFLUENCE_OFFSET = 0.01     # m, MinimumDistanceLowerBoundConstraint influence_distance_offset
B2_MAX_SAMPLES = 1_000_000
ZERO_HIT_UPPER = 3.0        # 95% upper bound on the hit rate after zero hits: ZERO_HIT_UPPER / n


def random_q_free(scene: Scene, rng: np.random.Generator) -> np.ndarray:
    lo, hi = scene.free_limits()
    return rng.uniform(lo, hi)


def is_valid(check_scene: Scene, q_full: np.ndarray, beta: float) -> tuple[bool, float]:
    """(valid, min signed distance) on the prismatic scene with the slab
    centred on the arm plane; roll joints must be inside the band."""
    q_full = np.asarray(q_full, dtype=float).ravel()
    for j in params.ROLL_JOINTS:
        if abs(q_full[j]) > beta + 1e-6:
            return False, float("nan")
    d = min_distance(check_scene, q_full, 0.0, cap=0.1)
    return bool(d >= 0.0), float(d)


def b1_ik(ik_scene: Scene, check_scene: Scene, beta: float, bound: float, rng: np.random.Generator,
          K: int = K_DEFAULT, major_iterations: int = MAJOR_ITERATIONS) -> dict:
    """K IK solves from uniform random initial guesses.  Each restart is
    recorded: SNOPT info, solve time, the solution, the realized clearance
    and whether it passes the shared validity check."""
    idx = ik_scene.arm_position_indices()
    opts = SolverOptions()
    opts.SetOption(SnoptSolver.id(), "Major iterations limit", int(major_iterations))
    solver = SnoptSolver()
    restarts = []
    t_all = time.perf_counter()
    for k in range(K):
        q0_free = random_q_free(ik_scene, rng)
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
        q_full = ik_scene.q_full(q_plant[idx])
        valid, d = is_valid(check_scene, q_full, beta)
        restarts.append({"k": k, "reported_success": bool(result.is_success()),
                         "info": int(result.get_solver_details().info), "solve_time": dt,
                         "valid": valid, "min_distance": d, "q": q_full.tolist(), "q_guess": q0_free.tolist()})
    total = time.perf_counter() - t_all
    n_valid = sum(r["valid"] for r in restarts)
    n_reported = sum(r["reported_success"] for r in restarts)
    return {"K": K, "bound": float(bound), "n_valid": n_valid, "n_reported_success": n_reported,
            "n_false_positive": sum(r["reported_success"] and not r["valid"] for r in restarts),
            "p_hat": n_valid / K, "total_time": total,
            "restart_time_median": float(np.median([r["solve_time"] for r in restarts])),
            "time_to_first_valid": next((sum(r["solve_time"] for r in restarts[:i + 1])
                                         for i, r in enumerate(restarts) if r["valid"]), None),
            "restarts": restarts}


def b2_sampling(check_scene: Scene, beta: float, rng: np.random.Generator, budget_s: float,
                max_samples: int = B2_MAX_SAMPLES) -> dict:
    """Uniform samples in the joint limits until the wall-clock budget or the
    sample cap; a hit is a sample that passes the shared validity check
    (in_collision first, then min_distance to record the clearance)."""
    t0 = time.perf_counter()
    n = hits = 0
    first_hit = None
    hit_qs = []
    while n < max_samples and time.perf_counter() - t0 < budget_s:
        q_full = check_scene.q_full(random_q_free(check_scene, rng))
        n += 1
        if not in_collision(check_scene, q_full, 0.0):
            valid, d = is_valid(check_scene, q_full, beta)
            if valid:
                hits += 1
                if first_hit is None:
                    first_hit = {"n": n, "time": time.perf_counter() - t0, "min_distance": d}
                if len(hit_qs) < 5:
                    hit_qs.append(q_full.tolist())
    elapsed = time.perf_counter() - t0
    return {"budget_s": float(budget_s), "n_samples": n, "n_hits": hits, "hit_rate": hits / n if n else None,
            "hit_rate_upper95": (ZERO_HIT_UPPER / n) if (n and hits == 0) else None,
            "elapsed": elapsed, "samples_per_s": n / elapsed if elapsed > 0 else None,
            "first_hit": first_hit, "hit_qs": hit_qs}


@dataclass
class StageBTrial:
    seed: int
    beta: float
    eps: float
    K: int = K_DEFAULT
    bound_factor: float = 0.5      # IK bound = bound_factor * eps
    b2_budget: float | None = None  # seconds; None = B1's total time in this trial
    b2_max_samples: int = B2_MAX_SAMPLES

    @property
    def key(self):
        return ("B", self.seed, self.beta, self.eps, 1)


def run_stage_b_trial(trial: StageBTrial) -> dict:
    """One Stage B trial: B1 then B2 on the same wall.  One JSON record."""
    t_start = time.perf_counter()
    r = {"method": "B", "seed": int(trial.seed), "beta": float(trial.beta), "eps": float(trial.eps), "n_walls": 1,
         "drake_version": DRAKE_VERSION, "hitw_commit": _commit(), "K": trial.K, "bound_factor": trial.bound_factor}
    seq = trial_sequence(trial.seed, trial.beta, trial.eps, 1)
    if seq is None:
        r.update({"status": "infeasible", "note": "no sequence"})
        return r
    c = seq.crossings[0]
    wall = seq.walls[0]
    r.update({"sequence": [c.name], "roll_draw": [c.roll_draw], "q_star": c.q_star, "wall_boxes": wall.n_boxes})
    check_scene = make_scene([wall], beta=trial.beta)
    ik_scene = make_scene([wall], beta=trial.beta, fixed_wall_y=0.0)
    valid_star, d_star = is_valid(check_scene, c.q_star, trial.beta)
    r["q_star_valid"] = valid_star
    r["q_star_min_distance"] = d_star
    rng = np.random.default_rng(1_000_000 + trial.seed)        # independent of the sequence rng
    b1 = b1_ik(ik_scene, check_scene, trial.beta, trial.bound_factor * trial.eps, rng, K=trial.K)
    budget = trial.b2_budget if trial.b2_budget is not None else b1["total_time"]
    b2 = b2_sampling(check_scene, trial.beta, rng, budget, max_samples=trial.b2_max_samples)
    # distance of the valid IK solutions from the hidden q* (free joints)
    q_star_free = check_scene.q_free(c.q_star)
    b1["valid_dist_to_q_star"] = [float(np.linalg.norm(check_scene.q_free(x["q"]) - q_star_free))
                                  for x in b1["restarts"] if x["valid"]]
    r.update({"b1": b1, "b2": b2, "status": "success" if b1["n_valid"] or b2["n_hits"] else "infeasible",
              "total_time": time.perf_counter() - t_start})
    return _jsonable(r)
