"""M1: KinematicTrajectoryOptimization through the given crossings.

The spline carries every plant position: the free arm joints and the wall's
prismatic coordinate.  The wall row is pinned so the spline reproduces
wall_y(t) exactly (a clamped B-spline with control points at the Greville
abscissae reproduces any linear function), and the duration is fixed to T.
Drake's MinimumDistanceLowerBoundConstraint on the full plant then sees the
wall at its true position at every sampled s.

Windows are enforced on the control points: every control point whose basis
function is active over [t_i - delta_i, t_i + delta_i] is boxed into
q*_i +/- tol_i / sqrt(n_free), so by the convex-hull property the path stays
within tol_i (2-norm) of q*_i throughout the window.  Collision constraints
are enforced only at sampled s; the dense re-check (evaluate.recheck) is the
arbiter.

Solver effort is recorded as the number of function evaluations, counted
through a visualization callback that Drake's SNOPT wrapper invokes on every
evaluation (verified not to change the solution).  SNOPT's major iteration
count is not obtainable in Drake 1.58: SnoptSolverDetails carries only info,
F, xmul, Fmul and solve_time, the wrapper passes null snLog/snSTOP callbacks,
and the "Print file" option yields only the SNOPT banner whatever the print
level.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from pydrake.all import (
    BsplineBasis,
    BsplineTrajectory,
    KinematicTrajectoryOptimization,
    KnotVectorType,
    MinimumDistanceLowerBoundConstraint,
    SnoptSolver,
    SolverOptions,
)

from .scene import Scene
from .sequences import Sequence, reference_path

SPLINE_ORDER = 4
CPS_PER_INTERVAL = 15          # control points per crossing interval
DISTANCE_DT = 0.1              # s between sampled minimum-distance constraints
INFLUENCE_OFFSET = 0.02        # m, influence distance = bound + this
MAJOR_ITER_LIMIT = 500
RETRY_NOISE = 0.05             # rad, perturbation of the warm start on retry

# SNOPT info codes (snopt7 manual, section 4): 1-3 optimal, 11-16 infeasible,
# 31-34 iteration or time limits.
_INFO_STATUS = {**{i: "success" for i in (1, 2, 3)},
                **{i: "infeasible" for i in range(11, 17)},
                **{i: "iteration_limit" for i in (31, 32, 33)},
                34: "timeout"}


@dataclass
class PlanResult:
    status: str                               # success | infeasible | iteration_limit | timeout
    traj: BsplineTrajectory | None            # plant-space q(t) on [0, T], or None
    solve_time: float                         # total wall-clock in the solver, s (both attempts)
    attempts: int
    solver_info: list[int]                    # SNOPT info code per attempt
    cost: float | None = None
    n_evals: int = 0                          # SNOPT function evaluations, all attempts
    iterations: int | None = None             # major iterations: not obtainable, see module docstring
    extras: dict = field(default_factory=dict)


def greville(basis: BsplineBasis) -> np.ndarray:
    k = np.asarray(basis.knots())
    o = basis.order()
    return np.array([k[j + 1:j + o].mean() for j in range(basis.num_basis_functions())])


def plant_q_of_t(traj: BsplineTrajectory, scene: Scene):
    """Full 7-vector arm configuration as a function of time from the
    plant-space trajectory."""
    idx = scene.arm_position_indices()

    def q_of_t(t: float) -> np.ndarray:
        return scene.q_full(traj.value(float(t)).ravel()[idx])
    return q_of_t


def warm_start(seq: Sequence, scene: Scene, basis: BsplineBasis, rng: np.random.Generator | None = None,
               noise: float = 0.0) -> np.ndarray:
    """Control points (n_pos, N): arm rows from reference_path sampled at the
    Greville abscissae, wall row = wall_y at those times."""
    n_pos = scene.plant.num_positions()
    g = greville(basis)
    ref = reference_path(seq)
    wall_y = seq.wall_y_of_t()
    C = np.zeros((n_pos, len(g)))
    arm_idx = scene.arm_position_indices()
    for j, gj in enumerate(g):
        t = seq.T * gj
        C[arm_idx, j] = scene.q_free(ref.value(t).ravel())
        C[scene.wall_joint.position_start(), j] = wall_y(t)
    if noise > 0:
        C[arm_idx, :] += rng.normal(scale=noise, size=(len(arm_idx), len(g)))
    return C


def plan_trajopt(
    seq: Sequence,
    scene: Scene,
    *,
    bound: float | None = None,
    cps_per_interval: int = CPS_PER_INTERVAL,
    distance_dt: float = DISTANCE_DT,
    major_iter_limit: int = MAJOR_ITER_LIMIT,
    retry: bool = True,
    acceleration_bounds: bool = True,
    seed: int = 0,
    verbose: bool = False,
) -> PlanResult:
    assert scene.wall_joint is not None, "plan_trajopt needs the prismatic-wall scene"
    assert scene.beta == seq.beta
    plant = scene.plant
    n_pos = plant.num_positions()
    arm_idx = scene.arm_position_indices()
    wall_idx = scene.wall_joint.position_start()
    n_free = len(arm_idx)
    T = seq.T
    bound = seq.eps / 2 if bound is None else bound
    N = cps_per_interval * (seq.n_walls + 1)

    trajopt = KinematicTrajectoryOptimization(n_pos, N, SPLINE_ORDER, T)
    prog = trajopt.get_mutable_prog()
    cp = trajopt.control_points()
    basis = trajopt.basis()
    wall_y = seq.wall_y_of_t()

    trajopt.AddDurationConstraint(T, T)
    for j, gj in enumerate(greville(basis)):
        y = wall_y(T * gj)
        prog.AddBoundingBoxConstraint(y, y, cp[wall_idx, j])

    # Bounds must be finite (the velocity bound multiplies the duration
    # variable inside the linear constraint); the wall row gets slack bounds.
    y_reach = abs(seq.speed) * T + 1.0
    lo_free, hi_free = scene.free_limits()
    lo, hi = np.full(n_pos, -y_reach), np.full(n_pos, y_reach)
    lo[arm_idx], hi[arm_idx] = lo_free, hi_free
    trajopt.AddPositionBounds(lo, hi)
    v = np.full(n_pos, 2.0 * abs(seq.speed) + 1.0)
    v[arm_idx] = scene.velocity_limits()
    trajopt.AddVelocityBounds(-v, v)
    if acceleration_bounds:
        a = np.full(n_pos, 1.0)
        a[arm_idx] = scene.acceleration_limits()
        trajopt.AddAccelerationBounds(-a, a)

    q0 = np.full(n_pos, -np.inf)
    q0_hi = np.full(n_pos, np.inf)
    q0[arm_idx] = q0_hi[arm_idx] = scene.q_free(seq.q0)
    trajopt.AddPathPositionConstraint(q0, q0_hi, 0.0)
    zero_lo, zero_hi = np.full(n_pos, -np.inf), np.full(n_pos, np.inf)
    zero_lo[arm_idx] = zero_hi[arm_idx] = 0.0
    trajopt.AddPathVelocityConstraint(zero_lo, zero_hi, 0.0)
    trajopt.AddPathVelocityConstraint(zero_lo, zero_hi, 1.0)

    window_cps = []
    for c in seq.crossings:
        s_lo, s_hi = (c.t - c.delta) / T, (c.t + c.delta) / T
        active = basis.ComputeActiveBasisFunctionIndices([s_lo, s_hi])
        half = c.tol / np.sqrt(n_free)
        q_star = scene.q_free(c.q_star)
        for j in active:
            prog.AddBoundingBoxConstraint(q_star - half, q_star + half, cp[arm_idx, j])
        window_cps.append(list(active))

    distance = MinimumDistanceLowerBoundConstraint(plant, bound, scene.plant_context, None, INFLUENCE_OFFSET)
    s_samples = np.linspace(0.0, 1.0, int(np.ceil(T / distance_dt)) + 1)
    for s in s_samples:
        trajopt.AddPathPositionConstraint(distance, float(s))

    trajopt.AddPathEnergyCost(1.0)

    evals = [0]
    prog.AddVisualizationCallback(lambda _x: evals.__setitem__(0, evals[0] + 1), prog.decision_variables())

    solver = SnoptSolver()
    options = SolverOptions()
    options.SetOption(solver.solver_id(), "Major iterations limit", major_iter_limit)
    options.SetOption(solver.solver_id(), "Major optimality tolerance", 1e-4)
    if verbose:
        options.SetOption(solver.solver_id(), "Print file", "/tmp/hitw_snopt.out")

    rng = np.random.default_rng(seed)
    infos, evals_per_attempt, total = [], [], 0.0
    status, traj, cost = "infeasible", None, None
    for attempt in range(2 if retry else 1):
        evals[0] = 0
        C = warm_start(seq, scene, basis, rng, noise=RETRY_NOISE if attempt else 0.0)
        prog.SetInitialGuess(cp, C)
        prog.SetInitialGuess(trajopt.duration(), T)
        t0 = time.perf_counter()
        result = solver.Solve(prog, None, options)
        total += time.perf_counter() - t0
        evals_per_attempt.append(evals[0])
        info = int(result.get_solver_details().info)
        infos.append(info)
        status = _INFO_STATUS.get(info, "infeasible")
        if result.is_success():
            status = "success"
            traj = trajopt.ReconstructTrajectory(result)
            cost = float(result.get_optimal_cost())
            break
    return PlanResult(status=status, traj=traj, solve_time=total, attempts=len(infos),
                      solver_info=infos, cost=cost, n_evals=sum(evals_per_attempt),
                      extras={"n_control_points": N, "n_distance_samples": len(s_samples),
                              "bound": bound, "window_control_points": window_cps,
                              "evals_per_attempt": evals_per_attempt})
