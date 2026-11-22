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
    """q_of_t returns the full 7-vector at time t in [0, seq.T]."""
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
    """(status, failure_label) for the trial record.  A solver-reported
    success that fails the re-check is the false positive we measure."""
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


# --- trial harness ----------------------------------------------------------

import importlib.metadata as _md
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import params
from .scene import make_scene
from .sequences import random_sequence

DRAKE_VERSION = _md.version("drake")
STATUSES = ("success", "timeout", "iteration_limit", "infeasible")
FAILURE_LABELS = (None, "infeasible_reported", "reported_feasible_but_collides",
                  "collision_free_but_misses_window", "timeout")


@dataclass
class PlanOutcome:
    """What every planner returns to run_trial."""
    status: str                                  # solver-reported, one of STATUSES
    q_of_t: Callable[[float], np.ndarray] | None # full 7-vector, or None
    solve_time: float
    n_evals: int | None = None
    iterations: int | None = None
    region_build_time: float | None = None
    region_volume: dict | None = None             # M3: per-region estimates (see plan_gcs.region_volume)
    solver_info: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)


def plan_m1(seq: Sequence, scene: Scene, seed: int, **kw) -> PlanOutcome:
    from .plan_trajopt import plan_trajopt, plant_q_of_t
    res = plan_trajopt(seq, scene, seed=seed, **kw)
    return PlanOutcome(status=res.status,
                       q_of_t=plant_q_of_t(res.traj, scene) if res.traj is not None else None,
                       solve_time=res.solve_time, n_evals=res.n_evals, iterations=res.iterations,
                       solver_info=res.solver_info,
                       extras={"attempts": res.attempts, "cost": res.cost, **res.extras})


def plan_m2(seq: Sequence, scene: Scene, seed: int, **kw) -> PlanOutcome:
    from .plan_gcs import gcs_q_of_t, gcs_wall_error, plan_m2 as _m2
    res = _m2(seq, scene, seed, **kw)
    extras = {"solver": res.solver_name, "solution_result": res.solution_result, **res.extras}
    if res.traj is not None:
        extras["wall_error"] = gcs_wall_error(res.traj, seq, scene)
    return PlanOutcome(status=res.status, q_of_t=gcs_q_of_t(res.traj, scene, seq) if res.traj is not None else None,
                       solve_time=res.solve_time, extras=extras)


def _plan_m3(variant: str, margin: str = "scaled"):
    def plan(seq: Sequence, scene: Scene, seed: int, **kw) -> PlanOutcome:
        from .plan_gcs import gcs_q_of_t, gcs_wall_error, plan_m3 as _m3
        res = _m3(seq, scene, seed, variant=variant, margin=margin, **kw)
        extras = {"solver": res.solver_name, "solution_result": res.solution_result, **res.extras}
        if res.traj is not None:
            extras["wall_error"] = gcs_wall_error(res.traj, seq, scene)
        return PlanOutcome(status=res.status,
                           q_of_t=gcs_q_of_t(res.traj, scene, seq) if res.traj is not None else None,
                           solve_time=res.solve_time, region_build_time=res.region_build_time,
                           region_volume=res.region_volume, extras=extras)
    return plan


def plan_m4(seq: Sequence, scene: Scene, seed: int, **kw) -> PlanOutcome:
    from .plan_rrt import plan_rrt, rrt_q_of_t
    res = plan_rrt(seq, scene, seed, **kw)
    return PlanOutcome(status=res.status, q_of_t=rrt_q_of_t(res.knots, seq.T) if res.knots else None,
                       solve_time=res.solve_time, iterations=res.iterations, extras=res.extras)


# Method names are what the method is.  The kickoff's milestone labels (M1..M4)
# remain accepted as aliases, and results files written under them are
# normalized on load (load_results), so old and new files read the same.
METHODS: dict[str, Callable[..., PlanOutcome]] = {
    "KTO": plan_m1,                                  # KinematicTrajectoryOptimization, SNOPT
    "GCS": plan_m2,                                  # degenerate GCS chain of boxes, no IRIS
    "IrisZo+GCS": _plan_m3("zo"),                    # IRIS regions (eps-scaled margin) + GCS
    "IrisNp2+GCS": _plan_m3("np2"),
    "IrisNp+GCS": _plan_m3("np"),                    # legacy IRIS baseline
    "IrisZo+GCS-defmargin": _plan_m3("zo", margin="default"),   # Drake's default 1e-2 margin
    "RRT-Connect": plan_m4,                          # time-aware bidirectional RRT-Connect + shortcut
}
METHOD_ALIASES = {"M1": "KTO", "M2": "GCS", "M3": "IrisZo+GCS", "M3np2": "IrisNp2+GCS",
                  "M3np": "IrisNp+GCS", "M3dm": "IrisZo+GCS-defmargin", "M4": "RRT-Connect"}
STAGE_A_ORDER = list(METHODS)


def canonical_method(name: str) -> str:
    """The method's real name; accepts the milestone aliases."""
    name = METHOD_ALIASES.get(name, name)
    if name not in METHODS and name != "B":
        raise KeyError(f"unknown method {name!r}; known: {list(METHODS)} + aliases {list(METHOD_ALIASES)}")
    return name


def trial_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def trial_sequence(seed: int, beta: float, eps: float, n_walls: int) -> Sequence | None:
    """The sequence for a trial is a function of (seed, beta, n_walls) only:
    the same poses and roll draws are used for every method and every eps."""
    scene = make_scene([], beta=beta)
    return random_sequence(n_walls, trial_rng(seed), beta, eps, scene)


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5, cwd=Path(__file__).resolve().parents[1]).stdout.strip() or None
    except Exception:
        return None


def trial_key(r: dict) -> tuple:
    return (r["method"], int(r["seed"]), float(r["beta"]), float(r["eps"]), int(r["n_walls"]))


def base_record(method: str, seed: int, beta: float, eps: float, n_walls: int) -> dict:
    return {"method": canonical_method(method), "seed": int(seed), "beta": float(beta), "eps": float(eps),
            "n_walls": int(n_walls), "drake_version": DRAKE_VERSION, "hitw_commit": _commit(),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def timeout_record(method: str, seed: int, beta: float, eps: float, n_walls: int, cap: float) -> dict:
    """The record written by the sweep driver when a trial's process is killed."""
    r = base_record(method, seed, beta, eps, n_walls)
    r.update({"status": "timeout", "failure_label": "timeout", "solve_time": float(cap),
              "time_to_success": None, "wall_clock_cap": float(cap), "killed": True,
              "region_build_time": None, "region_volume": None, "iterations": None, "n_evals": None,
              "min_signed_distance": None, "path_length": None, "peak_joint_vel": None,
              "roll_draw": None, "sequence": None})
    return r


def run_trial(method: str, seed: int, beta: float, eps: float, n_walls: int, *,
              seq: Sequence | None = None, scene: Scene | None = None,
              recheck_dt: float = RECHECK_DT, cap: float | None = None, **method_kwargs) -> dict:
    """One trial: build (or take) the sequence, plan, dense re-check, and
    return the JSON-ready record.  The wall-clock cap is only recorded here;
    the sweep driver enforces it by killing the process."""
    r = base_record(method, seed, beta, eps, n_walls)
    r["wall_clock_cap"] = cap
    t_start = time.perf_counter()
    if seq is None:
        seq = trial_sequence(seed, beta, eps, n_walls)
    if seq is None:
        r.update({"status": "infeasible", "failure_label": "infeasible_reported",
                  "sequence": None, "note": "random_sequence found no transit-feasible sequence"})
        return r
    if scene is None:
        scene = make_scene(seq.walls, beta=beta)
    r["sequence"] = [c.name for c in seq.crossings]
    r["roll_draw"] = [c.roll_draw for c in seq.crossings]
    r["q_stars"] = [c.q_star for c in seq.crossings]
    r["T"] = seq.T
    r["tol"] = [c.tol for c in seq.crossings]
    r["delta"] = [c.delta for c in seq.crossings]
    out = METHODS[canonical_method(method)](seq, scene, seed, **method_kwargs)
    rc = recheck(out.q_of_t, seq, scene, dt=recheck_dt) if out.q_of_t is not None else None
    status, label = classify(out.status, rc)
    r.update({
        "status": status, "failure_label": label, "solver_status": out.status,
        "solve_time": out.solve_time, "time_to_success": out.solve_time if status == "success" else None,
        "region_build_time": out.region_build_time, "region_volume": out.region_volume,
        "iterations": out.iterations, "n_evals": out.n_evals, "solver_info": out.solver_info,
        "min_signed_distance": rc.min_signed_distance if rc else None,
        "t_min": rc.t_min if rc else None,
        "window_excess": rc.window_excess if rc else None,
        "path_length": rc.path_length if rc else None,
        "peak_joint_vel": rc.peak_joint_vel if rc else None,
        "recheck_time": rc.elapsed if rc else None, "recheck_dt": recheck_dt,
        "recheck_cap": RECHECK_CAP, "killed": False,
        "total_time": time.perf_counter() - t_start,
        "planner_extras": out.extras,
    })
    assert r["status"] in STATUSES and r["failure_label"] in FAILURE_LABELS
    return _jsonable(r)


def append_result(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(_jsonable(record)) + "\n")


def load_results(path: str | Path) -> list[dict]:
    """Records of a results.jsonl, with milestone method aliases normalized."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                r["method"] = METHOD_ALIASES.get(r.get("method"), r.get("method"))
                out.append(r)
    return out


def done_keys(path: str | Path) -> set[tuple]:
    return {trial_key(r) for r in load_results(path)}
