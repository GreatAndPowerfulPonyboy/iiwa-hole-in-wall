"""Trial harness: the problem for a seed, the planner call, grading, and
the append-only results file."""

from __future__ import annotations

import importlib.metadata as _md
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .api import Budget, PlanOutcome, PlannerSpec, Services, get_planner, is_path_ref, parse_ref
from .evaluate import FAILURE_LABELS, RECHECK_CAP, RECHECK_DT, STATUSES, classify, recheck
from .scene import Scene, make_scene
from .sequences import Sequence, random_sequence

DRAKE_VERSION = _md.version("drake")

# Every planner draws from planner_rng (scheme "v2"); records converted from
# the earlier format carry "v1".
RNG_SCHEMES = {"connect": "v2", "find": "v2"}
# A planner can stop only at its own check points, so it may finish the step
# in progress after its budget expires; an excess below this is not an overrun.
OVERRUN_SLACK = 0.05        # s


def _budget_fields(budget: float | None, clock: float | None) -> dict:
    """budget_excess (clock - budget, s) and budget_overrun (excess > OVERRUN_SLACK)."""
    excess = None if budget is None or clock is None else clock - budget
    return {"budget_excess": excess, "budget_overrun": excess is not None and excess > OVERRUN_SLACK}


def resolve(ref: str, config=None) -> tuple[PlannerSpec, object]:
    """Resolve a planner reference and its config.

    Parameters
    ----------
    ref : str
        A registered id, ``"path/to/file.py:function"``, either followed by
        ``[key=value,...]`` (a config label is a valid reference).
    config : dict or config instance, optional
        Further options; an option given both here and in `ref` must agree.

    Returns
    -------
    spec : PlannerSpec
    cfg : config instance

    Raises
    ------
    KeyError
        If the id is not registered (the message lists the valid ids).
    ValueError
        If an option is unknown, cannot be converted, or is given twice with
        different values.
    """
    head, overrides = parse_ref(ref)
    spec = get_planner(head)
    if config is not None and isinstance(config, spec.config):
        if overrides:
            cfg_ref = spec.make_config(overrides)
            if any(getattr(cfg_ref, k) != getattr(config, k) for k in overrides):
                raise ValueError(f"{ref!r} disagrees with the config passed with it")
        return spec, config
    extra = dict(config or {})
    if overrides:
        both = {k for k in set(overrides) & set(extra)
                if spec.make_config({k: overrides[k]}) != spec.make_config({k: extra[k]})}
        if both:
            raise ValueError(f"option(s) {sorted(both)} given with different values in {ref!r} and the config")
    return spec, spec.make_config({**overrides, **extra})


def resolve_planner(ref: str) -> PlannerSpec:
    """Return the planner of a reference (see `resolve`)."""
    return resolve(ref)[0]


def trial_rng(seed: int) -> np.random.Generator:
    """Return the problem's random stream (sequence choice and roll draws)."""
    return np.random.default_rng(seed)


def planner_rng(seed: int) -> np.random.Generator:
    """Return the planner's random stream, independent of the problem's.

    Parameters
    ----------
    seed : int
        Trial seed.

    Returns
    -------
    numpy.random.Generator

    Notes
    -----
    The problem stream is seeded with ``SeedSequence(seed)``; this is its
    first spawned child, ``SeedSequence(seed, spawn_key=(0,))``, whose state
    is derived from a different input and is independent of it.  It is the
    same for every planner, so paired comparisons share their random inputs
    too.
    """
    return np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])


def trial_sequence(seed: int, beta: float, eps: float, n_walls: int) -> Sequence | None:
    """Return the problem of a trial.

    Parameters
    ----------
    seed : int
        Trial seed.
    beta : float
        Roll band in radians.
    eps : float
        Clearance in meters.
    n_walls : int
        Number of walls.

    Returns
    -------
    Sequence or None
        The same poses and roll draws for every planner and every eps (a
        function of seed, beta and n_walls only); None if no
        transit-feasible sequence was found.
    """
    scene = make_scene([], beta=beta)
    return random_sequence(n_walls, trial_rng(seed), beta, eps, scene)


def plan_with(method: str, seq: Sequence, scene: Scene, seed: int, config=None) -> PlanOutcome:
    """Plan `seq` with a connect planner, its resolved config and the planner stream of `seed`."""
    spec, cfg = resolve(method, config)
    return spec.func(seq, Services(seq, scene, drake_seed=seed), Budget(None), planner_rng(seed), cfg)


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
    """Return the key that identifies a trial in a results file.

    Parameters
    ----------
    r : dict
        A record.

    Returns
    -------
    tuple
        (method, config_label, seed, beta, eps, n_walls).
    """
    return (r["method"], r["config_label"], int(r["seed"]), float(r["beta"]), float(r["eps"]), int(r["n_walls"]))


def base_record(method: str, seed: int, beta: float, eps: float, n_walls: int, config=None) -> dict:
    """Return the fields every record of a trial carries, before it runs."""
    spec, cfg = resolve(method, config)
    return {"task": spec.task, "method": spec.id, "config_label": spec.config_label(cfg),
            "seed": int(seed), "beta": float(beta), "eps": float(eps), "n_walls": int(n_walls),
            "planner_config": spec.config_dict(cfg), "planner_source": spec.source(),
            "drake_version": DRAKE_VERSION, "hitw_commit": _commit(), "rng_scheme": RNG_SCHEMES[spec.task],
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def timeout_record(method: str, seed: int, beta: float, eps: float, n_walls: int, cap: float | None, config=None, *,
                   planner_time: float | None = None, startup_time: float | None = None,
                   kill_timer: float | None = None) -> dict:
    """Return the record the sweep driver writes when a trial's process is killed.

    Parameters
    ----------
    method, seed, beta, eps, n_walls, config
        The trial.
    cap : float or None
        The planner's budget in seconds.
    planner_time : float, optional
        Seconds on the budget clock when the process was killed; None if
        the planner had not started.  Recorded as solve_time and
        planner_wall_clock.
    startup_time : float, optional
        Seconds from process start to the planner call, if it was reached.
    kill_timer : float, optional
        Seconds after process start at which the process was killed.

    Returns
    -------
    dict
    """
    r = base_record(method, seed, beta, eps, n_walls, config)
    r.update({"status": "timeout", "failure_label": "timeout", "solve_time": planner_time,
              "planner_wall_clock": planner_time, "startup_time": startup_time, "budget": cap,
              **_budget_fields(cap, planner_time), "kill_timer": kill_timer,
              "time_to_success": None, "wall_clock_cap": cap, "killed": True,
              "region_build_time": None, "region_volume": None, "iterations": None, "n_evals": None,
              "min_signed_distance": None, "path_length": None, "peak_joint_vel": None,
              "roll_draw": None, "sequence": None})
    return r


def run_trial(method: str, seed: int, beta: float, eps: float, n_walls: int, *,
              seq: Sequence | None = None, scene: Scene | None = None,
              recheck_dt: float = RECHECK_DT, cap: float | None = None, config=None,
              t_process_start: float | None = None, on_planner_start=None) -> dict:
    """Run one connect trial and return its JSON-ready record.

    Build (or take) the problem, call the planner with its resolved config,
    the trial's services, a budget of `cap` seconds (its clock starts
    immediately before the call) and the planner random stream, then
    re-check densely and classify.

    Parameters
    ----------
    method : str
        Planner reference: a registered id or ``"path/to/file.py:function"``,
        optionally with ``[key=value,...]``.
    seed : int
        Trial seed: the problem is a function of (seed, beta, n_walls).
    beta : float
        Roll band in radians.
    eps : float
        Clearance in meters.
    n_walls : int
        Number of walls.
    seq : Sequence, optional
        A prebuilt problem; None builds it from the seed.
    scene : Scene, optional
        The prismatic-wall scene of `seq`; None builds it.
    recheck_dt : float
        Re-check grid step in seconds.
    cap : float or None
        The planner's wall-clock budget in seconds; None means unlimited.
        Planners check it where they can; the sweep driver enforces it by
        killing the process at cap plus a startup grace.
    config : dict or config instance, optional
        The planner's options; None means its defaults.
    t_process_start : float, optional
        ``time.time()`` when the trial's process started, so startup_time
        covers spawn and imports; None measures from the call.
    on_planner_start : callable, optional
        Called with ``time.time()`` immediately before the planner.

    Returns
    -------
    dict
        The trial record.  ``status`` is "unsupported" (and nothing runs)
        for a planner registered with ``supports_beta_positive=False`` at
        beta > 0.

    Raises
    ------
    KeyError
        If the planner id is not registered.
    ValueError
        If the planner is not a connect planner or an option is invalid;
        raised before the problem is built.
    TypeError
        If the planner does not return a `PlanOutcome`, or its q_of_t does
        not return a finite 7-vector at t = 0, T / 2 and T.

    Examples
    --------
    >>> r = run_trial("gcs-box", seed=0, beta=0.0, eps=0.02, n_walls=1)
    >>> r["status"], r["config_label"], r["task"]
    ('success', 'gcs-box', 'connect')
    """
    t_entry = time.time()
    spec, cfg = resolve(method, config)
    if spec.task != "connect":
        raise ValueError(f"{spec.id} is a {spec.task} planner; run_trial runs connect planners")
    r = base_record(method, seed, beta, eps, n_walls, cfg)
    r["wall_clock_cap"] = cap
    r["budget"] = cap
    if beta > 0 and not spec.supports_beta_positive:
        r.update({"status": "unsupported", "failure_label": None, "solver_status": None, "killed": False,
                  "note": f"{spec.id} is registered for beta = 0 only"})
        return _jsonable(r)
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
    svc = Services(seq, scene, drake_seed=seed)
    rng = planner_rng(seed)
    t_planner = time.time()
    r["startup_time"] = t_planner - (t_process_start if t_process_start is not None else t_entry)
    if on_planner_start is not None:
        on_planner_start(t_planner)
    budget = Budget(cap)
    out = spec.func(seq, svc, budget, rng, cfg)
    r["planner_wall_clock"] = budget.elapsed()
    r.update(_budget_fields(cap, r["planner_wall_clock"]))
    if not isinstance(out, PlanOutcome):
        raise TypeError(f"{spec.id} returned {type(out).__name__}, not PlanOutcome")
    if out.q_of_t is not None:
        for t in (0.0, 0.5 * seq.T, seq.T):
            q = np.asarray(out.q_of_t(t))
            if q.shape != (7,) or not np.all(np.isfinite(q)):
                raise TypeError(f"{spec.id}: q_of_t({t:g}) is {q.shape if q.shape != (7,) else 'not finite'}; "
                                f"it must return a finite ndarray of shape (7,) on [0, T]")
    rc = recheck(out.q_of_t, seq, scene, dt=recheck_dt) if out.q_of_t is not None else None
    status, label = classify(out.status, rc)
    r.update({
        "status": status, "failure_label": label, "solver_status": out.status,
        "solve_time": out.solve_time,
        "time_to_success": out.solve_time + (out.region_build_time or 0.0) if status == "success" else None,
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


def find_trial(method: str, seed: int, beta: float, eps: float, *, config=None, cap: float | None = None,
               budget: float | None = None, budget_source: str | None = None,
               t_process_start: float | None = None, on_planner_start=None) -> dict:
    """Run one find trial and return its JSON-ready record.

    The finder sees the wall of the 1-wall problem of (seed, beta) at eps
    through a `WallView` and the wall's services, draws from the planner
    stream, and only proposes candidates.  The harness validates each one
    (`find.validate`, the find task's success standard), counts attempts,
    timestamps the first valid one on the budget clock, and stops asking for
    candidates once the budget expires or the finder is exhausted.

    Parameters
    ----------
    method : str
        Find planner reference: a registered id or ``"path/to/file.py:function"``,
        optionally with ``[key=value,...]``.
    seed : int
        Trial seed.
    beta : float
        Roll band in radians.
    eps : float
        Clearance in meters.
    config : dict or config instance, optional
        The finder's options; None means its defaults.
    cap : float or None
        Wall-clock budget in seconds; None means unlimited.
    budget : float, optional
        A budget that replaces `cap` (equal compute with a reference finder).
    budget_source : str, optional
        Where `budget` came from, recorded as is; defaults to "cap" or
        "unlimited".
    t_process_start : float, optional
        ``time.time()`` when the trial's process started.
    on_planner_start : callable, optional
        Called with ``time.time()`` immediately before the finder.

    Returns
    -------
    dict
        The trial record: attempts, valid attempts, search time, time and
        attempt of the first valid candidate, up to five valid candidates
        and their distance to the hidden configuration, why the search
        stopped ("budget" or "exhausted"), and per attempt the finder's info
        when it yields any.

    Raises
    ------
    ValueError
        If the planner is not a find planner or an option is invalid.
    TypeError
        If a candidate is not a 7-vector.
    """
    from .api import WallView
    from .find import validate

    t_entry = time.time()
    spec, cfg = resolve(method, config)
    if spec.task != "find":
        raise ValueError(f"{spec.id} is a {spec.task} planner; find_trial runs find planners")
    wall_clock = budget if budget is not None else cap
    r = base_record(method, seed, beta, eps, 1, cfg)
    r.update({"wall_clock_cap": cap, "budget": wall_clock,
              "budget_source": budget_source or ("cap" if cap is not None else "unlimited")})
    if beta > 0 and not spec.supports_beta_positive:
        r.update({"status": "unsupported", "killed": False, "note": f"{spec.id} is registered for beta = 0 only"})
        return _jsonable(r)
    t_start = time.perf_counter()
    seq = trial_sequence(seed, beta, eps, 1)
    if seq is None:
        r.update({"status": "infeasible", "killed": False, "note": "no sequence"})
        return _jsonable(r)
    c, wall = seq.crossings[0], seq.walls[0]
    check_scene = make_scene([wall], beta=beta)
    valid_star, d_star = validate(check_scene, c.q_star, beta)
    r.update({"sequence": [c.name], "roll_draw": [c.roll_draw], "q_star": c.q_star, "wall_boxes": wall.n_boxes,
              "q_star_valid": valid_star, "q_star_min_distance": d_star})
    view = WallView.of(wall, beta, eps, seq.speed)
    svc = Services.for_wall(view, drake_seed=seed)
    rng = planner_rng(seed)
    q_star_free = check_scene.q_free(c.q_star)
    n = n_valid = n_reported = n_false = 0
    first, valid_qs, dists, attempts = None, [], [], []
    stopped = "exhausted"
    t_planner = time.time()
    r["startup_time"] = t_planner - (t_process_start if t_process_start is not None else t_entry)
    if on_planner_start is not None:
        on_planner_start(t_planner)
    clock = Budget(wall_clock)
    candidates = spec.func(view, svc, clock, rng, cfg)
    try:
        while True:
            if clock.expired():
                stopped = "budget"
                break
            try:
                item = next(candidates)
            except StopIteration:
                break
            q, info = item if isinstance(item, tuple) else (item, None)
            q = np.asarray(q, dtype=float).ravel()
            if q.shape != (7,):
                raise TypeError(f"{spec.id} proposed a candidate of shape {q.shape}, not (7,)")
            n += 1
            valid, d = validate(check_scene, q, beta)
            if valid:
                n_valid += 1
                if first is None:
                    first = {"attempt": n, "time": clock.elapsed(), "min_distance": d}
                if len(valid_qs) < 5:
                    valid_qs.append(q.tolist())
                dists.append(float(np.linalg.norm(check_scene.q_free(q) - q_star_free)))
            if info is not None:
                attempts.append({**info, "valid": valid, "min_distance": d, "q": q.tolist()})
                if info.get("reported_success"):
                    n_reported += 1
                    n_false += not valid
    finally:
        candidates.close()
    search = clock.elapsed()
    r.update({"status": "success" if n_valid else "infeasible", "killed": False,
              "n_attempts": n, "n_valid": n_valid, "search_time": search, "planner_wall_clock": search,
              **_budget_fields(wall_clock, search),
              "time_to_first_valid": first["time"] if first else None,
              "first_valid_attempt": first["attempt"] if first else None,
              "first_valid_min_distance": first["min_distance"] if first else None,
              "attempts_per_s": n / search if search > 0 else None, "stopped": stopped,
              "valid_qs": valid_qs, "valid_dist_to_q_star": dists,
              "total_time": time.perf_counter() - t_start})
    if attempts:
        r["attempts"] = attempts
        if any("reported_success" in a for a in attempts):
            r.update({"n_reported_success": n_reported, "n_false_positive": n_false})
    return _jsonable(r)


def append_result(path: str | Path, record: dict) -> None:
    """Append one record to a results file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(_jsonable(record)) + "\n")


REQUIRED_FIELDS = ("task", "method", "config_label", "planner_config", "rng_scheme")


def load_results(path: str | Path) -> list[dict]:
    """Return the records of a results file.

    Parameters
    ----------
    path : str or Path
        A results file; a missing file has no records.

    Returns
    -------
    list of dict

    Raises
    ------
    ValueError
        If a record lacks any of `REQUIRED_FIELDS`; files in the earlier
        record format are converted with ``scripts/migrate_results.py``.
    """
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            missing = [k for k in REQUIRED_FIELDS if k not in r]
            if missing:
                raise ValueError(f"{path}:{n}: record of method {r.get('method')!r} lacks {missing}: it is not "
                                 f"in the current record format; convert the file with scripts/migrate_results.py")
            out.append(r)
    return out


def done_keys(path: str | Path) -> set[tuple]:
    """Return the trial keys already present in a results file."""
    return {trial_key(r) for r in load_results(path)}


def planner_ref(ref: str) -> str:
    """Return the planner part of `ref` in the form a subprocess can load: an id, or an absolute path reference."""
    head, _ = parse_ref(ref)
    if is_path_ref(head):
        file, _, fname = head.rpartition(":")
        return f"{Path(file).resolve()}:{fname}"
    return get_planner(head).id
