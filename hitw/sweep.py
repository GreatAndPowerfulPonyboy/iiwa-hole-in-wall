"""Sweep driver: every trial runs in a fresh subprocess with a kill timer, so
a native solve that ignores its iteration cap still counts as a timeout.
Workers write their own records to the results file; the parent writes a
timeout record for a killed trial.  Trials already present in the file are
skipped, so a sweep can be resumed.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .harness import (append_result, done_keys, find_trial, load_results, planner_ref, resolve, run_trial,
                      timeout_record)

STARTUP_GRACE = 30.0


@dataclass(frozen=True)
class Trial:
    """One cell of the grid.

    Attributes
    ----------
    method : str
        Planner id.
    config_label : str
        Id plus the options that differ from their defaults.
    seed, beta, eps, n_walls
        The problem (beta in radians, eps in meters).
    planner : str
        Reference a subprocess loads the planner by: an id or an absolute
        ``path.py:function``.
    config : tuple of (str, value)
        The resolved config, field by field.
    task : str
        "connect" or "find".
    budget_from : str or None
        Config label of a reference run whose median search time on the same
        (seed, beta, eps) is this trial's budget (equal compute); None means
        the sweep's cap.
    """

    method: str
    config_label: str
    seed: int
    beta: float
    eps: float
    n_walls: int
    planner: str
    config: tuple
    task: str
    budget_from: str | None = None

    @property
    def key(self) -> tuple:
        return (self.method, self.config_label, self.seed, self.beta, self.eps, self.n_walls)


def grid(entries, seeds, betas, eps_values, n_walls_values, *, budget_from: dict | None = None) -> list[Trial]:
    """Return the trials of a full grid.

    Parameters
    ----------
    entries : list of str or (str, dict)
        Planner references, each optionally with options: ``"kto"``,
        ``"iris-gcs[iris=np2]"``, ``("rrt-connect", {"max_samples": 100})``
        or ``"path/to/file.py:function"``.  Each entry is a separate run, so
        one planner can appear with several configs.
    seeds : list of int
    betas : list of float
        Roll bands in radians.
    eps_values : list of float
        Clearances in meters.
    n_walls_values : list of int
        Wall counts for connect planners; find planners always use 1 wall.
    budget_from : dict, optional
        {config label or method id: reference config label}: give those
        find runs the reference run's median search time on the same
        (seed, beta, eps) as their budget, e.g.
        ``{"uniform-sampling": "ik-restarts"}``.  The reference must be a
        find run of the grid.

    Returns
    -------
    list of Trial

    Raises
    ------
    KeyError, ValueError
        If a reference or an option is invalid, two entries resolve to the
        same config label, or a `budget_from` entry does not name two
        different find runs of the grid.
    """
    resolved, labels = [], {}
    for entry in entries:
        ref, options = (entry, None) if isinstance(entry, str) else entry
        spec, cfg = resolve(ref, options)
        label = spec.config_label(cfg)
        if label in labels:
            raise ValueError(f"two grid entries resolve to {label!r}")
        labels[label] = spec
        resolved.append((spec, label, planner_ref(ref), tuple(spec.config_dict(cfg).items())))
    budget_from = dict(budget_from or {})
    finds = {lab for lab, spec in labels.items() if spec.task == "find"}
    for who, ref in budget_from.items():
        targets = [lab for lab in finds if lab == who or labels[lab].id == who]
        if not targets or ref not in finds or ref in targets:
            raise ValueError(f"budget_from {who!r} -> {ref!r}: both must be different find runs of the grid "
                             f"(find runs: {sorted(finds)})")
    trials = []
    for (spec, label, ref, cfg), b, n, e, s in itertools.product(resolved, betas, n_walls_values, eps_values, seeds):
        if spec.task == "find" and n != n_walls_values[0]:
            continue
        trials.append(Trial(spec.id, label, int(s), float(b), float(e), 1 if spec.task == "find" else int(n),
                            ref, cfg, spec.task, budget_from.get(label, budget_from.get(spec.id))))
    return trials


def _crash_record(t: Trial, cap: float | None, elapsed: float, e: Exception) -> dict:
    r = timeout_record(t.planner, t.seed, t.beta, t.eps, t.n_walls, cap, dict(t.config))
    r.update({"status": "infeasible", "failure_label": "infeasible_reported", "killed": False,
              "solve_time": elapsed, "total_time": elapsed, "error": f"{type(e).__name__}: {e}"})
    return r


def _worker(t: Trial, out_path: str, cap: float | None, budget: float | None, budget_source: str | None,
            t_spawn: float, started) -> None:
    t0 = time.perf_counter()
    resolve(t.planner, dict(t.config))            # a bad option is an error, never a trial outcome

    def mark(when: float) -> None:
        started.value = when

    try:
        if t.task == "find":
            r = find_trial(t.planner, t.seed, t.beta, t.eps, config=dict(t.config), cap=cap, budget=budget,
                           budget_source=budget_source, t_process_start=t_spawn, on_planner_start=mark)
        else:
            r = run_trial(t.planner, t.seed, t.beta, t.eps, t.n_walls, cap=cap, config=dict(t.config),
                          t_process_start=t_spawn, on_planner_start=mark)
    except Exception as e:  # a crash is recorded, not lost; its timing is the real elapsed time
        r = _crash_record(t, cap, time.perf_counter() - t0, e)
    append_result(out_path, r)


def reference_budget(records: list[dict], t: Trial) -> float | None:
    """Return the median search time of run `t.budget_from` on t's (seed, beta, eps), or None if there is none."""
    times = [r.get("search_time", r.get("planner_wall_clock")) for r in records
             if r["config_label"] == t.budget_from and int(r["seed"]) == t.seed
             and float(r["beta"]) == t.beta and float(r["eps"]) == t.eps]
    times = [x for x in times if x is not None]
    return float(np.median(times)) if times else None


@dataclass
class SweepStats:
    run: int = 0
    skipped: int = 0
    killed: int = 0
    no_reference: int = 0
    statuses: dict = field(default_factory=dict)


def run_sweep(trials: list[Trial], out_path: str | Path, *, cap: float | dict | None = None,
              startup_grace: float = STARTUP_GRACE, workers: int = 4, verbose: bool = True,
              poll: float = 0.1) -> SweepStats:
    """Run every trial not already in `out_path`, each in its own subprocess.

    Parameters
    ----------
    trials : list of Trial
        The grid (see `grid`).
    out_path : str or Path
        Results file; records are appended, trials already in it are skipped.
    cap : float, dict or None
        The planner's wall-clock budget in seconds, handed to it as its
        `Budget`; a dict maps a config label or, failing that, a method id
        to its budget (missing entries run unlimited); None for unlimited.
        A trial with ``budget_from`` uses its reference's time instead.
    startup_grace : float
        Seconds added to the budget for everything else in the process
        (spawn, imports, problem and scene build, the dense re-check); the
        process is killed at budget + startup_grace after it starts.
    workers : int
        Concurrent processes.
    verbose : bool
        Print one line per finished trial.
    poll : float
        Seconds between process checks.

    Returns
    -------
    SweepStats
        Trials run, skipped, killed, and left for want of a reference time
        (``no_reference``; rerun the sweep once those references exist).

    Raises
    ------
    ValueError
        Before anything runs, if `cap` names a label or method not in
        `trials`.

    Notes
    -----
    Trials with ``budget_from`` run after all the others, reading their
    reference's records from `out_path`.
    """
    out_path = str(out_path)
    caps = cap if isinstance(cap, dict) else {}
    known = {t.config_label for t in trials} | {t.method for t in trials}
    stray = sorted(set(caps) - known)
    if stray:
        raise ValueError(f"caps name {stray}, which are neither a config label nor a method of the trials")

    def cap_of(t):
        return caps.get(t.config_label, caps.get(t.method)) if isinstance(cap, dict) else cap

    ctx = mp.get_context("spawn")
    done = done_keys(out_path)
    todo = [t for t in trials if t.key not in done]
    stats = SweepStats(skipped=len(trials) - len(todo))
    if verbose:
        print(f"sweep: {len(todo)} trials to run, {stats.skipped} already done, {workers} workers, cap {cap}")
    t_sweep = time.perf_counter()
    for dependent in (False, True):
        queue = [(t, None) for t in todo if (t.budget_from is not None) == dependent]
        if dependent and queue:
            records = load_results(out_path)
            ready = []
            for t, _ in queue:
                b = reference_budget(records, t)
                if b is None:
                    stats.no_reference += 1
                    if verbose:
                        print(f"  WAIT {t.config_label} seed={t.seed} beta={t.beta} eps={1e3*t.eps:.1f}mm: "
                              f"no {t.budget_from} record to take its budget from", flush=True)
                else:
                    ready.append((t, b))
            queue = ready
        _run_queue(queue, out_path, cap_of, startup_grace, workers, verbose, poll, ctx, stats, len(todo))
    if verbose:
        print(f"sweep finished in {time.perf_counter() - t_sweep:.0f}s: {stats.run} run, {stats.killed} killed"
              + (f", {stats.no_reference} waiting for a reference" if stats.no_reference else ""))
    return stats


def _run_queue(queue, out_path, cap_of, startup_grace, workers, verbose, poll, ctx, stats, n_todo):
    """Run (trial, budget or None) pairs in subprocesses under their kill timers."""
    active = []

    def name(t):
        return f"{t.config_label} seed={t.seed} beta={t.beta} eps={1e3*t.eps:.1f}mm walls={t.n_walls}"

    while queue or active:
        while queue and len(active) < workers:
            t, budget = queue.pop(0)
            source = f"equal:{t.budget_from}" if budget is not None else None
            wall_clock = budget if budget is not None else cap_of(t)
            kill = None if wall_clock is None else wall_clock + startup_grace
            started = ctx.Value("d", 0.0)
            t_spawn = time.time()
            p = ctx.Process(target=_worker, args=(t, out_path, cap_of(t), budget, source, t_spawn, started),
                            daemon=True)
            p.start()
            active.append((t, p, time.perf_counter(), t_spawn, started, wall_clock, kill))
        time.sleep(poll)
        still = []
        for t, p, t0, t_spawn, started, wall_clock, kill in active:
            if not p.is_alive():
                p.join()
                stats.run += 1
                if verbose:
                    print(f"  done {name(t)} ({time.perf_counter() - t0:.1f}s)  [{stats.run}/{n_todo}]", flush=True)
            elif kill is not None and time.perf_counter() - t0 > kill:
                p.terminate()
                p.join(2.0)
                if p.is_alive():
                    p.kill()
                    p.join()
                now, t_planner = time.time(), started.value or None
                append_result(out_path, timeout_record(
                    t.planner, t.seed, t.beta, t.eps, t.n_walls, wall_clock, dict(t.config),
                    planner_time=(now - t_planner) if t_planner else None,
                    startup_time=(t_planner - t_spawn) if t_planner else None, kill_timer=kill))
                stats.run += 1
                stats.killed += 1
                if verbose:
                    print(f"  KILLED {name(t)} after {kill}s  [{stats.run}/{n_todo}]", flush=True)
            else:
                still.append((t, p, t0, t_spawn, started, wall_clock, kill))
        active = still
