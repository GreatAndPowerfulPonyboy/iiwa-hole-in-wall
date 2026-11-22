"""Sweep driver: every trial runs in a fresh subprocess with a kill timer, so
a native solve that ignores its iteration cap still counts as a timeout.
Workers write their own record to results.jsonl; the parent writes a timeout
record for a killed trial.  Trials already present in the file are skipped,
so a sweep can be resumed.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from pathlib import Path

from .evaluate import append_result, canonical_method, done_keys, run_trial, timeout_record


@dataclass(frozen=True)
class Trial:
    method: str
    seed: int
    beta: float
    eps: float
    n_walls: int

    @property
    def key(self) -> tuple:
        return (self.method, self.seed, self.beta, self.eps, self.n_walls)


def grid(methods, seeds, betas, eps_values, n_walls_values) -> list[Trial]:
    return [Trial(canonical_method(m), int(s), float(b), float(e), int(n))
            for m, b, n, e, s in itertools.product(methods, betas, n_walls_values, eps_values, seeds)]


def _worker(trial: Trial, out_path: str, kwargs: dict, cap: float | None) -> None:
    t0 = time.perf_counter()
    try:
        if trial.method == "B":                       # Stage B: kwargs are StageBTrial fields
            from .stage_b import StageBTrial, run_stage_b_trial
            r = run_stage_b_trial(StageBTrial(trial.seed, trial.beta, trial.eps, **kwargs))
            r["wall_clock_cap"] = cap
        else:
            r = run_trial(trial.method, trial.seed, trial.beta, trial.eps, trial.n_walls, cap=cap, **kwargs)
    except Exception as e:  # a crash is recorded, not lost; its timing is the real elapsed time
        elapsed = time.perf_counter() - t0
        r = timeout_record(trial.method, trial.seed, trial.beta, trial.eps, trial.n_walls, cap or -1.0)
        r.update({"status": "infeasible", "failure_label": "infeasible_reported", "killed": False,
                  "solve_time": elapsed, "total_time": elapsed, "error": f"{type(e).__name__}: {e}"})
    append_result(out_path, r)


@dataclass
class SweepStats:
    run: int = 0
    skipped: int = 0
    killed: int = 0
    statuses: dict = field(default_factory=dict)


def run_sweep(trials: list[Trial], out_path: str | Path, *, cap: float | dict | None = None,
              workers: int = 4, method_kwargs: dict | None = None, verbose: bool = True,
              poll: float = 0.1) -> SweepStats:
    """Run every trial not already in out_path.  `cap` is the per-trial
    wall-clock cap in seconds (process start to record written), None for
    no cap, or a {method: cap} dict (frozen caps from the M5 calibration;
    a method missing from the dict runs uncapped)."""
    out_path = str(out_path)
    caps = cap if isinstance(cap, dict) else {}
    cap_of = (lambda t: caps.get(t.method)) if isinstance(cap, dict) else (lambda t: cap)
    ctx = mp.get_context("spawn")
    done = done_keys(out_path)
    todo = [t for t in trials if t.key not in done]
    stats = SweepStats(skipped=len(trials) - len(todo))
    if verbose:
        print(f"sweep: {len(todo)} trials to run, {stats.skipped} already done, {workers} workers, cap {cap}")
    active: list[tuple[Trial, mp.Process, float]] = []
    queue = list(todo)
    t_sweep = time.perf_counter()
    while queue or active:
        while queue and len(active) < workers:
            t = queue.pop(0)
            p = ctx.Process(target=_worker, args=(t, out_path, method_kwargs or {}, cap_of(t)), daemon=True)
            p.start()
            active.append((t, p, time.perf_counter()))
        time.sleep(poll)
        still = []
        for t, p, t0 in active:
            if not p.is_alive():
                p.join()
                stats.run += 1
                if verbose:
                    print(f"  done {t.method} seed={t.seed} beta={t.beta} eps={1e3*t.eps:.1f}mm walls={t.n_walls} "
                          f"({time.perf_counter() - t0:.1f}s)  [{stats.run}/{len(todo)}]", flush=True)
            elif cap_of(t) is not None and time.perf_counter() - t0 > cap_of(t):
                p.terminate()
                p.join(2.0)
                if p.is_alive():
                    p.kill()
                    p.join()
                append_result(out_path, timeout_record(t.method, t.seed, t.beta, t.eps, t.n_walls, cap_of(t)))
                stats.run += 1
                stats.killed += 1
                if verbose:
                    print(f"  KILLED {t.method} seed={t.seed} beta={t.beta} eps={1e3*t.eps:.1f}mm walls={t.n_walls} "
                          f"after {cap_of(t)}s  [{stats.run}/{len(todo)}]", flush=True)
            else:
                still.append((t, p, t0))
        active = still
    if verbose:
        print(f"sweep finished in {time.perf_counter() - t_sweep:.0f}s: {stats.run} run, {stats.killed} killed")
    return stats
