"""The hitw command line: list, check, run, sweep, analyze and demo planners.

Run:  python3 hitw <command> --help      (from the repository root, with the venv active)
"""

from __future__ import annotations

import argparse
import dataclasses
import multiprocessing as mp
import sys
import time
import tomllib
from pathlib import Path

CHECK_SEED = 0
CHECK_EPS = 0.01
CHECK_BETAS = (0.0, 0.1)
CHECK_BUDGET = 60.0
OVERRUN_FACTOR = 2.0


def parse_seeds(spec: str | int | list) -> list[int]:
    """Read seeds given as ``"0-99"``, ``"0,1,5"``, ``"0-4,10"``, an int or a list of ints."""
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, list):
        return [int(s) for s in spec]
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def parse_options(items: list[str] | None) -> dict:
    """Read ``key=value`` strings into a dict (values stay strings; the planner's config converts them)."""
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--config expects key=value, got {item!r}")
        out[key.strip()] = value.strip()
    return out


def _load(files: list[str] | None) -> None:
    from .api import load_planner_file
    for f in files or []:
        load_planner_file(f)


# --- list ---------------------------------------------------------------------

def cmd_list(args) -> int:
    """Print the registered planners by task with their options and defaults."""
    from .api import load_errors, planners
    from .config import config_fields
    _load(args.load)
    for name, error in load_errors().items():
        print(f"planners/{name} failed to import and is skipped: {error}")
    for task in ("connect", "find"):
        print(f"{task}:")
        for spec in planners(task):
            beta = "" if spec.supports_beta_positive else "  (beta = 0 only)"
            print(f"  {spec.id:20s} {spec.display_name}{beta}    [{spec.func.__module__}]")
            for name, default in config_fields(spec.config).items():
                print(f"      {name} = {default!r}")
    return 0


# --- check --------------------------------------------------------------------

def _check_one(ref: str, options: dict, task: str, beta: float, budget: float, conn) -> None:
    from .harness import find_trial, run_trial
    try:
        if task == "connect":
            r = run_trial(ref, CHECK_SEED, beta, CHECK_EPS, 1, cap=budget, config=options)
        else:
            r = find_trial(ref, CHECK_SEED, beta, CHECK_EPS, cap=budget, config=options)
        conn.send(("ok", r))
    except BaseException as e:  # reported to the parent, which decides the verdict
        conn.send(("error", f"{type(e).__name__}: {e}"))


def cmd_check(args) -> int:
    """Run a planner on two fixed problems, print the dense re-check or validity, and judge the call."""
    from .harness import planner_ref, resolve
    spec, cfg = resolve(args.planner, parse_options(args.config))
    ref, options = planner_ref(args.planner), dataclasses.asdict(cfg)
    print(f"check {spec.config_label(cfg)} ({spec.display_name}, task {spec.task}): seed {CHECK_SEED}, "
          f"eps {1e3 * CHECK_EPS:g} mm, 1 wall, budget {args.budget:g} s; nothing is written")
    ctx = mp.get_context("spawn")
    failed = False
    for beta in CHECK_BETAS:
        parent, child = ctx.Pipe(duplex=False)
        p = ctx.Process(target=_check_one, args=(ref, options, spec.task, beta, args.budget, child), daemon=True)
        t0 = time.perf_counter()
        p.start()
        limit = OVERRUN_FACTOR * args.budget + args.startup_grace
        got = parent.recv() if parent.poll(limit) else None
        p.join(5.0)
        if p.is_alive():
            p.kill()
        head = f"  beta = {beta:g}:"
        if got is None:
            print(f"{head} FAIL: no result after {limit:g} s (more than {OVERRUN_FACTOR:g}x the budget)")
            failed = True
            continue
        kind, r = got
        if kind == "error":
            print(f"{head} FAIL: {r}")
            failed = True
            continue
        if r["status"] == "unsupported":
            print(f"{head} unsupported ({r.get('note')}); not a failure")
            continue
        clock = r.get("planner_wall_clock") or 0.0
        if spec.task == "connect":
            d = r["min_signed_distance"]
            print(f"{head} {r['status']}" + (f" ({r['failure_label']})" if r["failure_label"] else "")
                  + f"; planner said {r['solver_status']}; dense re-check min distance "
                  + (f"{1e3 * d:+.2f} mm at t = {r['t_min']:.2f} s, window excess {r['window_excess']:+.1e} rad"
                     if d is not None else "n/a (no trajectory)")
                  + f"; planner {clock:.2f} s of {args.budget:g} s")
        else:
            print(f"{head} {r['n_valid']} valid of {r['n_attempts']} candidates (stopped: {r['stopped']}); "
                  f"first valid at attempt {r['first_valid_attempt']}; hidden pose valid {r['q_star_valid']}; "
                  f"search {clock:.2f} s of {args.budget:g} s")
        if clock > OVERRUN_FACTOR * args.budget:
            print(f"{head} FAIL: the planner used {clock:.2f} s, more than {OVERRUN_FACTOR:g}x its budget")
            failed = True
        print(f"      ({time.perf_counter() - t0:.1f} s with startup and grading)")
    print("check FAILED" if failed else "check passed")
    return 1 if failed else 0


# --- run and sweep ------------------------------------------------------------

def cmd_run(args) -> int:
    """Run one planner over a grid of seeds into a results file."""
    from .harness import resolve
    from .sweep import grid, run_sweep
    options = parse_options(args.config)
    spec, cfg = resolve(args.planner, options)
    trials = grid([(args.planner, options)], parse_seeds(args.seeds), args.beta, args.eps, args.walls)
    if args.budget_from:
        trials = [dataclasses.replace(t, budget_from=args.budget_from) for t in trials]
    run_sweep(trials, args.out, cap=args.cap, startup_grace=args.startup_grace, workers=args.workers)
    print(f"results in {args.out}")
    return 0


def load_experiment(path: str | Path) -> dict:
    """Read an experiment file into its trials, caps and settings.

    Parameters
    ----------
    path : str or Path
        TOML file with top-level ``out``, optional ``workers`` and
        ``startup_grace``, a ``[grid]`` table (``seeds``, ``betas``,
        ``eps``, ``n_walls``) and one ``[[run]]`` table per run with
        ``planner``, optional ``config`` (a table of options), ``cap``
        (seconds), ``budget_from`` (a reference run's config label) and any
        grid key to override for this run.

    Returns
    -------
    dict
        trials, caps ({config label: seconds}), out, workers, startup_grace
        and per-run trial counts.

    Raises
    ------
    ValueError
        If a run is malformed, two runs share a config label, or a
        budget_from names no find run of the file.
    """
    from .sweep import grid
    spec = tomllib.loads(Path(path).read_text())
    base = spec.get("grid", {})
    allowed = {"planner", "config", "cap", "budget_from", "seeds", "betas", "eps", "n_walls"}
    trials, caps, counts, labels = [], {}, {}, {}
    for i, run in enumerate(spec.get("run", [])):
        extra = set(run) - allowed
        if "planner" not in run or extra:
            raise ValueError(f"{path}: run {i + 1} needs 'planner' and takes only {sorted(allowed)}; got {sorted(run)}")
        g = {k: run.get(k, base.get(k)) for k in ("seeds", "betas", "eps", "n_walls")}
        missing = [k for k, v in g.items() if v is None]
        if missing:
            raise ValueError(f"{path}: run {i + 1} ({run['planner']}) has no {missing} in [grid] or the run")
        ts = grid([(run["planner"], run.get("config", {}))], parse_seeds(g["seeds"]), [float(b) for b in g["betas"]],
                  [float(e) for e in g["eps"]], [int(n) for n in g["n_walls"]])
        label = ts[0].config_label
        if label in labels:
            raise ValueError(f"{path}: runs {labels[label]} and {i + 1} are both {label!r}")
        labels[label] = i + 1
        if "budget_from" in run:
            ts = [dataclasses.replace(t, budget_from=run["budget_from"]) for t in ts]
        if "cap" in run:
            caps[label] = float(run["cap"])
        trials += ts
        counts[label] = len(ts)
    finds = {t.config_label for t in trials if t.task == "find"}
    for t in trials:
        if t.budget_from is not None and (t.task != "find" or t.budget_from not in finds):
            raise ValueError(f"{path}: {t.config_label} takes its budget from {t.budget_from!r}, "
                             f"which is not a find run of the file (find runs: {sorted(finds)})")
    return {"trials": trials, "caps": caps, "counts": counts, "out": spec.get("out"),
            "workers": int(spec.get("workers", 4)), "startup_grace": float(spec.get("startup_grace", 30.0))}


def cmd_sweep(args) -> int:
    """Run the trials of an experiment file."""
    from .sweep import run_sweep
    exp = load_experiment(args.experiment)
    out = args.out or exp["out"]
    if not out:
        raise SystemExit(f"{args.experiment} has no 'out'; pass --out")
    print(f"{args.experiment}: {len(exp['trials'])} trials -> {out}")
    sources = {t.config_label: t.budget_from for t in exp["trials"]}
    for label, n in exp["counts"].items():
        cap = exp["caps"].get(label)
        budget = (f"equal to {sources[label]}'s search time" if sources[label]
                  else "unlimited" if cap is None else f"{cap:g} s")
        print(f"  {label:32s} {n:6d} trials, budget {budget}")
    if args.dry_run:
        return 0
    run_sweep(exp["trials"], out, cap=exp["caps"], startup_grace=exp["startup_grace"],
              workers=args.workers or exp["workers"])
    return 0


# --- analyze ------------------------------------------------------------------

def load_planner_sources(path: Path) -> None:
    """Import the planner files named by a results file's records, so their display names are known."""
    import json

    from .api import _REPO_ROOT, load_planner_file
    seen = set()
    for line in path.read_text().splitlines():
        src = (json.loads(line).get("planner_source") or {}) if line.strip() else {}
        f = src.get("path")
        if not f or f in seen or src.get("module", "").startswith("hitw."):
            continue
        seen.add(f)
        file = Path(f) if Path(f).is_absolute() else _REPO_ROOT / f
        if file.is_file():
            try:
                load_planner_file(file)
            except Exception as e:  # a planner that no longer imports still has its records analysed
                print(f"note: could not import {file} for its display name ({type(e).__name__}: {e})")

def cmd_analyze(args) -> int:
    """Print and write the summary tables of a results file."""
    import pandas as pd

    from .analysis import find_frame, paired_table, results_frame, summarize, summarize_find
    path = Path(args.results)
    load_planner_sources(path)
    out = Path(args.out) if args.out else path.parent
    out.mkdir(parents=True, exist_ok=True)
    written = []
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    df = results_frame(path)
    if len(df):
        summ = summarize(df)
        cols = ["display", "beta", "n_walls", "eps", "n", "k", "p", "lo", "hi", "solve_median", "min_d_min",
                "n_unsupported"]
        print(summ[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
        written += [out / f"{path.stem}_summary.csv", out / f"{path.stem}_paired.csv"]
        summ.to_csv(written[-2], index=False)
        paired_table(df).to_csv(written[-1], index=False)
    fd = find_frame(path)
    if len(fd):
        fs = summarize_find(fd)
        print(fs[["display", "beta", "eps", "trials", "valid", "attempts", "p", "lo", "hi", "upper95",
                  "p_within_K"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
        written.append(out / f"{path.stem}_find_summary.csv")
        fs.to_csv(written[-1], index=False)
    if not written:
        print(f"{path}: no records")
        return 1
    print("wrote:\n" + "\n".join(f"  {p}" for p in written))
    return 0


# --- demo ---------------------------------------------------------------------

def cmd_demo(args) -> int:
    """Plan one connect problem, re-check it densely and play it in Meshcat."""
    from . import viz
    from .evaluate import classify, recheck
    from .harness import plan_with, resolve, trial_sequence
    from .scene import make_scene
    spec, cfg = resolve(args.planner, parse_options(args.config))
    if spec.task != "connect":
        raise SystemExit(f"{spec.id} is a {spec.task} planner; demo plays connect planners")
    if viz.HEADLESS:
        raise SystemExit("HEADLESS is set; unset it to start Meshcat")
    seq = trial_sequence(args.seed, args.beta, args.eps, args.walls)
    if seq is None:
        raise SystemExit("no transit-feasible problem for this seed")
    scene = make_scene(seq.walls, beta=args.beta)
    print(f"{spec.config_label(cfg)} on seed {args.seed}: poses {[c.name for c in seq.crossings]}, "
          f"beta {args.beta:g}, eps {1e3 * args.eps:g} mm, T = {seq.T:g} s")
    out = plan_with(args.planner, seq, scene, args.seed, parse_options(args.config))
    rc = recheck(out.q_of_t, seq, scene) if out.q_of_t is not None else None
    status, label = classify(out.status, rc)
    print(f"  planner: {out.status}, solve {out.solve_time:.2f} s; dense re-check: {status}"
          + (f" ({label})" if label else "")
          + (f", min distance {1e3 * rc.min_signed_distance:+.2f} mm at t = {rc.t_min:.2f} s" if rc else ""))
    if out.q_of_t is None:
        print("  no trajectory to play")
        return 1
    meshcat = viz.get_meshcat()
    viz.reset_meshcat()
    vis = make_scene(seq.walls, beta=args.beta, meshcat=meshcat)
    path = viz.play(vis, out.q_of_t, seq.wall_y_of_t(), seq.T, fps=args.fps, html_path=args.html)
    if path:
        print(f"  recording: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    if args.keep_open:
        viz.keep_open()
    return 0


# --- parser -------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of the hitw command."""
    ap = argparse.ArgumentParser(prog="hitw", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="registered planners by task, with options and defaults")
    p.add_argument("--load", nargs="*", help="planner files outside planners/ to import first (path/to/file.py)")
    p.set_defaults(func=cmd_list)

    ref_help = "planner id or path/to/file.py:function, optionally with [key=value,...]"
    p = sub.add_parser("check", help="run a planner on two fixed problems; nothing is written")
    p.add_argument("planner", help=ref_help)
    p.add_argument("--config", nargs="*", metavar="KEY=VALUE")
    p.add_argument("--budget", type=float, default=CHECK_BUDGET, help="planner budget, s")
    p.add_argument("--startup-grace", type=float, default=30.0, help="s allowed for startup and grading")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("run", help="run one planner over seeds into a results file")
    p.add_argument("--planner", required=True, help=ref_help)
    p.add_argument("--config", nargs="*", metavar="KEY=VALUE")
    p.add_argument("--beta", nargs="+", type=float, default=[0.0])
    p.add_argument("--eps", nargs="+", type=float, default=[0.01], help="clearance, m")
    p.add_argument("--walls", nargs="+", type=int, default=[1])
    p.add_argument("--seeds", default="0-9", help="e.g. 0-99 or 0,1,5")
    p.add_argument("--out", default="results.jsonl")
    p.add_argument("--cap", type=float, default=None, help="planner budget, s")
    p.add_argument("--budget-from", default=None, help="find runs: take the budget from this run's records in --out")
    p.add_argument("--startup-grace", type=float, default=30.0)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("sweep", help="run an experiment file (experiments/<name>.toml)")
    p.add_argument("experiment")
    p.add_argument("--out", default=None, help="override the file's 'out'")
    p.add_argument("--workers", type=int, default=None, help="override the file's 'workers'")
    p.add_argument("--dry-run", action="store_true", help="print the trial counts and budgets only")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("analyze", help="summary tables of a results file (figures: scripts/figures.py)")
    p.add_argument("results")
    p.add_argument("--out", default=None, help="directory for the CSV tables (default: beside the file)")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("demo", help="plan one connect problem and play it in Meshcat")
    p.add_argument("--planner", required=True, help=ref_help)
    p.add_argument("--config", nargs="*", metavar="KEY=VALUE")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--eps", type=float, default=0.01)
    p.add_argument("--walls", type=int, default=1)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--html", default=None, help="also write the recording as self-contained HTML")
    p.add_argument("--keep-open", action="store_true", help="wait for Enter before exiting")
    p.set_defaults(func=cmd_demo)
    return ap


def main(argv: list[str] | None = None) -> int:
    """Run the hitw command line.

    Parameters
    ----------
    argv : list of str, optional
        Arguments without the program name; None means ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit status.
    """
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (KeyError, ValueError, FileNotFoundError, TypeError) as e:
        msg = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
        print(f"hitw {args.command}: error: {msg}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
