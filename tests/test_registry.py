"""Planner registry: lookup by id and by file path, records that identify the
planner's code and config, Budget, and the guard on NumPy seeding."""

import ast
import hashlib
import json
import tempfile
import time
from pathlib import Path

from hitw.api import Budget, get_planner, planner, planners
from hitw.harness import load_results, run_trial
from hitw.sweep import grid, run_sweep

ROOT = Path(__file__).resolve().parents[1]

EXTERNAL = '''
from hitw.api import PlanOutcome, planner
from hitw.sequences import reference_path


@planner("{id}", display_name="Reference path")
def follow_reference(seq, svc, budget, rng, cfg):
    ref = reference_path(seq)
    return PlanOutcome(status="success", q_of_t=lambda t: ref.value(t).ravel(), solve_time=0.0,
                       extras={{"drake_seed": svc.drake_seed}})


def not_registered(seq, svc, budget, rng, cfg):
    return None
'''


def _write(d: Path, name: str, pid: str) -> Path:
    p = d / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(EXTERNAL.format(id=pid))
    return p


def _raises(exc, fn, *words):
    try:
        fn()
    except exc as e:
        assert all(w in str(e) for w in words), str(e)
        return
    raise AssertionError(f"no {exc.__name__}")


def test_every_planner_in_planners_is_discovered():
    found = {s.id: (s.task, s.func.__module__) for s in planners() if s.func.__module__.startswith("planners.")}
    assert found == {"kto": ("connect", "planners.kto"), "gcs-box": ("connect", "planners.gcs"),
                     "iris-gcs": ("connect", "planners.gcs"), "rrt-connect": ("connect", "planners.rrt_connect"),
                     "linear": ("connect", "planners.example_linear"),
                     "ik-restarts": ("find", "planners.ik_restarts"),
                     "uniform-sampling": ("find", "planners.uniform_sampling"),
                     "bridge-test": ("find", "planners.bridge_test")}, found
    assert {s.task for s in planners("find")} == {"find"}
    _raises(KeyError, lambda: get_planner("KTO"), "kto", "rrt-connect")
    src = get_planner("kto").source()
    assert src["path"] == "planners/kto.py" and src["module"] == "planners.kto"
    assert src["sha256"] == hashlib.sha256((ROOT / "planners/kto.py").read_bytes()).hexdigest()
    by_path = get_planner(f"{ROOT / 'planners/example_linear.py'}:plan_linear")
    assert by_path is get_planner("linear")                                # one import, whichever way it is reached


def test_external_file_lookup_and_errors():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        a = _write(d / "a", "mine.py", "ext-a")
        b = _write(d / "b", "mine.py", "ext-b")                       # same file name, different directory
        sa, sb = get_planner(f"{a}:follow_reference"), get_planner(f"{b}:follow_reference")
        assert (sa.id, sb.id) == ("ext-a", "ext-b") and sa.func.__module__ != sb.func.__module__
        assert get_planner(f"{a}:follow_reference") is sa                # imported once
        assert sa.source()["path"] == str(a.resolve())
        assert sa.display_name == "Reference path" and sa.task == "connect"
        _raises(ValueError, lambda: get_planner(f"{a}:not_registered"), "not registered")
        _raises(ValueError, lambda: get_planner(f"{a}:missing"), "no function")
        _raises(FileNotFoundError, lambda: get_planner(f"{d}/none.py:f"), "does not exist")
        _raises(ValueError, lambda: get_planner(f"{a}:follow_reference").make_config({"x": 1}), "unknown option")
        c = _write(d / "c", "other.py", "ext-a")                      # an id taken by another function
        _raises(ValueError, lambda: get_planner(f"{c}:follow_reference"), "already registered")


def test_decorator_validation():
    _raises(ValueError, lambda: planner("a:b"), "':'")
    _raises(ValueError, lambda: planner("a[b]"), "'['")
    _raises(ValueError, lambda: planner("x", task="solve"), "connect")
    _raises(ValueError, lambda: planner("x", config=dict), "dataclass")


def test_external_planner_runs_and_records_its_source():
    with tempfile.TemporaryDirectory() as d:
        f = _write(Path(d), "ref_planner.py", "ext-ref")
        r = run_trial(f"{f}:follow_reference", seed=0, beta=0.0, eps=0.02, n_walls=1, recheck_dt=5e-3)
        assert r["method"] == "ext-ref" and r["task"] == "connect" and r["planner_config"] == {}
        assert r["planner_source"]["sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()
        assert r["planner_extras"]["drake_seed"] == 0
        assert r["status"] in ("success", "infeasible") and r["min_signed_distance"] is not None
        json.dumps(r)


def test_external_planner_in_a_sweep_subprocess():
    with tempfile.TemporaryDirectory() as d:
        f = _write(Path(d), "sub_planner.py", "ext-sub")
        out = Path(d) / "r.jsonl"
        trials = grid([f"{f}:follow_reference"], [0], [0.0], [0.02], [1])
        assert trials[0].method == "ext-sub" and trials[0].planner == f"{f.resolve()}:follow_reference"
        stats = run_sweep(trials, out, workers=1, cap=120.0, verbose=False)
        rs = load_results(out)
        assert stats.run == 1 and len(rs) == 1 and "error" not in rs[0], rs
        assert rs[0]["method"] == "ext-sub" and rs[0]["planner_source"]["path"] == str(f.resolve())


def test_budget():
    b = Budget(wall_clock=0.05)
    assert not b.expired() and 0 < b.remaining() <= 0.05
    time.sleep(0.06)
    assert b.expired() and b.remaining() == 0.0
    u = Budget(None)
    assert u.remaining() == float("inf") and not u.expired()


# --- NumPy must never be seeded with the trial seed or drake_seed ----------------

SEED_NAMES = {"seed", "drake_seed"}
# The only NumPy streams derived from the trial seed: the problem stream and
# the planner stream (a spawned child, independent of it).
ALLOWED = {("hitw/harness.py", "trial_rng"), ("hitw/harness.py", "planner_rng")}


def _seeded_default_rng_calls(source: str):
    """Yield (function name, seed names) for each default_rng call whose arguments mention a seed."""
    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not ((isinstance(f, ast.Name) and f.id == "default_rng") or
                (isinstance(f, ast.Attribute) and f.attr == "default_rng")):
            continue
        names = set()
        for arg in list(node.args) + [k.value for k in node.keywords]:
            for sub in ast.walk(arg):
                if isinstance(sub, ast.Name) and sub.id in SEED_NAMES:
                    names.add(sub.id)
                if isinstance(sub, ast.Attribute) and sub.attr in SEED_NAMES:
                    names.add(sub.attr)
        if not names:
            continue
        p = node
        while p in parents and not isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)):
            p = parents[p]
        yield (p.name if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)) else "<module>"), names


def test_guard_detects_seeded_default_rng():
    src = ("import numpy as np\n"
           "def a(seed):\n    return np.random.default_rng(seed + 1)\n"
           "def b(svc):\n    return np.random.default_rng(svc.drake_seed)\n"
           "def c(rng):\n    return np.random.default_rng(rng.integers(9))\n")
    assert dict(_seeded_default_rng_calls(src)) == {"a": {"seed"}, "b": {"drake_seed"}}


def test_no_numpy_stream_is_seeded_with_the_trial_seed_or_drake_seed():
    found = set()
    bad = []
    for path in sorted(list((ROOT / "hitw").rglob("*.py")) + list((ROOT / "planners").rglob("*.py"))):
        rel = str(path.relative_to(ROOT))
        for func, names in _seeded_default_rng_calls(path.read_text()):
            if "drake_seed" in names or (rel, func) not in ALLOWED:
                bad.append(f"{rel}:{func} seeds default_rng with {sorted(names)}")
            else:
                found.add((rel, func))
    assert not bad, "\n".join(bad)
    assert found == ALLOWED, f"allow-list entries no longer needed: {ALLOWED - found}"
