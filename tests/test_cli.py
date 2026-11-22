"""The hitw command line: list, check (and its failure modes), run, sweep, analyze."""

import contextlib
import io
import tempfile
from pathlib import Path

from hitw.cli import load_experiment, main, parse_options, parse_seeds
from hitw.harness import load_results

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = f"{ROOT / 'planners/example_linear.py'}:plan_linear"

BAD = '''
import time
import numpy as np
from hitw.api import PlanOutcome, planner


@planner("{p}-raises", supports_beta_positive=False)
def raises(seq, svc, budget, rng, cfg):
    raise RuntimeError("boom")


@planner("{p}-shape", supports_beta_positive=False)
def shape(seq, svc, budget, rng, cfg):
    return PlanOutcome(status="success", q_of_t=lambda t: np.zeros(8), solve_time=0.0)


@planner("{p}-slow", supports_beta_positive=False)
def slow(seq, svc, budget, rng, cfg):
    time.sleep(1.5)
    return PlanOutcome(status="infeasible", q_of_t=None, solve_time=1.5)


@planner("{p}-hangs", supports_beta_positive=False)
def hangs(seq, svc, budget, rng, cfg):
    time.sleep(600)
'''


def _run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue() + err.getvalue()


def test_argument_parsing():
    assert parse_seeds("0-3,7") == [0, 1, 2, 3, 7] and parse_seeds(5) == [5] and parse_seeds([1, 2]) == [1, 2]
    assert parse_options(["iris=np2", "margin = default"]) == {"iris": "np2", "margin": "default"}


def test_list_shows_builtin_and_loaded_planners():
    code, text = _run(["list", "--load", str(ROOT / "planners/example_linear.py")])
    assert code == 0
    for word in ("connect:", "find:", "kto", "iris-gcs", "iris = 'zo'", "uniform-sampling", "linear", "lead = 0.0"):
        assert word in text, word


def test_check_passes_the_example_and_catches_bad_planners():
    code, text = _run(["check", EXAMPLE])
    assert code == 0 and "check passed" in text and text.count("success") >= 2, text
    code, text = _run(["check", "ik-restarts[K=2]", "--budget", "30"])
    assert code == 0 and "valid of 2 candidates" in text, text
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "bad.py"
        f.write_text(BAD.format(p="cli"))
        for fn, budget, words in (("raises", 10, "RuntimeError: boom"), ("shape", 10, "(8,)"),
                                  ("slow", 0.5, "more than 2x its budget"), ("hangs", 0.5, "no result after")):
            code, text = _run(["check", f"{f}:{fn}", "--budget", str(budget), "--startup-grace", "8"])
            assert code == 1 and "check FAILED" in text and words in text, (fn, text)
            assert "unsupported" in text                                   # beta = 0.1 is not run


def test_run_and_analyze_the_example():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "r.jsonl"
        code, text = _run(["run", "--planner", EXAMPLE, "--seeds", "0-1", "--beta", "0", "0.1",
                           "--out", str(out), "--workers", "2", "--cap", "10"])
        rs = load_results(out)
        assert code == 0 and len(rs) == 4 and {r["method"] for r in rs} == {"linear"}, text
        assert all(r["planner_source"]["path"] == "planners/example_linear.py" for r in rs)
        code, text = _run(["analyze", str(out)])
        assert code == 0 and "Straight line + dwell" in text
        assert (Path(d) / "r_summary.csv").exists() and (Path(d) / "r_paired.csv").exists()


def test_sweep_from_an_experiment_file():
    with tempfile.TemporaryDirectory() as d:
        toml = Path(d) / "x.toml"
        out = Path(d) / "x.jsonl"
        toml.write_text(f'''
out = "{out}"
workers = 3
[grid]
seeds = "0"
betas = [0.0]
eps = [0.02]
n_walls = [1]
[[run]]
planner = "gcs-box"
cap = 30
[[run]]
planner = "ik-restarts"
config = {{ K = 2 }}
cap = 60
[[run]]
planner = "uniform-sampling"
budget_from = "ik-restarts[K=2]"
''')
        code, text = _run(["sweep", str(toml), "--dry-run"])
        assert code == 0 and "3 trials" in text and not out.exists()
        code, text = _run(["sweep", str(toml)])
        rs = {r["config_label"]: r for r in load_results(out)}
        assert code == 0 and set(rs) == {"gcs-box", "ik-restarts[K=2]", "uniform-sampling"}, text
        assert rs["gcs-box"]["budget"] == 30.0 and rs["uniform-sampling"]["budget"] == rs["ik-restarts[K=2]"]["search_time"]
        code, text = _run(["analyze", str(out)])
        assert code == 0 and (Path(d) / "x_find_summary.csv").exists()
        for body, words in (('[[run]]\nplanner = "kto"\ncolour = 1', "takes only"),
                            ('[[run]]\nplanner = "kto"\n[[run]]\nplanner = "kto[retry=true]"', "both 'kto'"),
                            ('[[run]]\nplanner = "uniform-sampling"\nbudget_from = "ik-restarts"', "not a find run")):
            toml.write_text(f'out = "{out}"\n[grid]\nseeds = "0"\nbetas = [0.0]\neps = [0.02]\nn_walls = [1]\n{body}\n')
            try:
                load_experiment(toml)
                raise AssertionError(body)
            except ValueError as e:
                assert words in str(e), (words, str(e))


def test_shipped_experiment_files_load():
    main_exp = load_experiment(ROOT / "experiments/main.toml")
    assert len(main_exp["trials"]) == 12000 and set(main_exp["caps"]) == {
        "kto", "gcs-box", "iris-gcs", "iris-gcs[iris=np2]", "rrt-connect"}
    assert len(load_experiment(ROOT / "experiments/regression.toml")["trials"]) == 90
    assert len(load_experiment(ROOT / "experiments/find.toml")["trials"]) == 240


def test_python3_hitw_runs_the_command_line():
    import subprocess
    import sys
    for cwd, target in ((ROOT, "hitw"), (Path("/"), str(ROOT / "hitw"))):
        p = subprocess.run([sys.executable, target, "list"], cwd=cwd, capture_output=True, text=True, timeout=120)
        assert p.returncode == 0 and "ik-restarts" in p.stdout, (cwd, p.stdout, p.stderr)
    p = subprocess.run([sys.executable, "-I", "-S", "hitw", "list"], cwd=ROOT, capture_output=True, text=True,
                       timeout=60)                                        # no site-packages: no dependencies
    assert p.returncode == 1 and "is not installed for" in p.stderr and "README.md" in p.stderr, p.stderr
