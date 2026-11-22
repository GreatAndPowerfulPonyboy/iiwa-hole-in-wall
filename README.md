# Comparison of Methods of Narrow Passage Motion Planning via Hole-In-The-Wall
Contributors: Victor Ogunsanya, Jason Huynh, Soma Ueno
Final project for MIT 6.4210, fall 2026.

A motion-planning benchmark in [Drake](https://drake.mit.edu).  A KUKA iiwa7
is welded at the origin while wall slabs slide along y and pass over it.
Each slab has an opening cut in the shape of the arm's silhouette at a
*crossing configuration* q\*, enlarged by a clearance eps.  The roll joints
are held within ±beta, with beta = 0 restricting to planar motion.  There are two tasks:

- **connect**: the crossing configurations q\*_i and times t_i are given;
  plan a timed trajectory from the start configuration through every wall.
- **find**: only the wall is given; find a configuration that fits through it.

Results are graded against a signed-distance collision recheck.

## Install
As a prelimininary, to not upset Drake's model downloader when parsing the iiwa, keep the repo under a path without spaces.
i.e store it under /home/user/Projects/hitw instead of /home/user/My Projects/hitw

### Dependencies
Python 3.13 with `drake` 1.58.0, `numpy` 2.5.4, `scipy` 1.18.1, `shapely`
2.2.0, `pandas` 3.1.1 and `matplotlib` 3.11.2, installed with
[uv](https://docs.astral.sh/uv/):

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate 
#If you use [fish, the friendly interactive shell](https://github.com/fish-shell/fish-shell/) the source command changes to source .venv/bin/activate.fish

```
## Quick start

From the repository root, with the venv active:

```sh
python3 hitw list                     # the planners, their options and defaults
python3 hitw <command> --help         # every command's options
```

The planners in `planners/` are the connectors `kto`
(KinematicTrajectoryOptimization), `gcs-box` (GCS over tolerance boxes),
`iris-gcs` (GCS over IRIS regions), `rrt-connect` and `linear` (the minimal
example), and the finders `ik-restarts`, `uniform-sampling` and
`bridge-test` (the tutorial's).

## Two kinds of planner

Every planner solves one of the two tasks, and its signature says which:

| kind | task | called as | gives back |
|---|---|---|---|
| **connector** | connect | `plan(seq, svc, budget, rng, cfg)` | one timed trajectory through all the walls, as a `PlanOutcome` |
| **finder** | find | `find(wall, svc, budget, rng, cfg)` | a stream of candidate configurations (`yield`), one at a time |

A connector sees the whole problem, the crossing configurations included,
and is graded on its trajectory.  A finder sees only the wall and never the
configuration the opening was cut for; the harness grades each candidate it
proposes.  `@planner(..., task="find")` registers a finder and the default
(`task="connect"`) a connector.  The tutorial below builds a finder;
[Writing a connector](#writing-a-connector) covers the other kind.

## Tutorial: adding a finding planner from a paper

We'll add the **bridge test** (Hsu, Jiang, Reif and Sun, *The bridge test
for sampling narrow passages with probabilistic roadmap planners*, ICRA
2003) and compare it with uniform sampling on the find task.
### Step 1: write the planner

Every planner is one file in `planners/`.  Create `planners/bridge_test.py`:

```python
from dataclasses import dataclass

import numpy as np

from hitw.api import planner


@dataclass(frozen=True)
class BridgeConfig:
    sigma: float = 0.1         # rad, spread of a bridge's second end around its first


@planner("bridge-test", task="find", config=BridgeConfig, display_name="Bridge test")
def find_bridge(wall, svc, budget, rng, cfg):
    scene, t = svc.scene, wall.t_cross
    lo, hi = scene.free_limits()
    while not budget.expired():
        a = rng.uniform(lo, hi)
        if not svc.in_collision(scene.q_full(a), t):
            continue
        b = np.clip(a + rng.normal(scale=cfg.sigma, size=a.shape), lo, hi)
        if not svc.in_collision(scene.q_full(b), t):
            continue
        mid = scene.q_full((a + b) / 2)
        if not svc.in_collision(mid, t):
            yield mid
```

- `@planner` registers the function under the id `bridge-test`; `task="find"`
  makes it a finder, and `config` is the dataclass of its options.
- `wall` is what the finder may know about the wall (its boxes, size,
  `t_cross` when it is centred on the arm, `beta`, `eps`), never the
  configuration the opening was cut for.  `svc.in_collision(q, t)` checks a
  configuration against the wall at time t; `scene.free_limits()` gives the
  joint limits with the roll joints held to ±beta; `scene.q_full` turns the
  free joints into the full 7-vector.
- A finder only proposes: each `yield` is a candidate, and the harness checks
  it, counts it and times it.
- Draw every random number from `rng`, and check `budget.expired()` in any
  loop that may run long between candidates.

### Step 2: see that hitw found it

```sh
$ python3 hitw list
...
find:
  bridge-test          Bridge test    [planners.bridge_test]
      sigma = 0.1
  ik-restarts          IK restarts    [planners.ik_restarts]
  ...
```

Files in `planners/` are picked up by themselves; one that fails to import
is named at the top of this listing.

### Step 3: check it

```sh
$ python3 hitw check bridge-test --budget 10
check bridge-test (Bridge test, task find): seed 0, eps 10 mm, 1 wall, budget 10 s; nothing is written
  beta = 0: 1 valid of 1 candidates (stopped: exhausted); first valid at attempt 1; hidden pose valid True; search 10.00 s of 10 s
  beta = 0.1: 0 valid of 0 candidates (stopped: exhausted); first valid at attempt None; hidden pose valid True; search 10.00 s of 10 s
check passed
```

`check` runs one problem at beta = 0 and one at 0.1 and writes nothing.  It
fails if the planner raises, proposes something that is not a 7-vector, or
uses more than twice its budget.

### Step 4: describe the comparison

An experiment is a TOML file: a grid of problems and the runs to put on it.
Create `experiments/bridge.toml`:

```toml
out = "out/bridge.jsonl"
workers = 6

[grid]
seeds = "0-4"
betas = [0.0, 0.1]
eps = [0.02, 0.005]          # clearance, m
n_walls = [1]

[[run]]
planner = "uniform-sampling"
cap = 10.0                   # s of search per wall

[[run]]
planner = "bridge-test"
cap = 10.0
```

```sh
$ python3 hitw sweep experiments/bridge.toml --dry-run
experiments/bridge.toml: 40 trials -> out/bridge.jsonl
  uniform-sampling                     20 trials, budget 10 s
  bridge-test                          20 trials, budget 10 s
```

### Step 5: run it

```sh
$ python3 hitw sweep experiments/bridge.toml
```

About 1.5 minutes on 6 workers.  Each trial runs in its own process and is
killed if it overruns its budget by more than 30 s.  Both runs face the same
walls, because the problems depend only on the seed.  Rerunning skips the
trials already in `out/bridge.jsonl`.

### Step 6: read the results

```sh
$ python3 hitw analyze out/bridge.jsonl
         display  beta   eps  trials  valid  attempts        p
     Bridge test     0  0.02       5      1         1        1
     Bridge test     0 0.005       5      0         0      NaN
     Bridge test   0.1  0.02       5      1         1        1
     Bridge test   0.1 0.005       5      0         0      NaN
Uniform sampling     0  0.02       5      3     99623 3.01e-05
Uniform sampling     0 0.005       5      0    102835        0
Uniform sampling   0.1  0.02       5      5     98128  5.1e-05
Uniform sampling   0.1 0.005       5      0     97448        0
```

(columns trimmed; the full table, with Wilson 95% intervals, is written to
`out/bridge_find_summary.csv`.)  Per attempt the bridge test looks perfect:
every midpoint it proposed was a valid crossing.  But over the same 50 s of
search per row it proposed one candidate, while uniform sampling found 3 and
5.  Compare finders by `valid` at equal budget, not by `p` per attempt.

### Step 7: try another setting

```sh
$ python3 hitw run --planner "bridge-test[sigma=0.3]" --seeds 0-4 --beta 0 0.1 --eps 0.02 0.005 \
      --cap 10 --out out/bridge.jsonl
$ python3 hitw analyze out/bridge.jsonl
```

The new rows appear as `Bridge test [sigma=0.3]` (run label
`bridge-test[sigma=0.3]` in the records): each setting is its own run, next
to the others in the same file.  In an experiment file the same
run is `planner = "bridge-test"` with `config = { sigma = 0.3 }`.

## Writing a connector

A connector gets the crossings and returns a timed trajectory:
`plan(seq, svc, budget, rng, cfg) -> PlanOutcome`.  The simplest one,
`linear` in `planners/example_linear.py`, moves in straight lines between the crossing
configurations and holds each while its wall passes:

```sh
python3 hitw check linear
python3 hitw demo --planner linear --seed 3 --beta 0.1 --html out/linear.html
```

| argument | contents |
|---|---|
| `seq` | `q0` (start, 7-vector, rad); `crossings`, each with `q_star`, `t` (s), `delta` (half-width of the window in which the wall overlaps the arm, s) and `tol` (allowed 2-norm deviation from `q_star` in the window, rad); `T` (duration, s); `beta`; `eps` (m); `wall_y_of_t()` |
| `svc` | `scene` (the scene the grading uses), `in_collision(q, t)`, `min_distance(q, t)`, `welded_scene(y)`, `swept_window_scene(i)`, `wall_absent_scene()`, `checker(scene)` (a Drake `SceneGraphCollisionChecker`, e.g. for IRIS), `drake_seed` (for Drake's own random generators) |
| `budget` | `remaining()`, `expired()` |
| `rng`, `cfg` | as for finders |

Return a `PlanOutcome(status, q_of_t, solve_time, ...)`.  `q_of_t(t)` must
give the full joint vector, shape (7,), for every t in [0, T].  The harness
then checks the trajectory every 1 ms with the wall at its true position: a
trial succeeds only if the arm never touches the wall or itself and stays
within `tol` of each `q_star` during its window.  The planner's own `status`
is recorded but not trusted: a reported success that fails the check is
counted as `reported_feasible_but_collides` or
`collision_free_but_misses_window`.  A planner that only handles beta = 0
is registered with `supports_beta_positive=False`, and its beta > 0 trials
are left out of success rates.

## Experiments and results

- `experiments/main.toml`: the connect sweep (five runs x 100 seeds x 6 eps x
  beta {0, 0.1} x {1, 2} walls, 12 000 trials, frozen budgets).
- `experiments/find.toml`: the find task (10 seeds x 6 eps x 2 betas; uniform
  sampling gets the IK restarts' time on the same wall).
- `experiments/regression.toml`: every planner on a small grid, about 3 minutes.

`python3 hitw sweep <file>` runs one (`--dry-run` prints the trial counts).
Results are JSON lines, one per trial, appended as trials finish.  Rerunning
a sweep skips trials already in its file.  Each record holds the problem, the
outcome and its timing, the resolved config, and the planner file's path and
SHA-256, along with the Drake version and the git commit.
`python3 hitw analyze` prints and writes the tables (Wilson 95% intervals,
exact McNemar tests between runs on shared seeds).
`python3 scripts/figures.py <results files> --out figures/` draws the
figures.

The results in `results/` come from the pilot and smoke runs.  The
`.v2.jsonl` files are those runs converted to the current record format by
`scripts/migrate_results.py`; analysis reads only the current format.  GCS
uses Clarabel (MOSEK was unlicensed), so its timings are not comparable to
published MOSEK numbers.

## Repository Layout

```
hitw/           the benchmark: api (planner contract, registry), harness (trials, results files),
                evaluate (the connect re-check), find (the find validity check), sweep, cli, the
                scene, walls and problems, analysis (statistics), viz (Meshcat)
planners/       every planner, one file each: kto.py, gcs.py (gcs-box, iris-gcs), rrt_connect.py,
                ik_restarts.py, uniform_sampling.py, bridge_test.py (the tutorial), example_linear.py
experiments/    experiment files for `hitw sweep`
results/        results files
scripts/        figures.py, calibrate_caps.py (planner budgets from a pilot), migrate_results.py,
                gen_letter_poses.py (the crossing poses), wall_clearance_demo.py, env_check.py
tests/          python3 tests/run_tests.py
```
