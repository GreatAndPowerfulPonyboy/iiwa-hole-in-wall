# iiwa-hole-in-wall

Hole-In-The-Wall (HITW): a motion-planning benchmark on a fixed-base KUKA iiwa7
in Drake. A wall slab translates along the y axis over the arm; each slab has a
cutout that is the buffered silhouette of the arm at a crossing configuration
q*_i, to be matched at time t_i. Final project for MIT 6.4210, fall 2026.

Stage A compares planners (KTO, degenerate GCS chain, IRIS+GCS, RRT-Connect)
that connect given crossing configurations. Stage B measures how hard it is to
find a valid crossing configuration as a function of clearance eps.

## Environment (pinned)

| item | value |
|---|---|
| Python | 3.13.12 (uv-managed CPython) |
| Drake | **1.58.0** (pip wheel, nanobind bindings) |
| manipulation (textbook pkg) | 2026.10.17 |
| numpy / scipy / shapely | 2.5.4 / 1.18.1 / 2.2.0 |
| OS | Linux 6.18 (WSL2), 16 threads, 7 GB RAM |

Every trial record carries `drake_version`. The venv is `.venv/` at the repo
root (a path with no spaces, which drake_models' downloader needs; see
`reference/6.4210/utils/drake_models.py`). The exact package set is frozen in
`requirements.lock.txt` (`uv pip freeze`). To recreate:

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
.venv/bin/python -c "import pydrake"
```

Robot model: `package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf`
(all collision geometries are boxes, which the motion bound and silhouette
projection rely on). drake_models is cached in `~/.cache/drake` (170 MB).

### Solvers in this Drake build

| solver | available | enabled | role here |
|---|---|---|---|
| SNOPT | yes | yes | KTO (M1), IK (B1) |
| IPOPT | yes | yes | fallback NLP |
| MOSEK | yes | **no (no license)** | dropped for now |
| Clarabel | yes | yes | GCS (results flagged as not comparable to published MOSEK timings) |
| Gurobi | no | no | not used |

Drake picks the solver automatically; GCS results obtained with Clarabel are
tagged as such in `results.jsonl`.

## Layout

```
hitw/            the package: scene, walls, kinematics, sequences, plan_trajopt (M1), plan_gcs (M2, M3),
                 plan_rrt (M4), stage_b (B1 IK restarts, B2 rejection sampling), evaluate (recheck, run_trial,
                 results.jsonl), sweep (subprocess driver, per-method caps), analysis, viz
results/         results.jsonl files from pilot and smoke sweeps, summaries and plots
scripts/         reproducible milestone checks (scripts/m0_env_check.py, ...)
tests/           unit tests (plain asserts; run with tests/run_tests.py)
reference/       read-only copies of course code we copy patterns from (do not import, do not edit)
PROGRESS.md      milestones, decisions, open questions, reproduction commands
```

## Reproduce milestone checks

```sh
.venv/bin/python scripts/m0_env_check.py     # M0: versions, solvers, iiwa load, distance-query cost
.venv/bin/python tests/run_tests.py          # M1: unit tests (plain asserts, no pytest)
.venv/bin/python scripts/m1_demo.py          # M1: realized clearance + Meshcat recordings in out/
.venv/bin/python scripts/m2_demo.py --eps 0.005   # M2: KTO through 1 and 2 walls at beta 0 and 0.1, dense re-check, recordings in out/
.venv/bin/python scripts/gen_letter_poses.py # regenerate the LETTER_POSES constants by IK
.venv/bin/python scripts/run_sweep.py --out results/x.jsonl --methods KTO GCS IrisZo+GCS RRT-Connect --seeds 0-1 --eps 0.02 0.002 --cap 300
                                             # M3/M4: any method grid, one subprocess per trial under a kill timer (resumable)
.venv/bin/python scripts/plot_results.py results/x.jsonl   # summary CSV + success-vs-eps plot
.venv/bin/python scripts/calibrate_caps.py results/pilot_m5.jsonl --ceiling 300   # M5: pilot table + proposed per-method caps JSON
.venv/bin/python scripts/run_stage_b.py --out results/stage_b.jsonl --seeds 0-9 --K 30   # M5: Stage B (B1 then B2 per trial)
.venv/bin/python scripts/plot_stage_b.py results/stage_b.jsonl                    # Stage B table + log-log probability plot
.venv/bin/python scripts/demo_long.py --random 6              # one long sequence through KTO, GCS, IrisZo+GCS, IrisNp2+GCS, recordings in out/
```

Methods (`hitw.evaluate.METHODS`, names are the method; the kickoff's milestone labels are accepted as aliases
and normalized when results are loaded): `KTO` (M1, KinematicTrajectoryOptimization + SNOPT), `GCS` (M2, degenerate
GCS chain of boxes, Clarabel), `IrisZo+GCS` (M3), `IrisNp2+GCS`, `IrisNp+GCS` (legacy), `IrisZo+GCS-defmargin`
(Drake's default IRIS margin, fails below about 1 cm), `RRT-Connect` (M4, time-aware, with shortcutting).
Every trial is judged by the same dense 1 ms re-check with the wall at its true position.
Stage B (`hitw/stage_b.py`): B1 = Drake InverseKinematics with a minimum-distance constraint from K uniform
random restarts (SNOPT); B2 = uniform rejection sampling under B1's wall-clock; validity for both is the
shared `min_distance >= 0` with the slab centred on the arm.

Set `HEADLESS=1` to run without Meshcat. Recordings are self-contained HTML
files written to `out/` (git-ignored); open them in a browser.
