"""Milestone 2 demo: M1 (KTO) through given crossings on a 1-wall and a 2-wall
sequence at beta = 0 and beta = 0.1, dense 1 ms re-check printed, and a
Meshcat recording of the arm passing through the moving wall for each case.

Run:  .venv/bin/python scripts/m2_demo.py [--eps 0.005] [--no-record]
      HEADLESS=1 .venv/bin/python scripts/m2_demo.py       # numbers only
Writes out/m2_demo_eps{mm}mm_beta{beta}_{walls}.html (self-contained; open in a browser).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from hitw import params
from hitw.evaluate import classify, recheck
from hitw.plan_trajopt import plan_trajopt, plant_q_of_t
from hitw.scene import make_scene
from hitw.sequences import LETTER_POSES, make_sequence
from hitw.viz import get_meshcat, play, reset_meshcat

CASES = [["L"], ["L", "SLASH"]]
SEED = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.005)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    out = Path(args.out)
    meshcat = None if args.no_record else get_meshcat()
    rows = []
    for beta in params.BETAS:
        for poses in CASES:
            rng = np.random.default_rng(SEED)
            seq = make_sequence([LETTER_POSES[p] for p in poses], beta=beta, eps=args.eps, rng=rng, names=poses)
            scene = make_scene(seq.walls, beta=beta)
            name = f"eps{1e3*args.eps:g}mm_beta{beta}_{'+'.join(poses)}"
            print(f"\n== {name}: eps = {1e3*args.eps:.0f} mm, T = {seq.T:.1f} s, rho = {seq.rho:.3f}")
            for c in seq.crossings:
                print(f"   {c.name:6s} t = {c.t:.1f} s, window +/- {c.delta:.3f} s, tol = {c.tol:.2e} rad, "
                      f"rolls = {np.array2string(c.roll_draw, precision=3, suppress_small=True)}, "
                      f"q* = {np.array2string(c.q_star, precision=3, suppress_small=True)}")
            res = plan_trajopt(seq, scene)
            print(f"   KTO: status = {res.status}, SNOPT info = {res.solver_info}, solve time = {res.solve_time:.2f} s, "
                  f"{res.n_evals} function evaluations, {res.extras['n_control_points']} control points, "
                  f"{res.extras['n_distance_samples']} distance samples")
            rc = None
            if res.traj is not None:
                q_of_t = plant_q_of_t(res.traj, scene)
                rc = recheck(q_of_t, seq, scene)
                print(f"   dense re-check ({rc.n_samples} samples at 1 ms, {rc.elapsed:.1f} s): "
                      f"min signed distance = {1e3*rc.min_signed_distance:+.2f} mm at t = {rc.t_min:.3f} s, "
                      f"window excess = {rc.window_excess:+.1e} rad, peak joint speed = {rc.peak_joint_vel:.2f} rad/s, "
                      f"path length = {rc.path_length:.2f} rad")
            status, label = classify(res.status, rc)
            print(f"   => {status}" + (f" ({label})" if label else ""))
            rows.append((name, status, res.solve_time, res.n_evals, rc.min_signed_distance if rc else None))
            if meshcat is not None and res.traj is not None:
                reset_meshcat()
                vis_scene = make_scene(seq.walls, beta=beta, meshcat=meshcat)
                path = play(vis_scene, q_of_t, seq.wall_y_of_t(), seq.T, fps=30,
                            html_path=out / f"m2_demo_{name}.html")
                print(f"   recording: {path} ({path.stat().st_size/1e6:.1f} MB)")
    print("\nsummary")
    for name, status, st, ne, d in rows:
        print(f"   {name:26s} {status:10s} solve {st:5.2f} s  {ne:5d} evals" + (f"  min d {1e3*d:+.2f} mm" if d is not None else ""))


if __name__ == "__main__":
    main()
