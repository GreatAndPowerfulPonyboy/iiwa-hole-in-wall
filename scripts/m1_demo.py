"""Milestone 1: scenes for beta = 0 and beta = 0.1, realized clearance, and a
Meshcat recording of the arm (static at q*) with one wall passing over it.

Run:  .venv/bin/python scripts/m1_demo.py          # writes out/m1_wall_demo_beta*.html
      HEADLESS=1 .venv/bin/python scripts/m1_demo.py   # numbers only
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from hitw import params
from hitw.scene import make_scene, min_distance, place_walls, signed_distance_pairs, wall_y_of_t
from hitw.viz import get_meshcat, play, reset_meshcat
from hitw.walls import make_wall

Q_STAR = np.array([0, 0.5, 0, -1.9, 0, 0.65, 0])


def main():
    out = Path("out")
    for beta in params.BETAS:
        rng = np.random.default_rng(0)
        q = Q_STAR.copy()
        if beta > 0:
            q[list(params.ROLL_JOINTS)] = rng.uniform(-beta, beta, size=4)
        eps = 0.01
        w = make_wall(q, eps)
        place_walls([w], [params.T_FIRST])
        meshcat = get_meshcat()
        reset_meshcat()
        s = make_scene([w], beta=beta, meshcat=meshcat)
        delta = params.window_half_width(w.arm_depth)
        print(f"\nbeta = {beta}: q* = {np.round(q, 3)}")
        print(f"  wall: eps = {1e3*eps:.0f} mm, {w.n_boxes} boxes, arm depth d = {w.arm_depth:.3f} m, "
              f"window half-width delta = {delta:.3f} s, tol = {params.window_tolerance(eps, s.rho):.5f} rad (rho = {s.rho:.3f})")
        d_wall = min(d for d, a, b in signed_distance_pairs(s, q, 0.0) if "wall" in (a, b))
        print(f"  realized wall clearance at q*: {1e3*d_wall:.2f} mm (nominal {1e3*eps:.0f} mm); "
              f"min over all pairs {1e3*min_distance(s, q, 0.0):.1f} mm")
        t0 = time.perf_counter()
        n = 100
        for _ in range(n):
            min_distance(s, q, 0.0)
        print(f"  min_distance cost with this wall at the arm: {1e3*(time.perf_counter()-t0)/n:.2f} ms")
        T = params.T_FIRST + params.T_TAIL
        y_of_t = wall_y_of_t([w])
        ts = np.arange(0.0, T + 1e-9, 0.001)
        t1 = time.perf_counter()
        dmin = min(min_distance(s, q, y_of_t(t), cap=0.1) for t in ts)
        print(f"  dense 1 ms check (arm static at q*, {len(ts)} samples, cap 0.1): min = {1e3*dmin:.2f} mm "
              f"in {time.perf_counter()-t1:.1f} s")
        path = play(s, lambda t: q, y_of_t, T, fps=30, html_path=out / f"m1_wall_demo_beta{beta}.html")
        if path is not None:
            print(f"  recording written to {path} ({path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
