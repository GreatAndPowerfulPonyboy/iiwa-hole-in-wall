"""Demo: one longer crossing sequence played through every non-sampling
planner (KTO, the GCS chain, IRIS + GCS), each judged by the dense 1 ms
re-check, with one Meshcat recording per planner.

Run:  .venv/bin/python scripts/demo_long.py                       # 6 walls, beta 0.1, eps 5 mm, seed 1
      .venv/bin/python scripts/demo_long.py --poses L SLASH SEVEN I Z U --beta 0 --eps 0.002 --spacing 4
      .venv/bin/python scripts/demo_long.py --random 8            # 8 walls drawn by random_sequence (transit-feasible)
      HEADLESS=1 .venv/bin/python scripts/demo_long.py --methods KTO GCS   # numbers only
Writes out/demo_long_{method}.html (self-contained; open in a browser).

The re-check costs about 1 s per second of trajectory and a recording about
1.3 MB per second at 30 fps; --fps 15 halves the file.  IRIS is sampling-based
inside (seeded), so IrisZo+GCS depends on --seed.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from hitw import params
from hitw.evaluate import METHODS, canonical_method, classify, recheck
from hitw.scene import make_scene
from hitw.sequences import LETTER_POSES, make_sequence, random_sequence, transit_feasible
from hitw.viz import get_meshcat, play, reset_meshcat

DEFAULT_METHODS = ["KTO", "GCS", "IrisZo+GCS", "IrisNp2+GCS"]
DEFAULT_POSES = ["L", "SLASH", "SEVEN", "I", "Z", "U"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", nargs="+", default=None, help=f"crossing poses in order, from {list(LETTER_POSES)}")
    ap.add_argument("--random", type=int, default=None, help="draw this many crossings with random_sequence instead")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--spacing", type=float, default=params.WINDOW_SPACING, help="seconds between crossings")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    if args.random:
        seq = random_sequence(args.random, rng, args.beta, args.eps, make_scene([], beta=args.beta), spacing=args.spacing)
        if seq is None:
            sys.exit("random_sequence found no transit-feasible sequence; try fewer walls or a larger --spacing")
    else:
        poses = args.poses or DEFAULT_POSES
        seq = make_sequence([LETTER_POSES[p] for p in poses], beta=args.beta, eps=args.eps, rng=rng,
                            names=poses, spacing=args.spacing)
    scene = make_scene(seq.walls, beta=args.beta)
    ok, info = transit_feasible(seq, scene)
    print(f"sequence {[c.name for c in seq.crossings]}: {seq.n_walls} walls, T = {seq.T:.1f} s, beta = {args.beta}, "
          f"eps = {1e3*args.eps:g} mm, straight-line transits feasible: {ok}")
    for c, leg in zip(seq.crossings, info["legs"]):
        print(f"   {c.name:6s} t = {c.t:.1f} s, window +/- {c.delta:.3f} s, tol = {c.tol:.2e} rad, "
              f"rolls = {np.array2string(c.roll_draw, precision=3, suppress_small=True)}; "
              f"leg: edge free {leg['edge_free']}, toppra {leg['toppra_time']:.2f} s of {leg['available']:.2f} s")
    meshcat = get_meshcat()
    out = Path(args.out)
    rows = []
    for name in args.methods:
        method = canonical_method(name)
        print(f"\n== {method}")
        res = METHODS[method](seq, scene, args.seed)
        build = f", region build {res.region_build_time:.1f} s" if res.region_build_time else ""
        print(f"   planner: {res.status}, solve {res.solve_time:.2f} s{build}")
        rc = recheck(res.q_of_t, seq, scene) if res.q_of_t is not None else None
        if rc is not None:
            print(f"   dense re-check ({rc.n_samples} samples, {rc.elapsed:.1f} s): min signed distance "
                  f"{1e3*rc.min_signed_distance:+.2f} mm at t = {rc.t_min:.2f} s, window excess {rc.window_excess:+.1e} rad, "
                  f"peak joint speed {rc.peak_joint_vel:.2f} rad/s, path length {rc.path_length:.2f} rad")
        status, label = classify(res.status, rc)
        print(f"   => {status}" + (f" ({label})" if label else ""))
        rows.append((method, status, res.solve_time, res.region_build_time, rc.min_signed_distance if rc else None))
        if meshcat is not None and res.q_of_t is not None:
            reset_meshcat()
            vis = make_scene(seq.walls, beta=args.beta, meshcat=meshcat)
            tag = method.replace("+", "_")
            path = play(vis, res.q_of_t, seq.wall_y_of_t(), seq.T, fps=args.fps, html_path=out / f"demo_long_{tag}.html")
            print(f"   recording: {path} ({path.stat().st_size/1e6:.1f} MB)")
    print("\nsummary")
    for m, st, t, b, d in rows:
        print(f"   {m:22s} {st:10s} solve {t:6.2f} s" + (f"  build {b:6.1f} s" if b else "              ")
              + (f"  min d {1e3*d:+.2f} mm" if d is not None else ""))


if __name__ == "__main__":
    main()
