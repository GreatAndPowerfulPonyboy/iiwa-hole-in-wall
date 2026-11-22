"""Generate the LETTER_POSES constants in hitw/sequences.py by IK on the planar
(beta = 0) arm: each target is a flange position (x, z) and a flange direction
in the x-z plane, solved with a self-clearance constraint from several warm
starts.  Paste the printed configurations into hitw/sequences.py.

Run:  .venv/bin/python scripts/gen_letter_poses.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from pydrake.all import InverseKinematics, SnoptSolver, SolverOptions

from hitw.scene import make_scene, min_distance
from hitw.walls import silhouette_polygon

SELF_CLEARANCE = 0.025      # m, lower bound on every pair during IK

# name: (flange position (x, z), flange z-axis direction (x, z), description).
# Targets were chosen from a reachability sweep (shoulder at z = 0.34, link_7
# origin 0.45-0.88 m from it) keeping >= 25 mm self clearance.
TARGETS = {
    "L": ((0.50, 0.74), (1.0, 0.0), "upper arm vertical, forearm horizontal"),
    "SEVEN": ((0.40, 0.10), (0.0, -1.0), "arm leaning forward, forearm hanging down"),
    "Z": ((-0.20, 1.00), (-np.sqrt(0.5), np.sqrt(0.5)), "forearm folded back over the shoulder"),
    "U": ((0.40, 0.85), (0.0, 1.0), "arm leaning forward, forearm turned back up"),
    "SLASH": ((0.55, 1.00), (np.sqrt(0.5), np.sqrt(0.5)), "gentle diagonal reach"),
}


def solve_ik(scene, target, direction, guesses):
    plant, ctx = scene.plant, scene.plant_context
    flange = plant.GetFrameByName("iiwa_link_7")
    best = None
    for q_guess in guesses:
        ik = InverseKinematics(plant, ctx)
        p = np.array([target[0], 0.0, target[1]])
        ik.AddPositionConstraint(flange, np.zeros(3), plant.world_frame(), p - 1e-4, p + 1e-4)
        n = np.array([direction[0], 0.0, direction[1]])
        ik.AddAngleBetweenVectorsConstraint(plant.world_frame(), n, flange, np.array([0, 0, 1.0]), 0.0, 0.01)
        ik.AddMinimumDistanceLowerBoundConstraint(SELF_CLEARANCE, 0.05)
        prog, q = ik.prog(), ik.q()
        prog.AddQuadraticCost((q - q_guess) @ (q - q_guess))
        prog.SetInitialGuess(q, q_guess)
        result = SnoptSolver().Solve(prog, None, SolverOptions())
        if result.is_success():
            qs = result.GetSolution(q)
            cost = float(np.abs(qs).sum())
            if best is None or cost < best[1]:
                best = (qs, cost)
    return None if best is None else best[0]


def main():
    scene = make_scene([], beta=0.0, fixed_wall_y=0.0)
    lo, hi = scene.free_limits()
    rng = np.random.default_rng(0)
    guesses = [np.zeros(3)] + [rng.uniform(lo, hi) for _ in range(20)]
    print("LETTER_POSES = {")
    for name, (target, direction, desc) in TARGETS.items():
        q_free = solve_ik(scene, target, direction, guesses)
        if q_free is None:
            print(f"    # {name}: IK failed")
            continue
        q = scene.q_full(q_free)
        d_self = min_distance(scene, q, 0.0)
        minx, minz, maxx, maxz = silhouette_polygon(q).bounds
        print(f"    # {desc}; self clearance {1e3*d_self:.1f} mm; silhouette x[{minx:.2f},{maxx:.2f}] z[{minz:.2f},{maxz:.2f}]")
        print(f'    "{name}": np.array([0.0, {q[1]:.4f}, 0.0, {q[3]:.4f}, 0.0, {q[5]:.4f}, 0.0]),')
    print("}")


if __name__ == "__main__":
    main()
