"""Physical parameters approved at milestone 0 (2026-11-13). See PROGRESS.md."""

import numpy as np

IIWA_URL = "package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf"
NUM_JOINTS = 7

# Joint groups (0-based indices).  Roll joints are banded to |q| <= beta,
# pitch joints are free within their limits.  With the roll joints at zero the
# three pitch axes are parallel to world y and the arm is planar in x-z.
ROLL_JOINTS = (0, 2, 4, 6)
PITCH_JOINTS = (1, 3, 5)

# Wall slab: the slab plane is the x-z plane, translating along world y.
WALL_THICKNESS = 0.05       # t, m
WALL_SPEED = 0.5            # v, m/s
SLAB_WIDTH = 2.0            # x in [-SLAB_WIDTH/2, SLAB_WIDTH/2]
SLAB_Z0 = 0.0               # bottom of the slab (floor)
SLAB_HEIGHT = 1.6           # z in [SLAB_Z0, SLAB_Z0 + SLAB_HEIGHT]
STRIP_HEIGHT = 0.01         # default strip height when cutting the slab, m

# Timing: first crossing at T_FIRST, consecutive crossings WINDOW_SPACING
# apart, trajectory ends T_TAIL after the last crossing.
T_FIRST = 3.0
WINDOW_SPACING = 3.0
T_TAIL = 3.0

# Imposed joint velocity and acceleration limits: this fraction of the model's.
LIMIT_SCALE = 0.5

# Report grid.
BETAS = (0.0, 0.1)
EPS_GRID = tuple(float(e) for e in np.geomspace(0.02, 0.002, 6))   # 20 mm .. 2 mm
N_WALLS = (1, 2)

# The window half-width in time: delta = (t + d) / (2 v), with arm depth d
# computed from the crossing configuration (see kinematics.arm_depth).
def window_half_width(arm_depth: float, thickness: float = WALL_THICKNESS,
                      speed: float = WALL_SPEED) -> float:
    return (thickness + arm_depth) / (2.0 * speed)


# Joint-space tolerance inside a window: eps / (2 rho).  The factor of 2 leaves
# a worst-case clearance of eps / 2 when the arm deviates by the full tolerance.
TOLERANCE_FACTOR = 2.0


def window_tolerance(eps: float, rho: float) -> float:
    return eps / (TOLERANCE_FACTOR * rho)
