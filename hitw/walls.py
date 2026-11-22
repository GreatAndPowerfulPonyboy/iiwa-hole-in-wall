"""Wall construction: the buffered x-z silhouette of the arm at a crossing
configuration, cut out of a slab that is emitted as axis-aligned boxes.

Clearance argument.  For an arm point a and a wall point w,
    ||a - w|| >= ||pi(a) - pi(w)|| >= eps,
where pi drops the y coordinate: pi(a) lies in the silhouette U (union of the
per-link convex hulls of the projected box corners), pi(w) lies in a wall box,
every wall box lies outside the opening O = U (+) disc(eps), and the distance
from U to the complement of O is eps.  Hence the realized clearance at q_star
is at least eps.  Off q_star the guarantee degrades by rho * ||q - q_star||_2
(see kinematics.motion_bound), so within the window tolerance eps / (2 rho)
the clearance stays at least eps / 2.

Strip cutting.  The slab is cut into horizontal strips.  Within a strip the
material is the complement of the opening's x-projection, so every box is the
full strip height and lies entirely outside the opening (rounded outward,
adding up to one strip's worth of silhouette slope as slack).  Vertically
adjacent boxes with the same x-interval are merged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import MultiPoint, Polygon, box as shapely_box
from shapely.ops import unary_union

from . import params
from .kinematics import arm_depth, bare_arm

# The buffer polygon is inscribed in the true disc; scaling the radius by
# 1/cos(pi / (2 quad_segs)) makes it circumscribe it, so O contains the exact
# Minkowski sum and the clearance bound is not undercut by polygonization.
BUFFER_QUAD_SEGS = 16
_BUFFER_SCALE = 1.0 / np.cos(np.pi / (2 * BUFFER_QUAD_SEGS))


@dataclass
class WallSpec:
    q_star: np.ndarray                 # full 7-vector crossing configuration
    eps: float                         # nominal clearance, m
    strip: float                       # strip height used for cutting, m
    thickness: float                   # t, m (along y)
    width: float                       # slab width (x)
    z0: float                          # slab bottom (z)
    height: float                      # slab height (z)
    boxes: list[tuple[float, float, float, float]]   # (cx, cz, w, h) in the slab frame
    silhouette: Polygon                # U, in x-z
    opening: Polygon                   # O = U buffered by eps, in x-z
    arm_depth: float                   # d at q_star
    y_offset: float = 0.0              # slab plane offset on the carrier body
    t_cross: float | None = None       # crossing time, set by the sequence
    extras: dict = field(default_factory=dict)

    @property
    def n_boxes(self) -> int:
        return len(self.boxes)


def silhouette_polygon(q_star: np.ndarray) -> Polygon:
    """Union of the per-link convex hulls of the projected collision-box
    corners, in x-z."""
    hulls = []
    for pts in bare_arm().corners(q_star).values():
        hulls.append(MultiPoint([(p[0], p[2]) for p in pts]).convex_hull)
    return unary_union(hulls)


def _merge_intervals(intervals):
    intervals = sorted(intervals)
    merged = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _strip_material_intervals(opening, x_lo, x_hi, z_lo, z_hi):
    """x-intervals of material (outside the opening) within one strip."""
    strip = shapely_box(x_lo, z_lo, x_hi, z_hi)
    inter = opening.intersection(strip)
    if inter.is_empty:
        return [(x_lo, x_hi)]
    geoms = getattr(inter, "geoms", [inter])
    holes = _merge_intervals(
        [(g.bounds[0], g.bounds[2]) for g in geoms if not g.is_empty and g.area > 0]
    )
    material = []
    cursor = x_lo
    for lo, hi in holes:
        if lo > cursor:
            material.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < x_hi:
        material.append((cursor, x_hi))
    return material


def make_wall(
    q_star: np.ndarray,
    eps: float,
    strip: float = params.STRIP_HEIGHT,
    thickness: float = params.WALL_THICKNESS,
    width: float = params.SLAB_WIDTH,
    z0: float = params.SLAB_Z0,
    height: float = params.SLAB_HEIGHT,
    merge: bool = True,
) -> WallSpec:
    q_star = np.asarray(q_star, dtype=float).ravel()
    U = silhouette_polygon(q_star)
    O = U.buffer(eps * _BUFFER_SCALE, quad_segs=BUFFER_QUAD_SEGS)
    x_lo, x_hi = -width / 2, width / 2
    minx, minz, maxx, maxz = O.bounds
    if not (minx > x_lo and maxx < x_hi and maxz < z0 + height):
        raise ValueError(
            f"opening {O.bounds} is not strictly inside the slab "
            f"x[{x_lo},{x_hi}] z[{z0},{z0 + height}]: the slab edge would constrain the arm"
        )
    n_strips = int(np.ceil(height / strip))
    columns: list[tuple[float, float, float, float]] = []   # (xlo, xhi, zlo, zhi)
    for i in range(n_strips):
        z_lo = z0 + i * strip
        z_hi = min(z0 + (i + 1) * strip, z0 + height)
        for a, b in _strip_material_intervals(O, x_lo, x_hi, z_lo, z_hi):
            if merge and columns:
                # merge with a box directly below that has the same x-interval
                for k in range(len(columns) - 1, -1, -1):
                    xa, xb, za, zb = columns[k]
                    if abs(zb - z_lo) < 1e-12 and abs(xa - a) < 1e-12 and abs(xb - b) < 1e-12:
                        columns[k] = (xa, xb, za, z_hi)
                        break
                    if zb < z_lo - 1e-12:
                        columns.append((a, b, z_lo, z_hi))
                        break
                else:
                    columns.append((a, b, z_lo, z_hi))
            else:
                columns.append((a, b, z_lo, z_hi))
    boxes = [((a + b) / 2, (za + zb) / 2, b - a, zb - za) for a, b, za, zb in columns]
    return WallSpec(
        q_star=q_star, eps=float(eps), strip=strip, thickness=thickness, width=width,
        z0=z0, height=height, boxes=boxes, silhouette=U, opening=O,
        arm_depth=arm_depth(q_star),
    )


def wall_boxes_polygon(spec: WallSpec):
    """The material as a shapely geometry (for tests and plots)."""
    return unary_union([shapely_box(cx - w / 2, cz - h / 2, cx + w / 2, cz + h / 2)
                        for cx, cz, w, h in spec.boxes])


def planar_clearance(spec: WallSpec, points_xyz: np.ndarray) -> np.ndarray:
    """x-z distance from each point to the nearest wall box: the lower bound
    ||pi(a) - pi(w)|| of the clearance argument."""
    P = np.asarray(points_xyz)[:, [0, 2]]                 # (N, 2)
    B = np.asarray(spec.boxes)                            # (M, 4): cx, cz, w, h
    lo = B[:, :2] - B[:, 2:] / 2                          # (M, 2)
    hi = B[:, :2] + B[:, 2:] / 2
    # per point, per box: distance to axis-aligned rectangle
    d = np.maximum(0.0, np.maximum(lo[None] - P[:, None], P[:, None] - hi[None]))  # (N, M, 2)
    return np.sqrt((d ** 2).sum(-1)).min(axis=1)


def wall_sdf(spec: WallSpec, model_name: str = "wall") -> str:
    """SDF for the wall as one link carrying every box on a prismatic y joint.
    make_scene registers the boxes directly and does not use this."""
    parts = []
    for i, (cx, cz, w, h) in enumerate(spec.boxes):
        parts.append(f"""
      <collision name="box_{i}"><pose>{cx} {spec.y_offset} {cz} 0 0 0</pose>
        <geometry><box><size>{w} {spec.thickness} {h}</size></box></geometry></collision>
      <visual name="vbox_{i}"><pose>{cx} {spec.y_offset} {cz} 0 0 0</pose>
        <geometry><box><size>{w} {spec.thickness} {h}</size></box></geometry></visual>""")
    return f"""<?xml version="1.0"?>
<sdf version="1.7"><model name="{model_name}"><link name="wall">{''.join(parts)}
    </link>
    <joint name="wall_y" type="prismatic"><parent>world</parent><child>wall</child>
      <axis><xyz>0 1 0</xyz></axis></joint>
</model></sdf>"""
