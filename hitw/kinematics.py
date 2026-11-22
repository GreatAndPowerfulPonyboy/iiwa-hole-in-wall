"""Arm geometry without a wall: box corners at q, sampled points, joint radii
and the motion bound rho.  One bare 7-DOF plant (no welds) is shared so these
do not depend on the beta a scene is built with.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    DiagramBuilder,
    Parser,
    RigidTransform,
)

from .params import IIWA_URL, NUM_JOINTS

_CORNER_SIGNS = np.array(list(product([-1.0, 1.0], repeat=3)))  # (8, 3)


class BareArm:
    """The iiwa7 alone, welded at the origin, with a held-open context."""

    def __init__(self) -> None:
        builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        (self.iiwa,) = Parser(self.plant).AddModelsFromUrl(IIWA_URL)
        self.plant.WeldFrames(self.plant.world_frame(), self.plant.GetFrameByName("iiwa_link_0"))
        self.plant.Finalize()
        self.diagram = builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)
        self.sg_context = self.scene_graph.GetMyContextFromRoot(self.context)
        self.inspector = self.scene_graph.model_inspector()
        # (body, geometry_id, half-extents) for every collision box of the arm.
        self.boxes = []
        for bi in self.plant.GetBodyIndices(self.iiwa):
            body = self.plant.get_body(bi)
            for gid in self.plant.GetCollisionGeometriesForBody(body):
                shape = self.inspector.GetShape(gid)
                assert isinstance(shape, Box), "silhouette and motion bound assume boxes"
                half = np.array([shape.width(), shape.depth(), shape.height()]) / 2
                self.boxes.append((body, gid, half))

    def set_q(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=float).ravel()
        assert q.shape == (NUM_JOINTS,), q.shape
        self.plant.SetPositions(self.plant_context, self.iiwa, q)

    def box_poses(self, q: np.ndarray) -> list[tuple[str, RigidTransform, np.ndarray]]:
        """(body name, X_WBox, half-extents) for every arm collision box at q."""
        self.set_q(q)
        qo = self.scene_graph.get_query_output_port().Eval(self.sg_context)
        return [(body.name(), qo.GetPoseInWorld(gid), half) for body, gid, half in self.boxes]

    def corners(self, q: np.ndarray) -> dict[str, np.ndarray]:
        """World corners of every collision box, grouped by body: {name: (8k, 3)}."""
        out: dict[str, list[np.ndarray]] = {}
        for name, X, half in self.box_poses(q):
            pts = (X.rotation().matrix() @ (half * _CORNER_SIGNS).T).T + X.translation()
            out.setdefault(name, []).append(pts)
        return {k: np.vstack(v) for k, v in out.items()}

    def all_corners(self, q: np.ndarray) -> np.ndarray:
        return np.vstack(list(self.corners(q).values()))

    def sample_points(self, q: np.ndarray, n_per_box: int, rng: np.random.Generator) -> np.ndarray:
        """Points drawn uniformly inside every collision box at q, plus the
        corners (the extreme points), in world coordinates: (N, 3)."""
        pts = []
        for _, X, half in self.box_poses(q):
            local = rng.uniform(-half, half, size=(n_per_box, 3))
            local = np.vstack([local, half * _CORNER_SIGNS])
            pts.append((X.rotation().matrix() @ local.T).T + X.translation())
        return np.vstack(pts)

    def joint_radii(self) -> np.ndarray:
        """r[j]: bound on the distance from joint j+1's axis to every point it
        moves, valid at every configuration.  Corners of the driven link are
        measured from the axis; later links are bounded by straightening the
        chain."""
        self.set_q(np.zeros(NUM_JOINTS))
        origins, axes, corners = [], [], []
        per_body = self.corners(np.zeros(NUM_JOINTS))
        for j in range(NUM_JOINTS):
            joint = self.plant.GetJointByName(f"iiwa_joint_{j + 1}")
            X_WJ = joint.frame_on_child().CalcPoseInWorld(self.plant_context)
            origins.append(X_WJ.translation())
            axes.append(X_WJ.rotation() @ joint.revolute_axis())
            corners.append(per_body[joint.child_body().name()])

        def dist_to_line(p, o, a):
            v = p - o
            return float(np.linalg.norm(v - (v @ a) * a))

        reach = np.zeros(NUM_JOINTS)
        radii = np.zeros(NUM_JOINTS)
        for j in reversed(range(NUM_JOINTS)):
            reach[j] = max(np.linalg.norm(c - origins[j]) for c in corners[j])
            radii[j] = max(dist_to_line(c, origins[j], axes[j]) for c in corners[j])
            if j + 1 < NUM_JOINTS:
                link_length = np.linalg.norm(origins[j + 1] - origins[j])
                reach[j] = max(reach[j], link_length + reach[j + 1])
                radii[j] = max(radii[j], dist_to_line(origins[j + 1], origins[j], axes[j]) + reach[j + 1])
        return radii


@lru_cache(maxsize=1)
def bare_arm() -> BareArm:
    return BareArm()


@lru_cache(maxsize=1)
def joint_radii() -> np.ndarray:
    return bare_arm().joint_radii()


def motion_bound(radii: np.ndarray, free_joints=None) -> float:
    """rho: no point of the arm travels farther than rho * ||q - q'||_2 when
    the configuration moves from q to q'.

    Rotating the joints one at a time moves any point by at most
    sum_j r_j |dq_j| <= ||r||_2 ||dq||_2.  With joints frozen (beta = 0) only
    the free joints' radii count.
    """
    radii = np.asarray(radii, dtype=float)
    if free_joints is not None:
        radii = radii[list(free_joints)]
    return float(np.linalg.norm(radii))


def arm_depth(q: np.ndarray) -> float:
    """d: the arm's full extent along y at q (twice the largest |y| of any
    collision-box corner)."""
    return float(2.0 * np.abs(bare_arm().all_corners(q)[:, 1]).max())
