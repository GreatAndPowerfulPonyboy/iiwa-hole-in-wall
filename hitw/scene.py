"""The scene: iiwa7 welded at the origin, roll joints banded to |q| <= beta,
and every wall slab of a sequence carried by one rigid body on a prismatic
joint along world y.

Drake skips collision pairs between two bodies both anchored to the world, so
a welded wall would be invisible to distance queries against the base link.
With `fixed_wall_y` the wall is welded (for IRIS / IK with a frozen wall);
only the link_0 pair is lost then, and link_0 never moves, so the base cutout
is checked in the prismatic scene (tests/test_walls.py).

beta = 0 replaces the roll joints with WeldJoints before Finalize (3-D
configuration space).  beta > 0 keeps all seven joints and sets the roll
joints' position limits to [-beta, beta] so IK, IRIS domains and GCS bounds
see the band.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    CoulombFriction,
    Diagram,
    DiagramBuilder,
    Meshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    MultibodyPlant,
    Parser,
    PrismaticJoint,
    RigidTransform,
    RobotDiagramBuilder,
    SceneGraph,
    SpatialInertia,
    UnitInertia,
    WeldJoint,
)

from . import params
from .kinematics import joint_radii, motion_bound
from .walls import WallSpec

CLEARANCE_CAP = 0.5
WALL_COLOR = np.array([0.55, 0.6, 0.75, 0.85])


def freeze_joint(plant: MultibodyPlant, joint) -> None:
    """Replace a joint with a weld at its zero pose, and remove its actuator."""
    for i in plant.GetJointActuatorIndices():
        actuator = plant.get_joint_actuator(i)
        if actuator.joint() == joint:
            plant.RemoveJointActuator(actuator)
            break
    weld = WeldJoint(joint.name(), joint.frame_on_parent(), joint.frame_on_child(), RigidTransform())
    plant.RemoveJoint(joint)
    plant.AddJoint(weld)


@dataclass
class Scene:
    diagram: Diagram
    plant: MultibodyPlant
    scene_graph: SceneGraph
    iiwa: object                      # ModelInstanceIndex
    wall_body: object                 # RigidBody
    wall_joint: object | None         # PrismaticJoint, or None when welded
    context: object
    plant_context: object
    sg_context: object
    beta: float
    free_joints: tuple[int, ...]      # 0-based indices into the 7-vector
    walls: list[WallSpec]
    robot_ids: set
    fixed_wall_y: float | None
    meshcat: Meshcat | None = None
    visualizer: object | None = None
    rho: float = 0.0
    _joints: list = field(default_factory=list)

    # --- configuration conversions -------------------------------------
    @property
    def num_free(self) -> int:
        return len(self.free_joints)

    def q_free(self, q_full: np.ndarray) -> np.ndarray:
        q_full = np.asarray(q_full, dtype=float).ravel()
        return q_full[list(self.free_joints)]

    def q_full(self, q_free: np.ndarray) -> np.ndarray:
        q = np.zeros(params.NUM_JOINTS)
        q[list(self.free_joints)] = np.asarray(q_free, dtype=float).ravel()
        return q

    def arm_position_indices(self) -> list[int]:
        """Plant position index of each free joint, in free-joint order."""
        return [self._joints[j].position_start() for j in self.free_joints]

    def free_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([self._joints[j].position_lower_limits()[0] for j in self.free_joints])
        hi = np.array([self._joints[j].position_upper_limits()[0] for j in self.free_joints])
        return lo, hi

    def velocity_limits(self) -> np.ndarray:
        """Imposed per-joint speed limits for the free joints (LIMIT_SCALE x model)."""
        return params.LIMIT_SCALE * np.array(
            [self._joints[j].velocity_upper_limits()[0] for j in self.free_joints])

    def acceleration_limits(self) -> np.ndarray:
        return params.LIMIT_SCALE * np.array(
            [self._joints[j].acceleration_upper_limits()[0] for j in self.free_joints])

    def check_q(self, q_full: np.ndarray) -> None:
        q_full = np.asarray(q_full, dtype=float).ravel()
        assert q_full.shape == (params.NUM_JOINTS,), q_full.shape
        for j in params.ROLL_JOINTS:
            if abs(q_full[j]) > self.beta + 1e-9:
                raise ValueError(f"roll joint {j + 1} = {q_full[j]:.4f} exceeds beta = {self.beta}")

    def publish(self) -> None:
        if self.meshcat is not None:
            self.diagram.ForcedPublish(self.context)


def make_scene(
    walls: Sequence[WallSpec] = (),
    beta: float = 0.0,
    *,
    fixed_wall_y: float | None = None,
    meshcat: Meshcat | None = None,
    time_step: float = 0.0,
    robot_diagram: bool = False,
) -> Scene:
    """robot_diagram=True builds a RobotDiagram (RobotDiagramBuilder) so the
    scene can back a SceneGraphCollisionChecker for IRIS."""
    if robot_diagram:
        rdb = RobotDiagramBuilder(time_step=time_step)
        builder, plant, scene_graph, parser = rdb.builder(), rdb.plant(), rdb.scene_graph(), rdb.parser()
    else:
        rdb = None
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
        parser = Parser(plant)
    (iiwa,) = parser.AddModelsFromUrl(params.IIWA_URL)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))

    joints = [plant.GetJointByName(f"iiwa_joint_{j + 1}") for j in range(params.NUM_JOINTS)]
    if beta <= 0.0:
        for j in params.ROLL_JOINTS:
            freeze_joint(plant, joints[j])
        free_joints = tuple(params.PITCH_JOINTS)
    else:
        for j in params.ROLL_JOINTS:
            joints[j].set_position_limits(np.array([-beta]), np.array([beta]))
        free_joints = tuple(range(params.NUM_JOINTS))

    # One carrier body for every slab.
    wall_model = plant.AddModelInstance("wall")
    wall = plant.AddRigidBody("wall", wall_model, SpatialInertia(1.0, np.zeros(3), UnitInertia(1, 1, 1)))
    if fixed_wall_y is None:
        wall_joint = plant.AddJoint(PrismaticJoint("wall_y", plant.world_frame(), wall.body_frame(), [0, 1, 0]))
    else:
        wall_joint = None
        plant.WeldFrames(plant.world_frame(), wall.body_frame(), RigidTransform([0, fixed_wall_y, 0]))
    friction = CoulombFriction(0.9, 0.5)
    for wi, spec in enumerate(walls):
        for bi, (cx, cz, w, h) in enumerate(spec.boxes):
            X = RigidTransform([cx, spec.y_offset, cz])
            shape = Box(w, spec.thickness, h)
            plant.RegisterCollisionGeometry(wall, X, shape, f"wall{wi}_box{bi}", friction)
            plant.RegisterVisualGeometry(wall, X, shape, f"wall{wi}_vbox{bi}", WALL_COLOR)
    plant.Finalize()

    visualizer = None
    if meshcat is not None:
        visualizer = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, meshcat,
            MeshcatVisualizerParams(delete_on_initialization_event=False))

    diagram = rdb.Build() if rdb is not None else builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    sg_context = scene_graph.GetMyContextFromRoot(context)
    # Re-fetch joints: freeze_joint replaced some with welds.
    joints = [plant.GetJointByName(f"iiwa_joint_{j + 1}") for j in range(params.NUM_JOINTS)]
    robot_ids = {gid for bi in plant.GetBodyIndices(iiwa)
                 for gid in plant.GetCollisionGeometriesForBody(plant.get_body(bi))}
    scene = Scene(
        diagram=diagram, plant=plant, scene_graph=scene_graph, iiwa=iiwa, wall_body=wall,
        wall_joint=wall_joint, context=context, plant_context=plant_context, sg_context=sg_context,
        beta=beta, free_joints=free_joints, walls=list(walls), robot_ids=robot_ids,
        fixed_wall_y=fixed_wall_y, meshcat=meshcat, visualizer=visualizer,
        rho=motion_bound(joint_radii(), free_joints), _joints=joints,
    )
    return scene


def set_state(scene: Scene, q: np.ndarray, wall_y: float) -> None:
    """Put the arm at the full 7-vector q (frozen joints must be ~0) and the
    wall carrier at translation wall_y."""
    q = np.asarray(q, dtype=float).ravel()
    for j in scene.free_joints:
        scene._joints[j].set_angle(scene.plant_context, float(q[j]))
    if scene.wall_joint is not None:
        scene.wall_joint.set_translation(scene.plant_context, float(wall_y))


def signed_distance_pairs(scene: Scene, q: np.ndarray, wall_y: float, cap: float = CLEARANCE_CAP):
    """All (distance, name_A, name_B) within cap: robot-vs-wall and arm
    self pairs (adjacent-link pairs are filtered by the model)."""
    set_state(scene, q, wall_y)
    qo = scene.scene_graph.get_query_output_port().Eval(scene.sg_context)
    insp = scene.scene_graph.model_inspector()
    out = []
    for p in qo.ComputeSignedDistancePairwiseClosestPoints(cap):
        a_robot, b_robot = p.id_A in scene.robot_ids, p.id_B in scene.robot_ids
        if not (a_robot or b_robot):
            continue
        out.append((float(p.distance),
                    scene.plant.GetBodyFromFrameId(insp.GetFrameId(p.id_A)).name(),
                    scene.plant.GetBodyFromFrameId(insp.GetFrameId(p.id_B)).name()))
    return out


def min_distance(scene: Scene, q: np.ndarray, wall_y: float, cap: float = CLEARANCE_CAP) -> float:
    """The shared distance function: smallest signed distance between an arm
    geometry and any other geometry (wall or another arm link), capped at
    `cap`.  Negative means penetration."""
    set_state(scene, q, wall_y)
    qo = scene.scene_graph.get_query_output_port().Eval(scene.sg_context)
    best = cap
    for p in qo.ComputeSignedDistancePairwiseClosestPoints(cap):
        if (p.id_A in scene.robot_ids or p.id_B in scene.robot_ids) and p.distance < best:
            best = p.distance
    return float(best)


def in_collision(scene: Scene, q: np.ndarray, wall_y: float) -> bool:
    """min_distance(...) < 0 without the distance; about 20x cheaper."""
    set_state(scene, q, wall_y)
    qo = scene.scene_graph.get_query_output_port().Eval(scene.sg_context)
    for p in qo.ComputePointPairPenetration():
        if p.id_A in scene.robot_ids or p.id_B in scene.robot_ids:
            return True
    return False


def wall_y_of_t(walls: Sequence[WallSpec], speed: float = params.WALL_SPEED):
    """Carrier translation as a function of time, given each slab's t_cross
    and y_offset: wall_y(t) = speed * (t - t_1), so slab i (offset
    -speed * (t_i - t_1)) is centred on the arm plane exactly at t_i."""
    t1 = walls[0].t_cross
    return lambda t: speed * (t - t1)


def place_walls(walls: Sequence[WallSpec], t_cross: Sequence[float], speed: float = params.WALL_SPEED) -> None:
    """Assign crossing times and the matching rigid y offsets on the carrier."""
    t1 = t_cross[0]
    for spec, t in zip(walls, t_cross):
        spec.t_cross = float(t)
        spec.y_offset = -speed * (t - t1)
