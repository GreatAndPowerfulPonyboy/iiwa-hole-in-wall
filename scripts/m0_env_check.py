"""Milestone 0: environment, model loading, and distance-query cost.

Run:  .venv/bin/python scripts/m0_env_check.py

Prints Drake version, solver availability, loads the iiwa7 (box collision
geometry), builds a wall body on a prismatic y joint carrying N strip boxes,
runs one signed-distance query at q = 0 against a single test box, then times
min-distance queries against the strip wall and projects dense re-check cost.
"""

import importlib.metadata as md
import platform
import sys
import time
from itertools import product

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    ClarabelSolver,
    CollisionCheckerParams,
    CoulombFriction,
    DiagramBuilder,
    GurobiSolver,
    IpoptSolver,
    MosekSolver,
    Parser,
    PrismaticJoint,
    RigidTransform,
    RobotDiagramBuilder,
    SceneGraphCollisionChecker,
    SnoptSolver,
    SpatialInertia,
    UnitInertia,
)

IIWA_URL = "package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf"
NUM_JOINTS = 7
CAP = 0.5


def header(s):
    print(f"\n=== {s} ===")


def report_env():
    header("environment")
    print("python", sys.version.split()[0], platform.platform())
    for p in ["drake", "manipulation", "numpy", "scipy", "shapely", "matplotlib", "pytest"]:
        try:
            print(f"  {p:14s} {md.version(p)}")
        except md.PackageNotFoundError:
            print(f"  {p:14s} NOT INSTALLED")
    header("solvers (available / enabled)")
    for S in [SnoptSolver, IpoptSolver, MosekSolver, ClarabelSolver, GurobiSolver]:
        s = S()
        print(f"  {S.__name__:14s} {str(s.available()):6s} {s.enabled()}")


def add_wall(plant, n_strips, strip_h, slab_w, slab_t, z0, gap_lo, gap_hi, gap_x):
    """One wall body on a prismatic joint along world y, carrying strip boxes.

    Each strip is split into a left and a right box around an opening between
    x = gap_x - half-gap and gap_x + half-gap for strips whose z lies in
    [gap_lo, gap_hi]; other strips are a single full-width box.  Returns the
    body and the number of boxes registered.
    """
    M = SpatialInertia(1.0, np.zeros(3), UnitInertia(1, 1, 1))
    wall_model = plant.AddModelInstance("wall")
    wall = plant.AddRigidBody("wall", wall_model, M)
    joint = plant.AddJoint(
        PrismaticJoint("wall_y", plant.world_frame(), wall.body_frame(), [0, 1, 0])
    )
    n_boxes = 0
    friction = CoulombFriction(0.9, 0.5)
    for i in range(n_strips):
        z = z0 + (i + 0.5) * strip_h
        if gap_lo <= z <= gap_hi:
            half_gap = 0.12
            w_l = (gap_x - half_gap) - (-slab_w / 2)
            x_l = -slab_w / 2 + w_l / 2
            plant.RegisterCollisionGeometry(
                wall, RigidTransform([x_l, 0, z]), Box(w_l, slab_t, strip_h),
                f"strip_{i}_L", friction)
            w_r = slab_w / 2 - (gap_x + half_gap)
            x_r = (gap_x + half_gap) + w_r / 2
            plant.RegisterCollisionGeometry(
                wall, RigidTransform([x_r, 0, z]), Box(w_r, slab_t, strip_h),
                f"strip_{i}_R", friction)
            n_boxes += 2
        else:
            plant.RegisterCollisionGeometry(
                wall, RigidTransform([0, 0, z]), Box(slab_w, slab_t, strip_h),
                f"strip_{i}", friction)
            n_boxes += 1
    return wall, joint, n_boxes


def build(n_strips=0, test_box=False):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    (iiwa,) = parser.AddModelsFromUrl(IIWA_URL)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))
    wall = joint = None
    n_boxes = 0
    if test_box:
        M = SpatialInertia(1.0, np.zeros(3), UnitInertia(1, 1, 1))
        wall = plant.AddRigidBody("wall", plant.AddModelInstance("wall"), M)
        joint = plant.AddJoint(
            PrismaticJoint("wall_y", plant.world_frame(), wall.body_frame(), [0, 1, 0]))
        plant.RegisterCollisionGeometry(
            wall, RigidTransform([0.0, 0, 0.6]), Box(1.0, 0.05, 1.0), "test_box",
            CoulombFriction(0.9, 0.5))
        n_boxes = 1
    elif n_strips:
        # 1.2 m tall slab of 1 cm strips, 1.5 m wide, 5 cm thick, opening for
        # the upright arm between z = 0.3 and 1.3 near x = 0.
        wall, joint, n_boxes = add_wall(
            plant, n_strips, strip_h=1.2 / n_strips, slab_w=1.5, slab_t=0.05,
            z0=0.0, gap_lo=0.25, gap_hi=1.3, gap_x=0.0)
    plant.Finalize()
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    sg_context = scene_graph.GetMyContextFromRoot(context)
    return dict(diagram=diagram, plant=plant, scene_graph=scene_graph, iiwa=iiwa,
                wall=wall, joint=joint, n_boxes=n_boxes, context=context,
                plant_context=plant_context, sg_context=sg_context)


def robot_ids(s):
    plant = s["plant"]
    return {gid for bi in plant.GetBodyIndices(s["iiwa"])
            for gid in plant.GetCollisionGeometriesForBody(plant.get_body(bi))}


def min_distance(s, q, wall_y, robot, cap=CAP):
    plant = s["plant"]
    plant.SetPositions(s["plant_context"], s["iiwa"], np.asarray(q, float).ravel())
    if s["joint"] is not None:
        s["joint"].set_translation(s["plant_context"], wall_y)
    qo = s["scene_graph"].get_query_output_port().Eval(s["sg_context"])
    pairs = qo.ComputeSignedDistancePairwiseClosestPoints(cap)
    ds = [p.distance for p in pairs]
    return min(ds, default=cap), len(pairs)


def model_facts(s):
    header("iiwa7 model facts")
    plant, insp = s["plant"], s["scene_graph"].model_inspector()
    print("  positions:", plant.num_positions(), " iiwa positions:", plant.num_positions(s["iiwa"]))
    lo, hi = plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
    vlo, vhi = plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits()
    alo, ahi = plant.GetAccelerationLowerLimits(), plant.GetAccelerationUpperLimits()
    for j in range(NUM_JOINTS):
        print(f"  joint {j+1}: q in [{lo[j]:+.3f}, {hi[j]:+.3f}] rad  "
              f"v <= {vhi[j]:.3f} rad/s  a <= {ahi[j]}")
    all_boxes = True
    plant.SetPositions(s["plant_context"], s["iiwa"], np.zeros(NUM_JOINTS))
    qo = s["scene_graph"].get_query_output_port().Eval(s["sg_context"])
    ymax = 0.0
    zmax = 0.0
    for bi in plant.GetBodyIndices(s["iiwa"]):
        body = plant.get_body(bi)
        for gid in plant.GetCollisionGeometriesForBody(body):
            shape = insp.GetShape(gid)
            is_box = isinstance(shape, Box)
            all_boxes &= is_box
            dims = (f"{shape.width():.3f}x{shape.depth():.3f}x{shape.height():.3f}"
                    if is_box else type(shape).__name__)
            X = qo.GetPoseInWorld(gid)
            if is_box:
                half = np.array([shape.width(), shape.depth(), shape.height()]) / 2
                for signs in product([-1, 1], repeat=3):
                    c = X @ (half * signs)
                    ymax = max(ymax, abs(c[1]))
                    zmax = max(zmax, c[2])
            print(f"  {body.name():12s} {type(shape).__name__:8s} {dims}")
    print(f"  all collision geometries are boxes: {all_boxes}")
    print(f"  at q=0: arm half-depth in y (max |y| of any corner) = {ymax:.4f} m, "
          f"top z = {zmax:.4f} m")


def main():
    report_env()

    header("load iiwa + one distance query at q = 0 against a test box")
    t0 = time.perf_counter()
    s = build(test_box=True)
    print(f"  built plant with iiwa + test box in {time.perf_counter()-t0:.2f} s")
    model_facts(s)
    robot = robot_ids(s)
    d, npairs = min_distance(s, np.zeros(7), wall_y=0.5, robot=robot)
    print(f"  wall_y = 0.50: min signed distance = {d:+.4f} m over {npairs} pairs")
    d, npairs = min_distance(s, np.zeros(7), wall_y=0.0, robot=robot)
    print(f"  wall_y = 0.00 (box through the arm): min signed distance = {d:+.4f} m over {npairs} pairs")

    header("nanobind check: (7,1) array to SetPositions")
    try:
        s["plant"].SetPositions(s["plant_context"], s["iiwa"], np.zeros((7, 1)))
        print("  (7,1) accepted")
    except TypeError as e:
        print("  (7,1) rejected with TypeError:", str(e).splitlines()[0][:100])
        print("  -> always pass .ravel()'d 1-D vectors")

    header("timing: min_distance against a strip wall")
    for n_strips in (120, 200):
        s = build(n_strips=n_strips)
        robot = robot_ids(s)
        q = np.array([0, 0.3, 0, -1.2, 0, 0.9, 0])
        print(f"  wall: {n_strips} strips -> {s['n_boxes']} boxes")
        for cap in (0.5, 0.1, 0.02):
            for wy, label in ((0.0, "wall at arm"), (1.0, "wall 1 m away")):
                min_distance(s, q, wy, robot, cap)  # warm
                n = 200
                t0 = time.perf_counter()
                for _ in range(n):
                    d, npairs = min_distance(s, q, wy, robot, cap)
                dt = (time.perf_counter() - t0) / n
                print(f"    cap={cap:4.2f} {label:14s}: {1e3*dt:6.2f} ms/query, "
                      f"{npairs:4d} pairs, min d = {d:+.4f}")
        s["plant"].SetPositions(s["plant_context"], s["iiwa"], q)
        s["joint"].set_translation(s["plant_context"], 0.0)
        qo_port = s["scene_graph"].get_query_output_port()
        n = 500
        t0 = time.perf_counter()
        for _ in range(n):
            s["plant"].SetPositions(s["plant_context"], s["iiwa"], q)
            qo = qo_port.Eval(s["sg_context"])
            hit = len(qo.ComputePointPairPenetration()) > 0
        dt = (time.perf_counter() - t0) / n
        print(f"    boolean ComputePointPairPenetration, wall at arm: {1e3*dt:6.2f} ms/query (hit={hit})")
        t0 = time.perf_counter()
        for _ in range(n):
            s["plant"].SetPositions(s["plant_context"], s["iiwa"], q)
            s["plant"].EvalBodyPoseInWorld(s["plant_context"], s["plant"].GetBodyByName("iiwa_link_7"))
        print(f"    kinematics only (SetPositions + one body pose): {1e3*(time.perf_counter()-t0)/n:6.3f} ms")

    header("SceneGraphCollisionChecker (used by IrisZo/IrisNp2) on the 120-strip wall")
    rdb = RobotDiagramBuilder(time_step=0.0)
    parser = rdb.parser()
    (iiwa,) = parser.AddModelsFromUrl(IIWA_URL)
    plant = rdb.plant()
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))
    wall, joint, nb = add_wall(plant, 120, 0.01, 1.5, 0.05, 0.0, 0.25, 1.3, 0.0)
    wall_model = wall.model_instance()
    diagram = rdb.Build()
    checker = SceneGraphCollisionChecker(**{"model": diagram,
                                            "robot_model_instances": [iiwa],
                                            "edge_step_size": 0.01})
    q_full = np.zeros(plant.num_positions())
    wi = joint.position_start()
    q_iiwa = np.array([0, 0.3, 0, -1.2, 0, 0.9, 0])
    idx = [plant.GetJointByName(f"iiwa_joint_{i+1}").position_start() for i in range(7)]
    q_full[idx] = q_iiwa
    q_full[wi] = 0.0
    n = 500
    t0 = time.perf_counter()
    for _ in range(n):
        free = checker.CheckConfigCollisionFree(q_full)
    print(f"  CheckConfigCollisionFree: {1e3*(time.perf_counter()-t0)/n:6.3f} ms (free={free})")
    t0 = time.perf_counter()
    for _ in range(n):
        dists = checker.CalcRobotClearance(q_full, 0.5)
    print(f"  CalcRobotClearance(0.5): {1e3*(time.perf_counter()-t0)/n:6.3f} ms, "
          f"{dists.size()} rows, min = {dists.distances().min() if dists.size() else float('nan'):+.4f}")
    print(f"  position ordering: iiwa joints at {idx}, wall at {wi}")


if __name__ == "__main__":
    main()
