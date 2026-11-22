"""M2 and M3: GcsTrajectoryOptimization along a chain of regions.

The GCS configuration is (q_free, y) where y is the wall carrier's
translation.  y's velocity is bounded to exactly the wall speed, so time is
implied by y and the chain of regions encodes the schedule:

    transit_0 -> window_1 -> transit_1 -> ... -> window_n

window_i  = Q_i x [y_i - v delta_i, y_i + v delta_i]
transit_j = R_j x [end of window j, start of window j+1]   (transit_0 starts at y(0))

M2 (degenerate chain, no IRIS): Q_i is the tolerance box q*_i +/- tol / sqrt(n)
and R_j is the whole joint box.  Nothing constrains collisions in transit;
the dense re-check is the arbiter.

M3 (IRIS + GCS): Q_i = IRIS region seeded at q*_i against the window's
swept wall (the thin slab swept over the window is one slab of thickness
t + 2 v delta_i, i.e. 2 t + d_i, centred on the arm), intersected with the
tolerance box; R_j = IRIS region seeded at the midpoint of the adjacent
configurations with the wall absent, required to contain both endpoints
(containment_points).  Region build time and volume estimates are recorded
separately from the GCS solve.  Variants: IrisZo (M3), IrisNp2 (M3np2),
legacy IrisNp (M3np), all with the eps-scaled configuration-space margin
(window_margin); M3dm is IrisZo with Drake's default 1e-2 margin, which
rejects the seed below eps of about 1 cm.

The path starts at (q0, y(0)) and ends at (q*_n, y_n + v delta_n), the close
of the last window; q(t) holds q*_n afterwards.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from pydrake.all import (
    GcsTrajectoryOptimization,
    GraphOfConvexSetsOptions,
    HPolyhedron,
    Hyperellipsoid,
    IrisNp,
    IrisNp2,
    IrisNp2Options,
    IrisOptions,
    IrisZo,
    IrisZoOptions,
    Parallelism,
    Point,
    RandomGenerator,
    SceneGraphCollisionChecker,
    SolutionResult,
)

from . import params
from .scene import Scene, make_scene
from .sequences import Sequence
from .walls import make_wall

GCS_ORDER = 3
H_MIN = 1e-3
MAX_ROUNDED_PATHS = 5
IRIS_THREADS = 2
IRIS_SEED_RADIUS = 1e-3
DEFAULT_MARGIN = 1e-2       # Drake's configuration_space_margin default (IrisZo, IrisNp2, IrisNp)
ENDPOINT_TOL = 1e-9         # membership tolerance for the source / target points
VOLUME_SAMPLES = 20000
VOLUME_REL_ACCURACY = 0.05


@dataclass
class GcsResult:
    status: str
    traj: object | None                 # CompositeTrajectory in (q_free, y), or None
    solve_time: float
    solver_name: str | None = None
    solution_result: str | None = None
    region_build_time: float | None = None
    region_volume: dict | None = None
    extras: dict = field(default_factory=dict)


# --- chain geometry ---------------------------------------------------------

def y_schedule(seq: Sequence) -> dict:
    """y-intervals of every window and transit, in chain order."""
    y = seq.wall_y_of_t()
    windows = [(y(c.t - c.delta), y(c.t + c.delta)) for c in seq.crossings]
    transits = [(y(0.0), windows[0][0])]
    for a, b in zip(windows[:-1], windows[1:]):
        transits.append((a[1], b[0]))
    return {"windows": windows, "transits": transits, "y0": y(0.0), "y_end": windows[-1][1]}


def _with_y(q_set, y_lo: float, y_hi: float) -> HPolyhedron:
    return q_set.CartesianProduct(HPolyhedron.MakeBox([y_lo], [y_hi]))


def tolerance_box(seq: Sequence, scene: Scene, i: int) -> HPolyhedron:
    c = seq.crossings[i]
    q = scene.q_free(c.q_star)
    half = c.tol / np.sqrt(scene.num_free)
    return HPolyhedron.MakeBox(q - half, q + half)


def joint_box(scene: Scene) -> HPolyhedron:
    lo, hi = scene.free_limits()
    return HPolyhedron.MakeBox(lo, hi)


def chain_points(seq: Sequence, scene: Scene) -> list[np.ndarray]:
    """q0, q*_1, ..., q*_n in free coordinates."""
    return [scene.q_free(seq.q0)] + [scene.q_free(c.q_star) for c in seq.crossings]


# --- GCS solve --------------------------------------------------------------

def solve_chain(seq: Sequence, scene: Scene, q_windows: list, q_transits: list, *,
                rounding_seed: int = 0, order: int = GCS_ORDER,
                max_rounded_paths: int = MAX_ROUNDED_PATHS) -> GcsResult:
    n = scene.num_free
    sched = y_schedule(seq)
    regions, kinds = [], []
    for i in range(seq.n_walls):
        regions.append(_with_y(q_transits[i], *sched["transits"][i])); kinds.append(f"transit_{i}")
        regions.append(_with_y(q_windows[i], *sched["windows"][i])); kinds.append(f"window_{i + 1}")
    edges = [(k, k + 1) for k in range(len(regions) - 1)]

    gcs = GcsTrajectoryOptimization(n + 1)
    main = gcs.AddRegions(regions, edges, order, H_MIN, seq.T, "chain")
    q0 = np.append(scene.q_free(seq.q0), sched["y0"])
    q_end = np.append(scene.q_free(seq.crossings[-1].q_star), sched["y_end"])
    # Order-0 point regions carry no velocity bound, so their duration must be
    # pinned to zero or the solver parks there while the real wall moves on.
    src = gcs.AddRegions([Point(q0)], 0, 0.0, 0.0, "source")
    tgt = gcs.AddRegions([Point(q_end)], 0, 0.0, 0.0, "target")
    # The endpoints lie on the y boundary of their regions (y0 is the start of
    # transit_0, y_end the close of the last window).  AddEdges' intersection
    # test can miss such a boundary point on an IRIS-intersected polytope
    # (observed in the M4 smoke run: "Source vertex ... has no outgoing edges"
    # and kInfeasibleConstraints), so membership is checked here and the two
    # edges are added explicitly.
    src_in = bool(regions[0].PointInSet(q0, ENDPOINT_TOL))
    tgt_in = bool(regions[-1].PointInSet(q_end, ENDPOINT_TOL))
    if not (src_in and tgt_in):
        return GcsResult(status="infeasible", traj=None, solve_time=0.0, solver_name=None,
                         solution_result="endpoint_not_in_region",
                         extras={"n_regions": len(regions), "kinds": kinds, "order": order, "cost": None,
                                 "source_in_region": src_in, "target_in_region": tgt_in})
    gcs.AddEdges(src, main, edges_between_regions=[(0, 0)])
    gcs.AddEdges(main, tgt, edges_between_regions=[(len(regions) - 1, 0)])
    v = scene.velocity_limits()
    gcs.AddVelocityBounds(np.append(-v, seq.speed), np.append(v, seq.speed))
    # No derivative continuity: AddPathContinuityConstraints(1) is on r'(s), and
    # with the wall row's exact speed it would force equal durations in every
    # region; AddContinuityConstraints(1) (true q-dot) is nonconvex.  The path
    # is C0 with bounded velocity; peak velocity is recorded by the re-check.
    gcs.AddPathLengthCost(1.0)

    opt = GraphOfConvexSetsOptions()
    opt.convex_relaxation = True
    opt.preprocessing = True
    opt.max_rounded_paths = max_rounded_paths
    opt.rounding_seed = int(rounding_seed)
    t0 = time.perf_counter()
    traj, result = gcs.SolvePath(src, tgt, opt)
    dt = time.perf_counter() - t0
    sr = result.get_solution_result()
    ok = result.is_success()
    status = "success" if ok else ("iteration_limit" if sr == SolutionResult.kIterationLimit else "infeasible")
    return GcsResult(status=status, traj=traj if ok else None, solve_time=dt,
                     solver_name=result.get_solver_id().name(), solution_result=str(sr),
                     extras={"n_regions": len(regions), "kinds": kinds, "order": order,
                             "cost": float(result.get_optimal_cost()) if ok else None})


def gcs_q_of_t(traj, scene: Scene, seq: Sequence):
    """Full 7-vector q(t): the GCS trajectory on [0, t_end], holding q*_n after."""
    n = scene.num_free
    t_end = traj.end_time()
    q_last = scene.q_full(traj.value(t_end).ravel()[:n])

    def q_of_t(t: float) -> np.ndarray:
        if t >= t_end:
            return q_last
        return scene.q_full(traj.value(max(0.0, float(t))).ravel()[:n])
    return q_of_t


def gcs_wall_error(traj, seq: Sequence, scene: Scene, n_samples: int = 50) -> float:
    """max |y_traj(t) - wall_y(t)|: should be ~solver tolerance."""
    y = seq.wall_y_of_t()
    n = scene.num_free
    ts = np.linspace(0.0, traj.end_time(), n_samples)
    return float(max(abs(traj.value(t).ravel()[n] - y(t)) for t in ts))


# --- M2 ---------------------------------------------------------------------

def plan_m2(seq: Sequence, scene: Scene, seed: int = 0, **kw) -> GcsResult:
    q_windows = [tolerance_box(seq, scene, i) for i in range(seq.n_walls)]
    q_transits = [joint_box(scene) for _ in range(seq.n_walls)]
    return solve_chain(seq, scene, q_windows, q_transits, rounding_seed=seed, **kw)


# --- M3: IRIS regions -------------------------------------------------------

def window_checker_scene(seq: Sequence, i: int) -> Scene:
    """Arm + the swept wall of window i (thickness 2 t + d_i), welded, centred."""
    c = seq.crossings[i]
    w = seq.walls[i]
    thick = w.thickness + 2.0 * seq.speed * c.delta
    swept = make_wall(c.q_star, seq.eps, strip=w.strip, thickness=thick, width=w.width,
                      z0=w.z0, height=w.height)
    return make_scene([swept], beta=seq.beta, fixed_wall_y=0.0, robot_diagram=True)


def transit_checker_scene(seq: Sequence) -> Scene:
    return make_scene([], beta=seq.beta, fixed_wall_y=0.0, robot_diagram=True)


def make_checker(scene: Scene, edge_step: float = 0.01) -> SceneGraphCollisionChecker:
    assert scene.fixed_wall_y is not None, "IRIS needs the welded-wall scene"
    return SceneGraphCollisionChecker(model=scene.diagram, robot_model_instances=[scene.iiwa],
                                      edge_step_size=edge_step)


def _common_options(opts, seed: int, threads: int, containment: np.ndarray | None, margin: float):
    so = opts.sampled_iris_options
    so.random_seed = int(seed)
    so.parallelism = Parallelism(threads)
    so.configuration_space_margin = float(margin)
    if containment is not None:
        so.containment_points = np.asarray(containment, dtype=float).T      # (dim, n_points)
    return opts


class IrisSeedError(RuntimeError):
    """IRIS refused the seed: the configuration is within configuration_space_margin
    of collision (Drake's own check), so no region with an interior exists there."""


def grow_region(checker, scene: Scene, seed_q: np.ndarray, *, variant: str = "zo", seed: int = 0,
                threads: int = IRIS_THREADS, containment: np.ndarray | None = None,
                domain: HPolyhedron | None = None, margin: float = DEFAULT_MARGIN) -> tuple[HPolyhedron, float]:
    """One IRIS region in the free-joint space, and its build time.  Raises
    IrisSeedError when Drake rejects the seed (its message names the margin)."""
    domain = domain or joint_box(scene)
    ell = Hyperellipsoid.MakeHypersphere(IRIS_SEED_RADIUS, np.asarray(seed_q, dtype=float))
    t0 = time.perf_counter()
    try:
        if variant == "zo":
            region = IrisZo(checker, ell, domain, _common_options(IrisZoOptions(), seed, threads, containment, margin))
        elif variant == "np2":
            region = IrisNp2(checker, ell, domain, _common_options(IrisNp2Options(), seed, threads, containment, margin))
        elif variant == "np":
            o = IrisOptions()
            o.random_seed = int(seed)
            o.configuration_space_margin = float(margin)
            scene.plant.SetPositions(scene.plant_context, scene.iiwa, np.asarray(seed_q, dtype=float))
            region = IrisNp(scene.plant, scene.plant_context, o)
        else:
            raise ValueError(variant)
    except RuntimeError as e:
        if "configuration_space_margin" in str(e):
            raise IrisSeedError(str(e)) from e
        raise
    return region, time.perf_counter() - t0


def window_margin(seq: Sequence, i: int, margin: str | float) -> float:
    """configuration_space_margin for window i.  "default" is Drake's 1e-2 rad,
    which exceeds the free radius eps / rho around q* once eps < rho * 1e-2
    (about 11 mm at beta = 0, 16 mm at beta = 0.1) and makes IRIS reject the
    seed.  "scaled" uses min(default, tol / 2) = min(1e-2, eps / (4 rho)): the
    motion bound puts q* at least eps / rho = 2 tol from collision, so a margin
    of tol / 2 always leaves an interior, and the region can still contain the
    tolerance box."""
    if margin == "default":
        return DEFAULT_MARGIN
    if margin == "scaled":
        return min(DEFAULT_MARGIN, 0.5 * seq.crossings[i].tol)
    return float(margin)


def region_volume(region: HPolyhedron, seed: int = 0) -> dict:
    """Two estimates: the maximum-volume inscribed ellipsoid (exact, cheap, a
    lower bound) and Monte Carlo sampling in the bounding box (reports the
    accuracy it reached: in 7-D a thin region may exhaust the sample budget)."""
    out = {"dim": region.ambient_dimension()}
    try:
        out["ellipsoid"] = float(region.MaximumVolumeInscribedEllipsoid().CalcVolume())
    except Exception as e:
        out["ellipsoid"] = None
        out["ellipsoid_error"] = str(e)[:100]
    try:
        sv = region.CalcVolumeViaSampling(RandomGenerator(seed), VOLUME_REL_ACCURACY, VOLUME_SAMPLES)
        out.update({"sampled": float(sv.volume), "sampled_rel_accuracy": float(sv.rel_accuracy),
                    "sampled_n": int(sv.num_samples)})
    except Exception as e:
        out["sampled"] = None
        out["sampled_error"] = str(e)[:100]
    return out


def plan_m3(seq: Sequence, scene: Scene, seed: int = 0, *, variant: str = "zo",
            threads: int = IRIS_THREADS, margin: str | float = "scaled", **kw) -> GcsResult:
    """IRIS regions (windows against the swept wall, transits with the wall
    absent) then the GCS chain.  A seed rejected by IRIS is a planner failure
    (status infeasible, solution_result "iris_seed_failed"), with the build time
    spent so far and the margins recorded."""
    pts = chain_points(seq, scene)
    build = 0.0
    volumes = {"windows": [], "transits": [], "window_build": [], "transit_build": [], "variant": variant,
               "window_margin": [window_margin(seq, i, margin) for i in range(seq.n_walls)],
               "transit_margin": DEFAULT_MARGIN, "margin_mode": margin if isinstance(margin, str) else "fixed"}

    def failed(e: Exception, where: str) -> GcsResult:
        return GcsResult(status="infeasible", traj=None, solve_time=0.0, solution_result="iris_seed_failed",
                         region_build_time=build, region_volume=volumes,
                         extras={"error": f"{where}: {str(e)[:160]}", "cost": None})

    # transit regions: wall absent, must contain both endpoints
    t_scene = transit_checker_scene(seq)
    t_checker = make_checker(t_scene)
    q_transits = []
    for j in range(seq.n_walls):
        a, b = pts[j], pts[j + 1]
        t0 = time.perf_counter()
        try:
            region, dt = grow_region(t_checker, t_scene, (a + b) / 2, variant=variant, seed=seed + 100 * j,
                                     threads=threads, containment=np.array([a, b]) if variant != "np" else None,
                                     margin=DEFAULT_MARGIN)
        except IrisSeedError as e:
            build += time.perf_counter() - t0
            return failed(e, f"transit_{j}")
        build += dt
        q_transits.append(region)
        vol = region_volume(region, seed)
        vol["contains_endpoints"] = bool(region.PointInSet(a, ENDPOINT_TOL) and region.PointInSet(b, ENDPOINT_TOL))
        volumes["transits"].append(vol)
        volumes["transit_build"].append(dt)
    # window regions: swept wall present, seeded at q*
    q_windows = []
    for i in range(seq.n_walls):
        w_scene = window_checker_scene(seq, i)
        w_checker = make_checker(w_scene)
        t0 = time.perf_counter()
        try:
            region, dt = grow_region(w_checker, w_scene, pts[i + 1], variant=variant, seed=seed + 100 * i + 7,
                                     threads=threads, margin=volumes["window_margin"][i])
        except IrisSeedError as e:
            build += time.perf_counter() - t0
            return failed(e, f"window_{i + 1}")
        build += dt
        vol = region_volume(region, seed)
        vol["contains_q_star"] = bool(region.PointInSet(pts[i + 1]))
        box = tolerance_box(seq, scene, i)
        vol["contains_tolerance_box"] = bool(box.ContainedIn(region))
        volumes["windows"].append(vol)
        volumes["window_build"].append(dt)
        q_windows.append(region.Intersection(box))
    res = solve_chain(seq, scene, q_windows, q_transits, rounding_seed=seed, **kw)
    res.region_build_time = build
    res.region_volume = volumes
    return res
