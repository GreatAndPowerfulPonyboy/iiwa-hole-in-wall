"""GcsTrajectoryOptimization along a chain of regions: gcs-box and iris-gcs.

The GCS configuration is (q_free, y) where y is the wall carrier's
translation.  y's velocity is bounded to exactly the wall speed, so time is
implied by y and the chain of regions encodes the schedule:

    transit_0 -> window_1 -> transit_1 -> ... -> window_n

window_i  = Q_i x [y_i - v delta_i, y_i + v delta_i]
transit_j = R_j x [end of window j, start of window j+1]   (transit_0 starts at y(0))

gcs-box (no IRIS): Q_i is the tolerance box q*_i +/- tol / sqrt(n) and R_j
is the whole joint box.  Nothing constrains collisions in transit;
the dense re-check is the arbiter.

iris-gcs: Q_i = IRIS region seeded at q*_i against the window's
swept wall (the thin slab swept over the window is one slab of thickness
t + 2 v delta_i, i.e. 2 t + d_i, centred on the arm), intersected with the
tolerance box; R_j = IRIS region seeded at the midpoint of the adjacent
configurations with the wall absent, required to contain both endpoints
(containment_points).  Region build time and volume estimates are recorded
separately from the GCS solve.  The IRIS algorithm is a config option
(IrisZo, IrisNp2 or the legacy IrisNp), and so is the window regions'
configuration-space margin: eps-scaled by default (window_margin), or
Drake's default 1e-2, which rejects the seed below eps of about 1 cm.

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
    SolutionResult,
)

from hitw import params
from hitw.api import Budget, PlanOutcome, Services, planner
from hitw.scene import Scene
from hitw.sequences import Sequence

GCS_ORDER = 3
H_MIN = 1e-3
MAX_ROUNDED_PATHS = 5
IRIS_THREADS = 2
IRIS_SEED_RADIUS = 1e-3
DEFAULT_MARGIN = 1e-2       # Drake's configuration_space_margin default (IrisZo, IrisNp2, IrisNp)
ENDPOINT_TOL = 1e-9         # membership tolerance for the source / target points
VOLUME_SAMPLES = 20000
VOLUME_REL_ACCURACY = 0.05


@dataclass(frozen=True)
class GcsBoxConfig:
    """Options of the GCS chain over tolerance boxes and the joint box.

    Parameters
    ----------
    order : int
        Bezier order of each region's trajectory piece.
    max_rounded_paths : int
        GraphOfConvexSetsOptions.max_rounded_paths.
    """

    order: int = GCS_ORDER
    max_rounded_paths: int = MAX_ROUNDED_PATHS


IRIS_VARIANTS = ("zo", "np2", "np")
MARGIN_MODES = ("scaled", "default")


@dataclass(frozen=True)
class IrisGcsConfig:
    """Options of IRIS + GCS.

    Parameters
    ----------
    iris : {"zo", "np2", "np"}
        IRIS algorithm: IrisZo, IrisNp2 or the legacy IrisNp.
    margin : {"scaled", "default"} or float
        configuration_space_margin of the window regions in radians:
        "scaled" is min(1e-2, tol / 2), "default" is Drake's 1e-2, a float
        is used as given (see `window_margin`).
    threads : int
        Parallelism of IrisZo and IrisNp2.
    order : int
        Bezier order of each region's trajectory piece.
    max_rounded_paths : int
        GraphOfConvexSetsOptions.max_rounded_paths.
    """

    iris: str = "zo"
    margin: str | float = "scaled"
    threads: int = IRIS_THREADS
    order: int = GCS_ORDER
    max_rounded_paths: int = MAX_ROUNDED_PATHS

    def __post_init__(self):
        if self.iris not in IRIS_VARIANTS:
            raise ValueError(f"iris must be one of {IRIS_VARIANTS}, got {self.iris!r}")
        if isinstance(self.margin, str):
            if self.margin not in MARGIN_MODES:
                raise ValueError(f"margin must be one of {MARGIN_MODES} or a positive float, got {self.margin!r}")
        elif not float(self.margin) > 0:
            raise ValueError(f"margin must be positive, got {self.margin!r}")


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
    # test can miss such a boundary point on an IRIS-intersected polytope,
    # leaving the source with no outgoing edge ("Source vertex ... has no
    # outgoing edges", then kInfeasibleConstraints), so membership is checked
    # here and the two edges are added explicitly.
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
    """Return ``q_of_t(t) -> ndarray, shape (7,)`` from a GCS (q_free, y) trajectory, held after it ends."""
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


# --- tolerance-box chain ----------------------------------------------------

def solve_gcs_box(seq: Sequence, scene: Scene, seed: int = 0, cfg: GcsBoxConfig | None = None) -> GcsResult:
    """Plan through a GCS chain of tolerance-box windows and whole-joint-box transits.

    Parameters
    ----------
    seq : Sequence
        The crossings, walls and timing.
    scene : Scene
        The prismatic-wall scene of `seq`.
    seed : int
        GCS rounding seed.
    cfg : GcsBoxConfig, optional
        Planner options; None means the defaults.

    Returns
    -------
    GcsResult
        Status and the (q_free, y) trajectory, or None.
    """
    cfg = cfg or GcsBoxConfig()
    q_windows = [tolerance_box(seq, scene, i) for i in range(seq.n_walls)]
    q_transits = [joint_box(scene) for _ in range(seq.n_walls)]
    return solve_chain(seq, scene, q_windows, q_transits, rounding_seed=seed, order=cfg.order,
                       max_rounded_paths=cfg.max_rounded_paths)


# --- IRIS regions -----------------------------------------------------------

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


def solve_iris_gcs(seq: Sequence, svc: Services, cfg: IrisGcsConfig | None = None,
                   deadline: float | None = None) -> GcsResult:
    """Plan through a GCS chain of IRIS regions.

    Window regions are grown against the swept wall, transit regions with
    the wall absent.

    Parameters
    ----------
    seq : Sequence
        The crossings, walls and timing.
    svc : Services
        Trial services: the scenes and checkers, and ``drake_seed``, the
        base of the IRIS random seeds and the GCS rounding seed.
    cfg : IrisGcsConfig, optional
        Planner options; None means the defaults.
    deadline : float, optional
        ``time.perf_counter()`` value checked after every region; once it
        has passed no further region is built (status "timeout",
        solution_result "budget_expired").  None means no limit.

    Returns
    -------
    GcsResult
        Status, the (q_free, y) trajectory (or None), region build time and
        volumes.  A seed rejected by IRIS is a planner failure (status
        infeasible, solution_result "iris_seed_failed"), with the build time
        spent so far and the margins recorded.
    """
    cfg = cfg or IrisGcsConfig()
    scene, seed = svc.scene, svc.drake_seed
    variant, threads, margin = cfg.iris, cfg.threads, cfg.margin
    pts = chain_points(seq, scene)
    build = 0.0
    volumes = {"windows": [], "transits": [], "window_build": [], "transit_build": [], "variant": variant,
               "window_margin": [window_margin(seq, i, margin) for i in range(seq.n_walls)],
               "transit_margin": DEFAULT_MARGIN, "margin_mode": margin if isinstance(margin, str) else "fixed"}

    def failed(e: Exception, where: str) -> GcsResult:
        return GcsResult(status="infeasible", traj=None, solve_time=0.0, solution_result="iris_seed_failed",
                         region_build_time=build, region_volume=volumes,
                         extras={"error": f"{where}: {str(e)[:160]}", "cost": None})

    def expired(where: str) -> GcsResult | None:
        if deadline is None or time.perf_counter() < deadline:
            return None
        return GcsResult(status="timeout", traj=None, solve_time=0.0, solution_result="budget_expired",
                         region_build_time=build, region_volume=volumes,
                         extras={"error": f"budget expired after {where}", "cost": None})

    # transit regions: wall absent, must contain both endpoints
    t_scene = svc.wall_absent_scene()
    t_checker = svc.checker(t_scene)
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
        if (stop := expired(f"transit_{j}")) is not None:
            return stop
    # window regions: swept wall present, seeded at q*
    q_windows = []
    for i in range(seq.n_walls):
        w_scene = svc.swept_window_scene(i)
        w_checker = svc.checker(w_scene)
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
        if (stop := expired(f"window_{i + 1}")) is not None:
            return stop
    res = solve_chain(seq, scene, q_windows, q_transits, rounding_seed=seed, order=cfg.order,
                      max_rounded_paths=cfg.max_rounded_paths)
    res.region_build_time = build
    res.region_volume = volumes
    return res


def _deadline(budget: Budget) -> float | None:
    return None if budget.wall_clock is None else budget.started + budget.wall_clock


def _outcome(res: GcsResult, seq: Sequence, scene: Scene) -> PlanOutcome:
    extras = {"solver": res.solver_name, "solution_result": res.solution_result, **res.extras}
    if res.traj is not None:
        extras["wall_error"] = gcs_wall_error(res.traj, seq, scene)
    return PlanOutcome(status=res.status,
                       q_of_t=gcs_q_of_t(res.traj, scene, seq) if res.traj is not None else None,
                       solve_time=res.solve_time, region_build_time=res.region_build_time,
                       region_volume=res.region_volume, extras=extras)


@planner("gcs-box", config=GcsBoxConfig, display_name="GCS (boxes)")
def plan_gcs_box(seq: Sequence, svc: Services, budget: Budget, rng: np.random.Generator,
                 cfg: GcsBoxConfig) -> PlanOutcome:
    """Plan through a GCS chain of tolerance-box windows and whole-joint-box transits.

    Parameters
    ----------
    seq : Sequence
        The crossings, walls and timing.
    svc : Services
        Trial services; ``svc.drake_seed`` is the GCS rounding seed.
    budget : Budget
        Wall-clock budget, not checked: the convex solve takes milliseconds
        and is bounded by the sweep's kill timer.
    rng : numpy.random.Generator
        The planner's random stream (unused).
    cfg : GcsBoxConfig
        Planner options.

    Returns
    -------
    PlanOutcome
        Status, q(t) from the GCS trajectory, and solve time.
    """
    return _outcome(solve_gcs_box(seq, svc.scene, svc.drake_seed, cfg), seq, svc.scene)


@planner("iris-gcs", config=IrisGcsConfig, display_name="IRIS + GCS")
def plan_iris_gcs(seq: Sequence, svc: Services, budget: Budget, rng: np.random.Generator,
                  cfg: IrisGcsConfig) -> PlanOutcome:
    """Plan through a GCS chain of IRIS regions.

    Parameters
    ----------
    seq : Sequence
        The crossings, walls and timing.
    svc : Services
        Trial services; ``svc.drake_seed`` seeds IRIS (``drake_seed + 100 j``
        for transit j, ``drake_seed + 100 i + 7`` for window i), the GCS
        rounding and the region-volume sampler.
    budget : Budget
        Wall-clock budget, checked after every IRIS region; the GCS solve
        itself is not interrupted.
    rng : numpy.random.Generator
        The planner's random stream (unused).
    cfg : IrisGcsConfig
        Planner options.

    Returns
    -------
    PlanOutcome
        Status, q(t), solve time, region build time and region volumes.
    """
    return _outcome(solve_iris_gcs(seq, svc, cfg, deadline=_deadline(budget)), seq, svc.scene)
