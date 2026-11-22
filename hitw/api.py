"""The contract between planners and the harness.

A planner is a function registered with `planner`.  The harness looks it up
by id (or by ``"path/to/file.py:function"`` for a planner outside the
package), resolves its typed config, calls it, and grades what it returns
with the shared dense re-check; what the planner reports about itself is
recorded but never trusted.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import importlib.util
import inspect
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np

from .config import config_fields, resolve_config

if TYPE_CHECKING:
    from .scene import Scene
    from .sequences import Sequence

TASKS = ("connect", "find")
_REPO_ROOT = Path(__file__).resolve().parents[1]
PLANNERS_DIR = _REPO_ROOT / "planners"


@dataclass
class PlanOutcome:
    """What a connect planner returns.

    Parameters
    ----------
    status : str
        Planner-reported status, one of ``evaluate.STATUSES``.  Recorded as
        ``solver_status``; success is decided by the dense re-check.
    q_of_t : callable or None
        ``q_of_t(t) -> ndarray, shape (7,)``: the full joint vector in radians
        at time t in seconds, for every t in [0, seq.T].  None when the
        planner has no trajectory.
    solve_time : float
        Seconds the planner spent solving.
    n_evals : int or None
        Solver function evaluations, if counted.
    iterations : int or None
        Solver iterations or samples, if counted.
    region_build_time : float or None
        Seconds spent building regions before the solve, if any.
    region_volume : dict or None
        Region size estimates, if any (see ``planners.gcs.region_volume``).
    solver_info : list
        Raw solver exit codes, one per attempt.
    extras : dict
        Anything else worth recording; must be JSON-serializable after numpy
        arrays are converted to lists.
    """

    status: str
    q_of_t: Callable[[float], np.ndarray] | None
    solve_time: float
    n_evals: int | None = None
    iterations: int | None = None
    region_build_time: float | None = None
    region_volume: dict | None = None
    solver_info: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)


@dataclass
class Budget:
    """A planner's wall-clock budget.

    Parameters
    ----------
    wall_clock : float or None
        Seconds the planner may use; None means unlimited.
    started : float
        ``time.perf_counter()`` when the clock started; the harness starts
        it immediately before calling the planner.

    Examples
    --------
    >>> b = Budget(wall_clock=2.0)
    >>> b.remaining() <= 2.0, b.expired()
    (True, False)
    >>> Budget(None).remaining()
    inf
    """

    wall_clock: float | None = None
    started: float = field(default_factory=time.perf_counter)

    def elapsed(self) -> float:
        """Return the seconds since the clock started."""
        return time.perf_counter() - self.started

    def remaining(self) -> float:
        """Return the seconds left, never negative; ``inf`` when unlimited."""
        if self.wall_clock is None:
            return float("inf")
        return max(0.0, self.wall_clock - self.elapsed())

    def expired(self) -> bool:
        """Return True once the budget is used up (never when unlimited)."""
        return self.wall_clock is not None and self.elapsed() >= self.wall_clock


class Services:
    """Per-trial resources the harness hands to a planner.

    Scenes and collision checkers are built on first use and cached for the
    trial.  Planners may build private models of their own, but grading is
    always on `scene`.

    Parameters
    ----------
    seq : Sequence
        The trial's problem.
    scene : Scene, optional
        Its prismatic-wall scene; None builds it.
    drake_seed : int
        See the attribute.

    Attributes
    ----------
    seq : Sequence
        The problem.
    scene : Scene
        The shared prismatic-wall scene, the one the dense re-check grades on.
    drake_seed : int
        Integer seed for Drake's internal random generators (IRIS
        random_seed, GCS rounding_seed).  Equal to the trial seed so existing
        results are unchanged.  Do not use it to seed NumPy; use the rng
        argument.

    Examples
    --------
    >>> from hitw.harness import trial_sequence
    >>> seq = trial_sequence(0, 0.0, 0.02, 1)
    >>> svc = Services(seq, drake_seed=0)
    >>> svc.in_collision(seq.crossings[0].q_star, seq.crossings[0].t)
    False
    >>> svc.wall_absent_scene() is svc.wall_absent_scene()
    True
    """

    def __init__(self, seq: "Sequence", scene: "Scene | None" = None, drake_seed: int = 0):
        from .scene import make_scene
        self.seq = seq
        self.scene = scene if scene is not None else make_scene(seq.walls, beta=seq.beta)
        self.drake_seed = int(drake_seed)
        self._walls, self._beta = seq.walls, seq.beta
        self._wall_y = seq.wall_y_of_t()
        self._scenes: dict = {}
        self._checkers: dict = {}

    @classmethod
    def for_wall(cls, view: "WallView", drake_seed: int = 0) -> "Services":
        """Build the services of a find trial from what the finder may know about the wall.

        Parameters
        ----------
        view : WallView
            The wall.
        drake_seed : int
            See `drake_seed`.

        Returns
        -------
        Services
            ``seq`` is None and `swept_window_scene` is unavailable; the
            scenes carry the wall's boxes only (the WallSpec they hold has
            no crossing configuration, silhouette, opening or arm depth).
        """
        from .scene import make_scene
        svc = cls.__new__(cls)
        svc.seq = None
        svc._walls, svc._beta = [view.wall_spec()], view.beta
        svc.scene = make_scene(svc._walls, beta=view.beta)
        svc.drake_seed = int(drake_seed)
        svc._wall_y = view.wall_y_of_t()
        svc._scenes, svc._checkers = {}, {}
        return svc

    def _cached(self, key, build):
        if key not in self._scenes:
            self._scenes[key] = build()
        return self._scenes[key]

    def welded_scene(self, wall_y: float = 0.0) -> "Scene":
        """Return the trial's walls welded at carrier translation `wall_y` (meters).

        Parameters
        ----------
        wall_y : float
            Carrier translation; 0 centres the first slab on the arm plane.

        Returns
        -------
        Scene
            A RobotDiagram scene (it can back `checker`) whose plant has the
            arm joints only.  Drake skips pairs of two world-anchored bodies,
            so the link_0-vs-wall pair is not checked in it.
        """
        from .scene import make_scene
        return self._cached(("welded", float(wall_y)), lambda: make_scene(
            self._walls, beta=self._beta, fixed_wall_y=float(wall_y), robot_diagram=True))

    def swept_window_scene(self, i: int) -> "Scene":
        """Return the arm with the wall of window i swept over the window, welded and centred.

        Parameters
        ----------
        i : int
            Crossing index (0-based).

        Returns
        -------
        Scene
            One slab of thickness t + 2 v delta_i (= 2 t + d_i) cut for
            q*_i: everything the moving wall occupies during the window.

        Raises
        ------
        ValueError
            In a find trial, where the crossing configuration is unknown.
        """
        from .scene import make_scene
        from .walls import make_wall

        if self.seq is None:
            raise ValueError("swept_window_scene needs the crossings, which a find trial does not have")

        def build():
            c, w = self.seq.crossings[i], self.seq.walls[i]
            swept = make_wall(c.q_star, self.seq.eps, strip=w.strip,
                              thickness=w.thickness + 2.0 * self.seq.speed * c.delta,
                              width=w.width, z0=w.z0, height=w.height)
            return make_scene([swept], beta=self.seq.beta, fixed_wall_y=0.0, robot_diagram=True)
        return self._cached(("swept", int(i)), build)

    def wall_absent_scene(self) -> "Scene":
        """Return the arm alone (RobotDiagram, so it can back `checker`)."""
        from .scene import make_scene
        return self._cached(("absent",), lambda: make_scene([], beta=self._beta, fixed_wall_y=0.0,
                                                           robot_diagram=True))

    def checker(self, scene: "Scene", edge_step: float = 0.01):
        """Return a SceneGraphCollisionChecker for a welded scene of this trial.

        Parameters
        ----------
        scene : Scene
            A scene from `welded_scene`, `swept_window_scene` or
            `wall_absent_scene`.
        edge_step : float
            Edge-check step in radians.

        Returns
        -------
        pydrake.planning.SceneGraphCollisionChecker
            Cached per (scene, edge_step).

        Raises
        ------
        ValueError
            If `scene` has a moving wall or is not a RobotDiagram.
        """
        from pydrake.all import SceneGraphCollisionChecker
        if scene.fixed_wall_y is None:
            raise ValueError("a collision checker needs a welded-wall scene")
        key = (id(scene), float(edge_step))
        if key not in self._checkers:
            self._checkers[key] = (scene, SceneGraphCollisionChecker(
                model=scene.diagram, robot_model_instances=[scene.iiwa], edge_step_size=edge_step))
        return self._checkers[key][1]

    def in_collision(self, q_full: np.ndarray, t: float) -> bool:
        """Return True if the arm at `q_full` collides at time `t` (seconds) on the shared scene.

        Parameters
        ----------
        q_full : ndarray, shape (7,)
            Joint vector in radians.
        t : float
            Time in seconds; the wall is at its position at t.

        Returns
        -------
        bool
        """
        from .scene import in_collision
        return in_collision(self.scene, q_full, self._wall_y(t))

    def min_distance(self, q_full: np.ndarray, t: float, cap: float = 0.5) -> float:
        """Return the shared distance function at `q_full` and time `t`.

        Parameters
        ----------
        q_full : ndarray, shape (7,)
            Joint vector in radians.
        t : float
            Time in seconds; the wall is at its position at t.
        cap : float
            Distance cap in meters.

        Returns
        -------
        float
            Smallest signed distance in meters between the arm and anything
            else, capped at `cap` (see `hitw.scene.min_distance`).
        """
        from .scene import min_distance
        return min_distance(self.scene, q_full, self._wall_y(t), cap=cap)


@dataclass(frozen=True)
class WallView:
    """The wall of a find trial, as a finder sees it.

    It holds the material of the slab and when it passes, never the
    configuration the opening was cut for.

    Attributes
    ----------
    boxes : tuple of (cx, cz, w, h)
        Material boxes in the slab's x-z frame, meters; the opening is the
        part of the slab they leave free.
    thickness : float
        Slab thickness along y, meters.
    width : float
        Slab extent in x, meters (x in [-width / 2, width / 2]).
    z0 : float
        Slab bottom, meters.
    height : float
        Slab extent in z, meters.
    t_cross : float
        Time in seconds at which the slab is centred on the arm plane.
    speed : float
        Wall speed along y, m/s.
    beta : float
        Roll band in radians: roll joints must satisfy |q_j| <= beta.
    eps : float
        Clearance the opening was cut with, meters.
    """

    boxes: tuple
    thickness: float
    width: float
    z0: float
    height: float
    t_cross: float
    speed: float
    beta: float
    eps: float

    @classmethod
    def of(cls, wall, beta: float, eps: float, speed: float) -> "WallView":
        """Return the view of a WallSpec (dropping its crossing configuration, silhouette and opening)."""
        return cls(boxes=tuple(tuple(float(v) for v in b) for b in wall.boxes), thickness=float(wall.thickness),
                   width=float(wall.width), z0=float(wall.z0), height=float(wall.height),
                   t_cross=float(wall.t_cross), speed=float(speed), beta=float(beta), eps=float(eps))

    def wall_y_of_t(self):
        """Return the carrier translation in meters as a function of time in seconds (0 at t_cross)."""
        return lambda t: self.speed * (t - self.t_cross)

    def wall_spec(self):
        """Return a WallSpec with these boxes and no crossing configuration, silhouette, opening or depth."""
        from .walls import WallSpec
        return WallSpec(q_star=np.full(7, np.nan), eps=self.eps, strip=float("nan"), thickness=self.thickness,
                        width=self.width, z0=self.z0, height=self.height, boxes=[tuple(b) for b in self.boxes],
                        silhouette=None, opening=None, arm_depth=float("nan"), y_offset=0.0, t_cross=self.t_cross)


@dataclass(frozen=True)
class NoConfig:
    """Config of a planner that takes no options."""


@dataclass(frozen=True)
class PlannerSpec:
    """A registered planner.

    Attributes
    ----------
    id : str
        Registered id, used on the command line and in results files.
    task : {"connect", "find"}
        Which problem the planner solves.
    func : callable
        The planner function.
    config : type
        Its config dataclass.
    display_name : str
        Name for plots and tables.
    supports_beta_positive : bool
        Whether the planner handles beta > 0 (seven free joints).
    """

    id: str
    task: str
    func: Callable
    config: type
    display_name: str
    supports_beta_positive: bool

    def make_config(self, overrides: Mapping[str, Any] | Any | None = None):
        """Resolve this planner's config from its defaults and `overrides`.

        Parameters
        ----------
        overrides : mapping or config instance, optional
            Options to change; see `hitw.config.resolve_config`.

        Returns
        -------
        config instance

        Raises
        ------
        ValueError
            If an option is unknown or cannot be converted.
        """
        try:
            return resolve_config(self.config, overrides)
        except ValueError as e:
            raise ValueError(f"{self.id}: {e}") from e

    def config_dict(self, cfg) -> dict:
        """Return a resolved config as a plain dict."""
        return dataclasses.asdict(cfg)

    def config_label(self, cfg) -> str:
        """Label a run by the planner id and the options that differ from their defaults.

        Parameters
        ----------
        cfg : config instance
            A resolved config of this planner.

        Returns
        -------
        str
            ``id`` when every option is at its default, otherwise
            ``id[key=value,...]`` in field order, e.g.
            ``iris-gcs[iris=np2]``.  `parse_ref` reads it back.
        """
        defaults = config_fields(self.config)
        changed = [(k, v) for k, v in dataclasses.asdict(cfg).items() if v != defaults[k]]
        if not changed:
            return self.id
        return f"{self.id}[{','.join(f'{k}={_format_value(v)}' for k, v in changed)}]"

    def source(self) -> dict:
        """Identify the code of this planner.

        Returns
        -------
        dict
            ``module`` (import name), ``path`` (relative to the repository
            when inside it, absolute otherwise) and ``sha256`` of the file
            defining the planner, read now.
        """
        path = Path(inspect.getsourcefile(self.func)).resolve()
        shown = path.relative_to(_REPO_ROOT) if path.is_relative_to(_REPO_ROOT) else path
        return {"module": self.func.__module__, "path": str(shown),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


_REGISTRY: dict[str, PlannerSpec] = {}


def _format_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "none"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def parse_ref(ref: str) -> tuple[str, dict[str, str]]:
    """Split a planner reference into the planner and its config overrides.

    Parameters
    ----------
    ref : str
        ``planner`` or ``planner[key=value,...]``, where ``planner`` is an id
        or ``"path/to/file.py:function"``; a config label is such a
        reference.

    Returns
    -------
    planner : str
        The part before the brackets.
    overrides : dict
        Option names to their values as strings (converted to the field
        types when the config is resolved).

    Raises
    ------
    ValueError
        If the bracket part is not a comma-separated list of key=value.

    Examples
    --------
    >>> parse_ref("iris-gcs[iris=np2,margin=default]")
    ('iris-gcs', {'iris': 'np2', 'margin': 'default'})
    """
    ref = ref.strip()
    if not ref.endswith("]") or "[" not in ref:
        return ref, {}
    head, _, body = ref[:-1].partition("[")
    overrides = {}
    for item in filter(None, (x.strip() for x in body.split(","))):
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"cannot read {item!r} in {ref!r}; expected key=value")
        overrides[key.strip()] = value.strip()
    return head.strip(), overrides


def planner(id: str, *, task: str = "connect", config: type | None = None, display_name: str | None = None,
            supports_beta_positive: bool = True):
    """Register a planner function under `id`.

    A connect planner is called as ``plan(seq, svc, budget, rng, cfg)`` and
    returns a `PlanOutcome`.  A find planner is called as
    ``find(wall, svc, budget, rng, cfg)`` with a `WallView` and the
    services of the wall (`Services.for_wall`), and yields candidates: full
    joint vectors, shape (7,), in radians, or ``(q, info)`` with a dict of
    per-attempt facts to record (for example the solver's exit code).  The
    harness validates every candidate, counts attempts, timestamps the
    first valid one and stops asking for more when the budget expires; the
    finder only proposes.

    Parameters
    ----------
    id : str
        Unique id, lowercase-kebab by convention; may not contain ``":"``,
        ``"["`` or ``"]"``.
    task : {"connect", "find"}
        Which problem the planner solves.
    config : type, optional
        Frozen dataclass of the planner's options, every field with a
        default; None means the planner takes no options.
    display_name : str, optional
        Name for plots and tables; defaults to `id`.
    supports_beta_positive : bool
        False if the planner only handles beta = 0.

    Returns
    -------
    callable
        A decorator that registers the function and returns it unchanged.

    Raises
    ------
    ValueError
        If `id` is taken by another function, is empty or contains ``":"``,
        ``"["`` or ``"]"``, `task` is unknown, or `config` is not a
        dataclass.

    Examples
    --------
    >>> from hitw.api import PlanOutcome, planner
    >>> @planner("hold-start", display_name="Hold start")
    ... def hold_start(seq, svc, budget, rng, cfg):
    ...     return PlanOutcome(status="success", q_of_t=lambda t: seq.q0, solve_time=0.0)
    """
    if not id or any(c in id for c in ":[]"):
        raise ValueError(f"planner id {id!r} must be non-empty and contain no ':', '[' or ']'")
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    config = config or NoConfig
    if not dataclasses.is_dataclass(config):
        raise ValueError(f"config of {id!r} must be a dataclass, got {config!r}")
    def register(func: Callable) -> Callable:
        old = _REGISTRY.get(id)
        if old is not None and (old.func.__module__, old.func.__qualname__) != (func.__module__, func.__qualname__):
            raise ValueError(f"planner id {id!r} is already registered by {old.func.__module__}.{old.func.__qualname__}")
        _REGISTRY[id] = PlannerSpec(id=id, task=task, func=func, config=config, display_name=display_name or id,
                                    supports_beta_positive=supports_beta_positive)
        return func
    return register


_LOAD_ERRORS: dict[str, str] = {}


def _module_name(path: Path) -> str:
    """``planners.<stem>`` for a file in planners/, else a name unique to its absolute path."""
    if path.parent == PLANNERS_DIR:
        return f"planners.{path.stem}"
    return f"hitw_planner_{hashlib.sha1(str(path).encode()).hexdigest()[:12]}_{path.stem}"


def _ensure_planners_package() -> None:
    if "planners" not in sys.modules:
        spec = importlib.util.spec_from_file_location("planners", PLANNERS_DIR / "__init__.py",
                                                      submodule_search_locations=[str(PLANNERS_DIR)])
        package = importlib.util.module_from_spec(spec)
        sys.modules["planners"] = package
        spec.loader.exec_module(package)


def load_planner_file(path: str | Path):
    """Import a planner file once, registering the planners in it.

    Parameters
    ----------
    path : str or Path
        A Python file.  A file in planners/ is imported as
        ``planners.<name>`` (the same module however it is reached); any
        other file under a name unique to its absolute path.

    Returns
    -------
    module

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"planner file {path} does not exist")
    name = _module_name(path)
    if name in sys.modules:
        return sys.modules[name]
    if name.startswith("planners."):
        _ensure_planners_package()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def _load_planners() -> None:
    """Import every planners/*.py not imported yet; a file that fails is recorded in _LOAD_ERRORS and skipped."""
    for f in sorted(PLANNERS_DIR.glob("*.py")):
        if f.name.startswith("_") or f.name in _LOAD_ERRORS:
            continue
        try:
            load_planner_file(f)
        except Exception as e:
            _LOAD_ERRORS[f.name] = f"{type(e).__name__}: {e}"


def load_errors() -> dict[str, str]:
    """Return {file name: error} for the files in planners/ that failed to import."""
    _load_planners()
    return dict(_LOAD_ERRORS)


def is_path_ref(ref: str) -> bool:
    """Return True if `ref` has the form ``"path/to/file.py:function"``."""
    head, sep, tail = ref.rpartition(":")
    return bool(sep) and head.endswith(".py") and bool(tail)


def get_planner(ref: str) -> PlannerSpec:
    """Look up a planner by registered id or by ``"path/to/file.py:function"``.

    Parameters
    ----------
    ref : str
        A registered id, or a Python file and the name of a function in it
        that is registered with `planner`.  The file is imported under a
        module name unique to its absolute path, so two files with the same
        name do not collide.

    Returns
    -------
    PlannerSpec

    Raises
    ------
    KeyError
        If no planner has the id; the message lists the registered ids.
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the function is missing from the file, is not registered, or is
        registered under more than one id.
    """
    if is_path_ref(ref):
        file, _, fname = ref.rpartition(":")
        module = load_planner_file(file)
        func = getattr(module, fname, None)
        if func is None:
            raise ValueError(f"{file} has no function {fname!r}")
        specs = [s for s in _REGISTRY.values() if s.func is func]
        if not specs:
            raise ValueError(f"{file}:{fname} is not registered; decorate it with @hitw.api.planner(...)")
        if len(specs) > 1:
            raise ValueError(f"{file}:{fname} is registered under several ids {[s.id for s in specs]}")
        return specs[0]
    _load_planners()
    if ref not in _REGISTRY:
        broken = f"; files in planners/ that failed to import: {_LOAD_ERRORS}" if _LOAD_ERRORS else ""
        raise KeyError(f"unknown planner {ref!r}; registered: {list(_REGISTRY)}, "
                       f"or pass 'path/to/file.py:function'{broken}")
    return _REGISTRY[ref]


def planners(task: str | None = None) -> list[PlannerSpec]:
    """Return the registered planners: those in planners/ by file name, then others in registration order.

    Parameters
    ----------
    task : {"connect", "find"}, optional
        Only planners for this task; None means all.

    Returns
    -------
    list of PlannerSpec
    """
    _load_planners()
    specs = sorted(enumerate(_REGISTRY.values()),
                   key=lambda e: (not e[1].func.__module__.startswith("planners."), e[1].func.__module__, e[0]))
    return [s for _, s in specs if task is None or s.task == task]
