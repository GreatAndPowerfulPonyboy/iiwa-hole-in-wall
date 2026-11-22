"""M4: bidirectional RRT-Connect with shortcutting, one segment per transit
between windows, time-aware.

The wall moves, so a configuration-space planner alone cannot say whether an
edge is free: each edge is a straight line in (q, t), checked with the wall
at wall_y(t) at every sample, at joint-space resolution finer than eps / rho,
and must respect the per-joint velocity limits.  Samples are configurations
only; the forward tree assigns each node its earliest arrival time (max-speed
travel from the parent) and the backward tree its latest departure time, so
the trees join wherever the max-speed travel time between two nodes fits
between their times.  (Sampling uniformly in (q, t) was tried first: most
samples are velocity-unreachable from their nearest node, and the trees never
joined on an easy 3-D test problem.)  Each window is an intermediate goal:
the segment ends at (q*_i, t_i - delta_i), the arm holds q*_i through the
window, and the next segment starts at (q*_i, t_i + delta_i).

Pattern: reference ps3/rrt_planning.py (extend / connect / join).
Acceleration limits are not enforced: the path is piecewise linear in time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .scene import Scene, in_collision
from .sequences import Sequence

MAX_SAMPLES = 5000          # per segment
STEP = 0.5                  # rad, max extension length in q
GOAL_BIAS = 0.1
SHORTCUT_ITERS = 200
MAX_EDGE_STEP = 0.01        # rad, upper bound on the edge-check resolution
CONNECT_MIN_PROGRESS = 1e-6 # metric progress per connect step below which the loop stops


@dataclass
class RrtResult:
    status: str
    knots: list | None                  # [(q_full, t), ...] in time order, or None
    solve_time: float
    iterations: int                     # samples drawn over all segments
    extras: dict = field(default_factory=dict)


class _Tree:
    """Nodes carry (q, t).  The forward tree stores the earliest arrival time at
    each node (max-speed travel from its parent), the backward tree the latest
    departure time; a node of one tree can be joined to a node of the other
    when the max-speed travel time fits between the two times."""

    def __init__(self, q: np.ndarray, t: float, forward: bool, vmax: np.ndarray):
        self.qs = [np.asarray(q, dtype=float)]
        self.ts = [float(t)]
        self.parents = [-1]
        self.forward = forward
        self.vmax = np.asarray(vmax, dtype=float)

    def add(self, q, t, parent) -> int:
        self.qs.append(np.asarray(q, dtype=float))
        self.ts.append(float(t))
        self.parents.append(parent)
        return len(self.qs) - 1

    def needed(self, q_from, q_to) -> float:
        """Max-speed travel time between two configurations."""
        return float((np.abs(np.asarray(q_to) - np.asarray(q_from)) / self.vmax).max())

    def nearest(self, q, t: float | None = None) -> int | None:
        """Nearest node in q.  With a target time t, only nodes that can reach
        (q, t) within the speed limits qualify (None if there is none)."""
        qs = np.array(self.qs)
        d = ((qs - q) ** 2).sum(1)
        if t is not None:
            ts = np.array(self.ts)
            slack = (t - ts) if self.forward else (ts - t)
            needed = (np.abs(qs - q) / self.vmax).max(axis=1)
            ok = slack >= needed
            if not ok.any():
                return None
            d[~ok] = np.inf
        return int(np.argmin(d))

    def path(self, idx) -> list[tuple[np.ndarray, float]]:
        out = []
        while idx != -1:
            out.append((self.qs[idx], self.ts[idx]))
            idx = self.parents[idx]
        return out[::-1]

    def is_root(self, q, t) -> bool:
        return bool(np.allclose(q, self.qs[0], atol=1e-9) and abs(t - self.ts[0]) < 1e-9)


class _Segment:
    def __init__(self, scene: Scene, wall_y, vmax: np.ndarray, res: float, rng: np.random.Generator):
        self.scene, self.wall_y, self.vmax, self.res, self.rng = scene, wall_y, vmax, res, rng
        self.lo, self.hi = scene.free_limits()
        self.checks = 0
        self.t_lo, self.t_hi = 0.0, np.inf

    def velocity_ok(self, qa, ta, qb, tb) -> bool:
        dt = abs(tb - ta)
        return dt > 0 and bool(np.all(np.abs(qb - qa) <= self.vmax * dt + 1e-12))

    def edge_free(self, qa, ta, qb, tb) -> bool:
        n = max(1, int(np.ceil(np.linalg.norm(qb - qa) / self.res)))
        for k in range(1, n + 1):
            a = k / n
            q = qa + (qb - qa) * a
            t = ta + (tb - ta) * a
            self.checks += 1
            if in_collision(self.scene, self.scene.q_full(q), self.wall_y(t)):
                return False
        return True

    def extend(self, tree: _Tree, qs, ts: float | None = None):
        """One step of at most STEP rad from the nearest node toward qs.  With
        ts = None the new node's time is the max-speed arrival (forward) or
        departure (backward) time; with a target time the step keeps the
        straight line in (q, t) toward (qs, ts), which the nearest() filter
        has made velocity-feasible."""
        near = tree.nearest(qs, ts)
        if near is None:
            return "Trapped", None
        qa, ta = tree.qs[near], tree.ts[near]
        dq = qs - qa
        dist = float(np.linalg.norm(dq))
        if dist < 1e-12:
            return "Trapped", None
        frac = min(1.0, STEP / dist)
        qn = qa + dq * frac
        if ts is None:
            dt = tree.needed(qa, qn)
            tn = ta + dt if tree.forward else ta - dt
            if not (self.t_lo <= tn <= self.t_hi):
                return "Trapped", None
        else:
            tn = ta + (ts - ta) * frac
        if not self.edge_free(qa, ta, qn, tn):
            return "Trapped", None
        idx = tree.add(qn, tn, near)
        reached = frac >= 1.0 and (ts is None or abs(tn - ts) < 1e-9)
        return ("Reached" if reached else "Advanced"), idx

    def connect(self, tree: _Tree, qs, ts):
        """Extend repeatedly toward the state (qs, ts) while each step makes
        progress in q."""
        best = np.inf
        while True:
            status, idx = self.extend(tree, qs, ts)
            if status != "Advanced":
                return status, idx
            d = float(np.linalg.norm(tree.qs[idx] - qs))
            if d >= best - CONNECT_MIN_PROGRESS:
                return "Trapped", idx
            best = d

    def sample(self):
        return self.rng.uniform(self.lo, self.hi)

    def plan(self, q_start, t_start, q_goal, t_goal, max_samples: int):
        """Bidirectional RRT-Connect between two (q, t) states.  Returns
        (knots or None, samples, status)."""
        if not self.velocity_ok(q_start, t_start, q_goal, t_goal):
            return None, 0, "infeasible"
        if self.edge_free(q_start, t_start, q_goal, t_goal):
            return [(q_start, t_start), (q_goal, t_goal)], 0, "success"
        self.t_lo, self.t_hi = t_start, t_goal
        a, b = _Tree(q_start, t_start, True, self.vmax), _Tree(q_goal, t_goal, False, self.vmax)
        for n in range(1, max_samples + 1):
            if self.rng.uniform() < GOAL_BIAS:
                qs, ts = (b.qs[0], b.ts[0])            # the other tree's root, with its time
            else:
                qs, ts = self.sample(), None
            status, idx = self.extend(a, qs, ts)
            if status != "Trapped":
                qn, tn = a.qs[idx], a.ts[idx]
                if b.is_root(qn, tn):                   # a reached the other endpoint directly
                    pa = a.path(idx)
                    return (pa if a.forward else pa[::-1]), n, "success"
                status_b, idx_b = self.connect(b, qn, tn)
                if status_b == "Reached":
                    pa, pb = a.path(idx), b.path(idx_b)
                    knots = (pa + pb[::-1][1:]) if a.forward else (pb + pa[::-1][1:])
                    return knots, n, "success"
            a, b = b, a
        return None, max_samples, "iteration_limit"

    def shortcut(self, knots, iters: int):
        knots = list(knots)
        for _ in range(iters):
            if len(knots) < 3:
                break
            i, j = sorted(self.rng.integers(0, len(knots), size=2))
            if j - i < 2:
                continue
            (qa, ta), (qb, tb) = knots[i], knots[j]
            if self.velocity_ok(qa, ta, qb, tb) and self.edge_free(qa, ta, qb, tb):
                knots = knots[:i + 1] + knots[j:]
        return knots


def plan_rrt(seq: Sequence, scene: Scene, seed: int = 0, *, max_samples: int = MAX_SAMPLES,
             shortcut_iters: int = SHORTCUT_ITERS, **kw) -> RrtResult:
    assert scene.wall_joint is not None
    rng = np.random.default_rng(seed)
    res = min(MAX_EDGE_STEP, 0.5 * seq.eps / seq.rho)
    seg = _Segment(scene, seq.wall_y_of_t(), scene.velocity_limits(), res, rng)
    t0 = time.perf_counter()
    knots_all: list[tuple[np.ndarray, float]] = []
    q_prev, t_prev = scene.q_free(seq.q0), 0.0
    total, per_segment = 0, []
    status = "success"
    for c in seq.crossings:
        q_goal = scene.q_free(c.q_star)
        knots, n, st = seg.plan(q_prev, t_prev, q_goal, c.t - c.delta, max_samples)
        total += n
        per_segment.append({"samples": n, "status": st, "knots": len(knots) if knots else 0})
        if knots is None:
            status = st
            break
        knots = seg.shortcut(knots, shortcut_iters)
        per_segment[-1]["knots_after_shortcut"] = len(knots)
        knots_all += knots if not knots_all else knots[1:]
        knots_all.append((q_goal, c.t + c.delta))         # hold through the window
        q_prev, t_prev = q_goal, c.t + c.delta
    dt = time.perf_counter() - t0
    if status != "success":
        return RrtResult(status=status, knots=None, solve_time=dt, iterations=total,
                         extras={"segments": per_segment, "edge_resolution": res, "checks": seg.checks})
    knots_full = [(scene.q_full(q), t) for q, t in knots_all]
    return RrtResult(status="success", knots=knots_full, solve_time=dt, iterations=total,
                     extras={"segments": per_segment, "edge_resolution": res, "checks": seg.checks})


def rrt_q_of_t(knots: list, T: float):
    """Piecewise-linear interpolation in time; holds the last knot until T."""
    ts = np.array([t for _, t in knots])
    qs = np.array([q for q, _ in knots])

    def q_of_t(t: float) -> np.ndarray:
        t = float(np.clip(t, ts[0], ts[-1]))
        return np.array([np.interp(t, ts, qs[:, j]) for j in range(qs.shape[1])])
    return q_of_t
