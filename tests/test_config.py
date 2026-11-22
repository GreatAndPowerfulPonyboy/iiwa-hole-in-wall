"""Per-planner typed configs, config labels, and the independent planner random stream."""

import tempfile
from pathlib import Path

import numpy as np

from hitw.api import get_planner, parse_ref
from hitw.config import config_fields, resolve_config
from planners.ik_restarts import K_DEFAULT, IkRestartsConfig
from planners.uniform_sampling import UniformSamplingConfig
from hitw.harness import planner_rng, resolve, run_trial, trial_rng
from planners.gcs import IrisGcsConfig
from planners.rrt_connect import MAX_SAMPLES, RrtConnectConfig
from planners.kto import MAJOR_ITER_LIMIT, KtoConfig
from hitw.sweep import grid, run_sweep


def _raises(fn, *words, exc=ValueError):
    try:
        fn()
    except exc as e:
        msg = str(e)
        assert all(w in msg for w in words), msg
        return msg
    raise AssertionError(f"no {exc.__name__}")


def test_defaults_equal_module_constants():
    assert KtoConfig().major_iter_limit == MAJOR_ITER_LIMIT
    assert RrtConnectConfig().max_samples == MAX_SAMPLES
    assert IkRestartsConfig().K == K_DEFAULT and UniformSamplingConfig().max_samples == 1_000_000
    assert IrisGcsConfig().iris == "zo" and IrisGcsConfig().margin == "scaled"
    assert list(config_fields(RrtConnectConfig)) == ["max_samples", "step", "goal_bias", "shortcut_iters"]


def test_unknown_option_names_valid_fields():
    _raises(lambda: resolve_config(RrtConnectConfig, {"max_sampels": 10}), "max_sampels", "max_samples", "shortcut_iters")
    _raises(lambda: resolve("kto", {"max_samples": 10}), "kto", "max_samples", "major_iter_limit")
    _raises(lambda: resolve("gcs-box", {"max_samples": 10}), "max_samples", "max_rounded_paths")
    _raises(lambda: resolve("rrt-connect", {"major_iter_limit": 1}), "major_iter_limit", "max_samples")
    _raises(lambda: resolve("ik-restarts[max_samples=10]"), "max_samples", "bound_factor")


def test_string_values_are_converted():
    c = resolve_config(KtoConfig, {"major_iter_limit": "50", "retry": "false", "bound": "none"})
    assert c.major_iter_limit == 50 and c.retry is False and c.bound is None
    assert resolve_config(KtoConfig, {"bound": "0.004"}).bound == 0.004
    assert resolve_config(IrisGcsConfig, {"margin": "0.001"}).margin == 0.001
    assert resolve_config(IrisGcsConfig, {"margin": "default"}).margin == "default"
    _raises(lambda: resolve_config(RrtConnectConfig, {"max_samples": "many"}), "max_samples")


def test_iris_config_validation():
    _raises(lambda: IrisGcsConfig(iris="np3"), "zo")
    _raises(lambda: IrisGcsConfig(margin="tight"), "scaled")
    _raises(lambda: IrisGcsConfig(margin=-1.0), "positive")


def test_config_labels_round_trip():
    spec = get_planner("iris-gcs")
    for overrides, label in (({}, "iris-gcs"), ({"iris": "np2"}, "iris-gcs[iris=np2]"),
                             ({"margin": "default"}, "iris-gcs[margin=default]"),
                             ({"iris": "np", "margin": 0.001}, "iris-gcs[iris=np,margin=0.001]"),
                             ({"iris": "zo"}, "iris-gcs")):
        cfg = spec.make_config(overrides)
        assert spec.config_label(cfg) == label, (overrides, spec.config_label(cfg))
        s, c = resolve(label)
        assert s is spec and c == cfg
    kto = get_planner("kto")
    assert kto.config_label(kto.make_config({"retry": False, "bound": 0.004})) == "kto[bound=0.004,retry=false]"
    assert resolve("kto[bound=0.004,retry=false]")[1] == KtoConfig(bound=0.004, retry=False)
    assert parse_ref("a.py:f[x=1, y = two]") == ("a.py:f", {"x": "1", "y": "two"})
    _raises(lambda: parse_ref("kto[retry]"), "key=value")
    assert resolve("kto[retry=false]", {"retry": "false"})[1].retry is False
    _raises(lambda: resolve("kto[retry=false]", {"retry": True}), "different values")


def test_old_names_are_rejected_with_the_valid_ids():
    for old in ("KTO", "M1", "GCS", "IrisNp2+GCS", "RRT-Connect", "B"):
        _raises(lambda: resolve(old), "kto", "gcs-box", "iris-gcs", "rrt-connect", "ik-restarts", exc=KeyError)
        _raises(lambda: run_trial(old, 0, 0.0, 0.02, 1), "kto", exc=KeyError)
        _raises(lambda: grid([old], [0], [0.0], [0.02], [1]), "kto", exc=KeyError)


def test_run_trial_rejects_bad_config_before_building_the_problem():
    _raises(lambda: run_trial("kto", 0, 0.0, 0.02, 1, config={"max_samples": 10}), "max_samples")
    _raises(lambda: run_trial("ik-restarts", 0, 0.0, 0.02, 1), "find planner")


def test_grid_rejects_bad_config_and_duplicate_labels():
    _raises(lambda: grid([("rrt-connect", {"max_sampels": 1})], [0], [0.0], [0.02], [1]), "max_sampels")
    _raises(lambda: grid(["iris-gcs", "iris-gcs[iris=zo]"], [0], [0.0], [0.02], [1]), "iris-gcs")
    trials = grid(["iris-gcs", "iris-gcs[iris=np2]", ("iris-gcs", {"margin": "default"})], [0, 1], [0.0], [0.02], [1, 2])
    assert sorted({t.config_label for t in trials}) == ["iris-gcs", "iris-gcs[iris=np2]", "iris-gcs[margin=default]"]
    assert len(trials) == 3 * 2 * 2 and len({t.key for t in trials}) == len(trials)
    finds = grid(["ik-restarts", "uniform-sampling"], [0], [0.0], [0.02], [1, 2])
    assert [t.n_walls for t in finds] == [1, 1]                           # find trials have one wall
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.jsonl"
        _raises(lambda: run_sweep(trials, p, cap={"IrisZo+GCS": 10.0}, verbose=False), "IrisZo+GCS")
        assert not p.exists()


def test_planner_stream_is_independent_and_reproducible():
    for seed in (0, 3, 99):
        a, b = trial_rng(seed), planner_rng(seed)
        assert a.bit_generator.state["state"] != b.bit_generator.state["state"]
        assert not np.allclose(a.uniform(size=8), b.uniform(size=8))
        assert np.array_equal(planner_rng(seed).uniform(size=8), planner_rng(seed).uniform(size=8))
    assert not np.allclose(planner_rng(0).uniform(size=8), planner_rng(1).uniform(size=8))
    # the problem stream is unchanged: still default_rng(seed)
    assert np.array_equal(trial_rng(5).uniform(size=4), np.random.default_rng(5).uniform(size=4))
