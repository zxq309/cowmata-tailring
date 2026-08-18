# -*- coding: utf-8 -*-
"""Contract tests for the 20260819 refactor.

Runnable two ways::

    python tests/test_contracts.py     # no pytest needed
    pytest tests/test_contracts.py

Nothing here needs torch, xgboost or a GPU, so the whole file runs on any
machine that can import numpy, pandas, scipy and scikit-learn.  Tests that need
the deployed joblib bundle skip themselves when it is absent.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cowmata.cache import (  # noqa: E402
    Calibration,
    Segment,
    estimate_storage_bytes,
    mask_to_intervals,
    open_cache,
    write_cache_v2,
)
from cowmata.daily import DailyConfig, individual_baseline, mil_day_alert  # noqa: E402
from cowmata.features import (  # noqa: E402
    FEATURE_VERSION,
    estimate_reference,
    gravity_split,
    segment_features,
    session_reference,
    window_stats,
)
from cowmata.labels import (  # noqa: E402
    DEPRECATED_EVENT_CODES,
    EVENT_CODES,
    STATE_ANNOTATION_CODES,
)
from cowmata.metrics import (  # noqa: E402
    bootstrap_ci,
    choose_threshold,
    edit_score,
    event_level_metrics,
    f1_at_tiou,
    postprocess_intervals,
    rate_plausibility,
    selection_score,
)
from cowmata.postprocess import assemble_intervals, hysteresis_mask  # noqa: E402
from cowmata.preprocessing import dense_label_frame, label_centers, label_grid  # noqa: E402
from cowmata.tools import mine_candidates, plan_storage  # noqa: E402
from cowmata.train import grouped_folds  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_CACHE = (
    PROJECT_ROOT / "examples/demo_data/supervised_cache/session_cache/demo_session_60s"
)
DEPLOYED_BUNDLE = PROJECT_ROOT / "weights/deploy/gbdt_full.joblib"


# ---------------------------------------------------------------- taxonomy
def test_taxonomy_is_internally_consistent() -> None:
    assert len(set(EVENT_CODES)) == len(EVENT_CODES)
    assert not set(EVENT_CODES) & set(DEPRECATED_EVENT_CODES)
    assert "MOUNTED_BY" in EVENT_CODES, "being mounted is the oestrus gold standard"
    assert "TAIL_WAGGING" in DEPRECATED_EVENT_CODES
    assert "FEEDING" in STATE_ANNOTATION_CODES, "historical feeding labels stay readable"


def test_deprecated_code_is_read_but_never_trained() -> None:
    times = np.arange(0, 20000, 500, dtype=np.int64)
    annotations = pd.DataFrame(
        [
            dict(
                device_mac="A",
                session_id="S",
                code="TAIL_WAGGING",
                t_start_rel_ms=1000,
                t_end_rel_ms=4000,
            )
        ]
    )
    body, events, masks = label_centers(times, annotations, None)
    assert events.shape[1] == len(EVENT_CODES)
    assert events.sum() == 0, "a deprecated code must not become a training target"


# ---------------------------------------------------------------- cache
def test_schema2_roundtrip_is_lossless_in_counts() -> None:
    rng = np.random.default_rng(0)
    counts = rng.integers(-3000, 3000, (400, 9)).astype(np.int16)
    quality = np.zeros(400, dtype=np.float32)
    quality[30:44] = 1.0
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "s"
        write_cache_v2(
            target,
            counts=counts,
            segments=[Segment(0, 0, 250, 0, 4980), Segment(1, 250, 400, 900000, 902980)],
            calibration=Calibration(4096.0),
            quality_flag=quality,
            metadata={"cache_key": "s", "cow_id": "c", "tail_position": "mid"},
        )
        # The context manager closes the memory map so the temporary directory
        # can be removed on Windows, where open files are locked.
        with open_cache(target) as cache:
            assert cache.schema_version == 2
            assert cache.tail_position == "mid"
            recovered = cache.physical()[:, 0] * 4096.0
            assert np.allclose(recovered, counts[:, 0], atol=1e-3)
            assert np.array_equal(cache.quality_flag(), quality)
            assert cache.times_ms(np.asarray([0, 249, 250]))[2] == 900000


def test_sparse_quality_intervals_are_exact() -> None:
    flag = np.zeros(100, dtype=np.float32)
    flag[3:7] = 1
    flag[50:51] = 1
    assert mask_to_intervals(flag) == [[3, 7], [50, 51]]


def test_schema1_demo_cache_still_reads() -> None:
    if not DEMO_CACHE.exists():
        print("  [skip] demo cache absent")
        return
    cache = open_cache(DEMO_CACHE)
    assert cache.schema_version == 1
    assert cache.physical(0, 10).shape == (10, 9)
    assert cache.n_frames == 2999


def test_storage_plan_matches_the_documented_numbers() -> None:
    plan = plan_storage(200, 7)
    assert plan["frames"] == 200 * 7 * 24 * 3600 * 50
    assert 95.0 < plan["schema2_gigabytes"] < 110.0
    assert plan["saving_ratio"] > 2.5
    assert estimate_storage_bytes(200, 7, 1)["bytes_per_frame"] == 52


# ---------------------------------------------------------------- labels
def _urination_annotations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(device_mac="A", session_id="S", code="STANDING", t_start_rel_ms=0, t_end_rel_ms=60000),
            dict(device_mac="A", session_id="S", code="URINATION", t_start_rel_ms=10000, t_end_rel_ms=30000),
        ]
    )


def test_tail_raised_policies_behave_as_documented() -> None:
    times = np.arange(0, 60000, 500, dtype=np.int64)
    annotations = _urination_annotations()
    tail = EVENT_CODES.index("TAIL_RAISED")
    inside = (times >= 10000) & (times < 30000)

    _, events, masks = label_centers(times, annotations, None, tail_raised_policy="legacy")
    assert events[inside, tail].sum() == 0 and masks[inside, tail].sum() > 0

    _, events, masks = label_centers(times, annotations, None, tail_raised_policy="derive")
    assert events[inside, tail].all() and masks[inside, tail].all()

    _, events, masks = label_centers(times, annotations, None, tail_raised_policy="exclude")
    assert events[inside, tail].sum() == 0 and masks[inside, tail].sum() == 0


def test_body_state_overlap_raises() -> None:
    times = np.arange(0, 10000, 500, dtype=np.int64)
    annotations = pd.DataFrame(
        [
            dict(device_mac="A", session_id="S", code="STANDING", t_start_rel_ms=0, t_end_rel_ms=10000),
            dict(device_mac="A", session_id="S", code="LYING", t_start_rel_ms=0, t_end_rel_ms=10000),
        ]
    )
    try:
        label_centers(times, annotations, None)
    except ValueError:
        return
    raise AssertionError("overlapping body annotations must raise")


def test_label_grid_never_bridges_a_segment() -> None:
    segments = [Segment(0, 0, 3000, 0, 59980), Segment(1, 3000, 5000, 3600000, 3639980)]
    centers, times, starts, stops = label_grid(segments)
    assert centers.size == 120 + 80
    crossing = (centers >= starts) & (centers < stops)
    assert crossing.all(), "every centre must lie inside its own segment"
    assert times[120] == 3600000, "the second segment restarts at its own timestamp"


def test_dense_frame_is_a_superset_of_the_supervised_frame() -> None:
    segments = [Segment(0, 0, 3000, 0, 59980)]
    dense = dense_label_frame(
        segments=segments,
        annotations=_urination_annotations(),
        cache_name="k",
        cow_id="c",
        device_key="A",
        device_mac="A",
        session_id="S",
    )
    assert len(dense) == 120
    assert {f"mask_{code}" for code in EVENT_CODES} <= set(dense.columns)


# ---------------------------------------------------------------- features
def _synthetic_block(n: int = 50 * 300, lever: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(11)
    array = np.zeros((n, 9), dtype=np.float32)
    array[:, 0] = rng.normal(0.0, 0.02, n)
    array[:, 1] = 0.2 + rng.normal(0.0, 0.02, n)
    array[:, 2] = 0.97 + rng.normal(0.0, 0.02, n)
    array[:, 3:6] = rng.normal(0.0, 1.0, (n, 3)) * lever
    array[5000:6500, 0] = 0.85
    array[5000:6500, 2] = 0.50
    array[5000:6500, 3:6] *= 20.0
    return array


def test_feature_version_1_has_exactly_the_deployed_104_columns() -> None:
    block = _synthetic_block(50 * 120)
    columns = list(
        segment_features(
            block, np.arange(0, block.shape[0], 25), causal=False, feature_version=1
        ).columns
    )
    assert len(columns) == 104, "the deployed GBDT was trained on 104 features"
    if not DEPLOYED_BUNDLE.exists():
        print("  [skip] deployed bundle absent; name comparison not run")
        return
    raw = DEPLOYED_BUNDLE.read_bytes()
    strings: list[str] = []
    index = 0
    while index < len(raw):
        if raw[index] == 0x8C:  # pickle SHORT_BINUNICODE
            length = raw[index + 1]
            try:
                strings.append(raw[index + 2 : index + 2 + length].decode("utf-8"))
            except UnicodeDecodeError:
                pass
            index += 2 + length
        else:
            index += 1
    start = strings.index(columns[0])
    assert strings[start : start + 104] == columns, (
        "feature_version=1 must reproduce the deployed feature list, in order"
    )


def test_feature_version_2_is_a_named_superset_with_different_gyro_values() -> None:
    block = _synthetic_block()
    reference = estimate_reference([block], causal=False)
    centers = np.arange(0, block.shape[0], 25)
    v1 = segment_features(block, centers, causal=False, reference=reference, feature_version=1)
    v2 = segment_features(block, centers, causal=False, reference=reference, feature_version=2)
    assert set(v1.columns) < set(v2.columns)
    assert FEATURE_VERSION == 2
    non_angular = [
        c for c in v1.columns if "gyro" not in c and "band" not in c and "dot" not in c
    ]
    assert all(np.allclose(v1[c], v2[c]) for c in non_angular)
    assert not np.allclose(v1["gyro_norm_mean_5s"], v2["gyro_norm_mean_5s"])


def test_orientation_calibration_puts_resting_tilt_near_zero() -> None:
    block = _synthetic_block()
    static, dynamic = gravity_split(block[:, 0:3].astype(np.float64), causal=False)
    reference = session_reference(
        static, np.linalg.norm(dynamic, axis=1), np.linalg.norm(block[:, 3:6], axis=1)
    )
    centers = np.arange(0, block.shape[0], 25)
    tilt = segment_features(block, centers, causal=False, reference=reference)[
        "tilt_mean_5s"
    ].to_numpy()
    episode = (centers >= 5200) & (centers < 6300)
    assert tilt[~episode].mean() < 3.0
    assert tilt[episode].mean() > 30.0


def test_amplitude_calibration_makes_lever_arms_comparable() -> None:
    """The same behaviour at the tail root and at mid-tail must score alike."""

    root = _synthetic_block(lever=1.0)
    mid = _synthetic_block(lever=3.0)
    centers = np.arange(0, root.shape[0], 25)
    values = []
    for block, version in ((root, 2), (mid, 2)):
        reference = estimate_reference([block], causal=False)
        table = segment_features(
            block, centers, causal=False, reference=reference, feature_version=version
        )
        values.append(float(table["gyro_norm_mean_5s"].mean()))
    calibrated_ratio = values[1] / values[0]
    uncalibrated = []
    for block in (root, mid):
        reference = estimate_reference([block], causal=False)
        table = segment_features(
            block, centers, causal=False, reference=reference, feature_version=1
        )
        uncalibrated.append(float(table["gyro_norm_mean_5s"].mean()))
    raw_ratio = uncalibrated[1] / uncalibrated[0]
    assert raw_ratio > 2.5, "the uncalibrated feature must track the lever arm"
    assert abs(calibrated_ratio - 1.0) < 0.15, (
        f"amplitude calibration failed: ratio {calibrated_ratio:.3f}"
    )


def test_causal_features_do_not_see_the_future() -> None:
    block = _synthetic_block()
    static, dynamic = gravity_split(block[:, 0:3].astype(np.float64), causal=True)
    reference = session_reference(
        static, np.linalg.norm(dynamic, axis=1), np.linalg.norm(block[:, 3:6], axis=1)
    )
    centers = np.arange(0, block.shape[0], 25)
    base = segment_features(block, centers, causal=True, reference=reference)
    edited = block.copy()
    rng = np.random.default_rng(5)
    edited[10000:] = rng.normal(0.0, 1.0, edited[10000:].shape).astype(np.float32)
    after = segment_features(edited, centers, causal=True, reference=reference)
    cut = int(np.searchsorted(centers, 9000))
    difference = np.nanmax(np.abs(base.iloc[:cut].to_numpy() - after.iloc[:cut].to_numpy()))
    assert difference == 0.0, "a causal feature changed when future samples changed"


def test_rolling_statistics_are_exact() -> None:
    mean, _ = window_stats(np.arange(10.0), -2, 1)
    assert np.allclose(mean, [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8])
    mean, std = window_stats(np.ones(5), -1, 2)
    assert np.allclose(mean, 1.0) and np.allclose(std, 0.0)


def test_features_have_no_nan() -> None:
    block = _synthetic_block(50 * 60)
    table = segment_features(block, np.arange(0, block.shape[0], 25), causal=False)
    assert not table.isna().any().any()


# ---------------------------------------------------------------- metrics
def test_threshold_matches_the_reference_implementation() -> None:
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(3)
    for _ in range(5):
        n = 1500
        target = (rng.random(n) < 0.05).astype(np.int8)
        probability = np.clip(0.3 * target + rng.random(n) * 0.8, 0, 1)
        candidates = np.unique(np.concatenate((np.linspace(0.05, 0.95, 37), probability)))
        reference = max(f1_score(target, probability >= t, zero_division=0) for t in candidates)
        chosen = choose_threshold(target, probability, np.ones(n, dtype=bool))
        assert abs(f1_score(target, probability >= chosen, zero_division=0) - reference) < 1e-12


def test_threshold_is_stable_on_degenerate_input() -> None:
    assert choose_threshold(np.zeros(10, np.int8), np.random.rand(10), np.ones(10, bool)) == 0.5
    assert choose_threshold(np.ones(10, np.int8), np.random.rand(10), np.ones(10, bool)) == 0.5


def test_edit_score_punishes_over_segmentation_and_nothing_else() -> None:
    truth = np.concatenate([np.zeros(20), np.ones(20), np.zeros(20)])
    assert edit_score(truth, truth) == 100.0
    shifted = np.concatenate([np.zeros(22), np.ones(18), np.zeros(20)])
    assert edit_score(truth, shifted) == 100.0, "a boundary shift is not over-segmentation"
    fragmented = truth.copy()
    fragmented[28] = 0
    fragmented[33] = 0
    assert edit_score(truth, fragmented) < 60.0


def test_f1_at_tiou_scales_with_the_event_length() -> None:
    truth = [(0, 10000)]
    close = [(500, 9500)]
    loose = [(0, 30000)]
    assert f1_at_tiou(truth, close)["f1@50"]["f1"] == 1.0
    assert f1_at_tiou(truth, loose)["f1@50"]["f1"] == 0.0
    assert f1_at_tiou(truth, loose)["f1@25"]["f1"] == 1.0


def test_bootstrap_resamples_cows_not_rows() -> None:
    values = np.concatenate([np.full(1000, 0.95), np.full(3, 0.4)])
    groups = np.asarray(["big"] * 1000 + ["a", "b", "c"])
    interval = bootstrap_ci(values, groups)
    assert interval["groups"] == 4
    assert interval["high"] - interval["low"] > 0.1, (
        "one dominant animal must not produce a narrow interval"
    )


def test_selection_score_ignores_unevaluable_events() -> None:
    score = selection_score(0.9, 0.8, {"URINATION": 0.5, "MOUNTED_BY": None})
    assert score["events_used"] == ["URINATION"]
    assert 0.0 < float(score["selection_score"]) < 1.0


def test_postprocess_merges_and_drops() -> None:
    assert postprocess_intervals([(0, 20000), (21000, 35000)], "URINATION") == [(0, 35000)]
    assert postprocess_intervals([(0, 500)], "URINATION") == []


def test_rate_plausibility_flags_an_impossible_rate() -> None:
    assert rate_plausibility("URINATION", 1372, 32.2)["verdict"] == "implausible"
    assert rate_plausibility("URINATION", 10, 30.0)["verdict"] == "plausible"


def test_event_report_refuses_to_claim_on_thin_evidence() -> None:
    times = np.arange(0, 60000, 500, dtype=np.int64)
    frame = pd.DataFrame(
        {
            "device_mac": "A",
            "session_id": "S",
            "cow_id": "cow-1",
            "center_time_ms": times,
            "target_URINATION": ((times >= 10000) & (times < 30000)).astype(np.uint8),
            "mask_URINATION": 1,
            "prob_URINATION": np.where((times >= 10000) & (times < 30000), 0.9, 0.1),
        }
    )
    report = event_level_metrics(frame, "URINATION", 0.5, annotations=_urination_annotations())
    assert report["true_events"] == 1
    assert report["recall"] == 1.0
    assert report["precision_f1_claimable"] is False
    assert report["evidence_level"] == "not_evaluable"
    assert report["segment_level"]["f1@50"]["f1"] is not None
    assert report["edit_score"] is not None


# ---------------------------------------------------------------- postprocess
def test_hysteresis_keeps_one_event_whole_where_one_threshold_splits_it() -> None:
    times = np.arange(0, 60000, 500, dtype=np.int64)
    scores = np.where((times >= 10000) & (times < 30000), 0.9, 0.05)
    scores[(times >= 19500) & (times < 20500)] = 0.35  # a 1 s dip inside one event

    single = assemble_intervals(
        times, scores, "URINATION", threshold=0.5, low_threshold=0.5, postprocess=False
    )
    hysteresis = assemble_intervals(times, scores, "URINATION", threshold=0.5, postprocess=False)
    assert len(single) == 2, "the 20260818 rule fragments this event"
    assert hysteresis == [(10000, 30000)], "hysteresis must keep it whole"


def test_hysteresis_never_starts_an_event_below_the_high_threshold() -> None:
    scores = np.full(50, 0.45)
    assert not hysteresis_mask(scores, high=0.8, low=0.3).any()


def test_boundary_snapping_moves_the_edges_to_the_boundary_peaks() -> None:
    times = np.arange(0, 60000, 500, dtype=np.int64)
    scores = np.where((times >= 10000) & (times < 30000), 0.9, 0.05)
    boundary = np.zeros_like(scores)
    boundary[(times >= 9500) & (times < 10500)] = 0.9
    boundary[(times >= 29500) & (times < 30500)] = 0.9
    snapped = assemble_intervals(times, scores, "URINATION", threshold=0.5, boundary=boundary)
    assert snapped == [(9500, 29500)]


# ---------------------------------------------------------------- splits
def test_grouped_folds_are_cow_disjoint_and_cover_every_cow() -> None:
    sessions = pd.DataFrame(
        {
            "cow_id": [f"c{i:03d}" for i in range(50)],
            "device_mac": [f"M{i}" for i in range(50)],
            "session_id": "S",
        }
    )
    folds = grouped_folds(sessions, n_folds=5)
    tested = [cow for fold in folds for cow in fold["test_cows"]]
    assert len(tested) == 50 and len(set(tested)) == 50
    for fold in folds:
        assert not set(fold["train_cows"]) & set(fold["test_cows"])
        assert not set(fold["validation_cows"]) & set(fold["test_cows"])
        assert not set(fold["validation_cows"]) & set(fold["train_cows"]), (
            "validation must be cow-disjoint from training, not merely session-disjoint"
        )


def test_leave_one_cow_out_is_the_limiting_case() -> None:
    sessions = pd.DataFrame(
        {"cow_id": list("abcdef"), "device_mac": list("ABCDEF"), "session_id": "S"}
    )
    folds = grouped_folds(sessions, n_folds=6)
    assert folds[0]["protocol"] == "strict_loco"
    assert all(len(fold["test_cows"]) == 1 for fold in folds)


# ---------------------------------------------------------------- daily
def test_baseline_after_the_event_excludes_the_event_and_its_neighbours() -> None:
    frame = pd.DataFrame({"day": list(range(7)), "lying_hours": [13, 13, 12, 5, 12, 13, 13.0]})
    centre, _, days = individual_baseline(
        frame, "lying_hours", event_day=3, config=DailyConfig(baseline_window="after")
    )
    assert days == 2, "days 5 and 6 only: day 4 is adjacent to the event"
    assert centre > 10.0, "the oestrus day must not enter its own baseline"


def test_mil_requires_several_windows_not_one_spike() -> None:
    quiet = np.full(1000, 0.1)
    quiet[7] = 0.99
    assert not mil_day_alert(quiet), "one spike is not a day of oestrus"
    active = np.concatenate([np.full(900, 0.1), np.full(100, 0.8)])
    assert mil_day_alert(active)


# ---------------------------------------------------------------- mining
def test_mining_refuses_to_drop_the_random_control_group() -> None:
    frame = pd.DataFrame(
        {
            "device_mac": "A",
            "session_id": "S",
            "cow_id": "c",
            "center_time_ms": np.arange(0, 100000, 500),
            "prob_URINATION": np.random.default_rng(0).random(200),
        }
    )
    try:
        mine_candidates(frame, ["URINATION"], random_fraction=0.0)
    except ValueError:
        return
    raise AssertionError("a review round without a random control must be refused")


def test_mined_rows_are_never_labels() -> None:
    rng = np.random.default_rng(0)
    frames = []
    for cow in ("c1", "c2", "c3"):
        times = np.arange(0, 3600000, 500)
        scores = rng.random(times.size) * 0.2
        scores[(times > 600000) & (times < 640000)] = 0.95
        frames.append(
            pd.DataFrame(
                {
                    "device_mac": cow,
                    "session_id": "S",
                    "cow_id": cow,
                    "center_time_ms": times,
                    "prob_URINATION": scores,
                }
            )
        )
    queue, manifest = mine_candidates(pd.concat(frames, ignore_index=True), ["URINATION"])
    assert (queue["review_decision"] == "").all()
    assert (queue["sampling_source"] == "random").any(), "a control group must be present"
    assert manifest["fractions"]["random_control"] >= 0.1


# ---------------------------------------------------------------- compatibility
def test_legacy_module_alias_resolves() -> None:
    import cowmata  # noqa: F401
    import cattle_imu.features as legacy

    assert legacy.SAMPLE_RATE_HZ == 50.0
    assert legacy.FEATURE_VERSION == FEATURE_VERSION


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"FAIL {test.__name__}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
