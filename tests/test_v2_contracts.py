# -*- coding: utf-8 -*-
"""Contract tests for the v2 patch.

Runnable two ways::

    python tests/test_v2_contracts.py     # no pytest needed
    pytest tests/test_v2_contracts.py

Tests that need torch are skipped automatically when torch is absent; every
other test runs on numpy/pandas/scikit-learn only.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cattle_imu.annotations import EVENT_CODES  # noqa: E402
from cattle_imu.features import (  # noqa: E402
    gravity_split,
    segment_features,
    session_reference,
    window_stats,
)
from cattle_imu.fusion import apply_fusion, fit_fusion_weights  # noqa: E402
from cattle_imu.metrics import (  # noqa: E402
    choose_threshold,
    event_level_metrics,
    postprocess_intervals,
    rate_plausibility,
    selection_score,
    truth_intervals_from_annotations,
)
from cattle_imu.preprocessing import label_centers  # noqa: E402
from cattle_imu.windowing import context_bounds, context_bounds_batch  # noqa: E402


# ---------------------------------------------------------------- windowing
def test_causal_window_never_crosses_a_segment() -> None:
    for center in range(200, 260):
        start, stop, dest = context_bounds(center, 200, 500, 2048, "causal")
        assert start >= 200, "context reached into the previous segment"
        assert stop == center + 1, "causal context must end at the label"
        assert dest + (stop - start) == 2048, "window must be right-aligned"


def test_centered_window_stays_inside_its_segment() -> None:
    for center in (200, 210, 350, 499):
        start, stop, dest = context_bounds(center, 200, 500, 64, "centered")
        assert 200 <= start < stop <= 500
        assert 0 <= dest <= 64 - (stop - start)


def test_causal_window_reads_no_future_sample() -> None:
    start, stop, _ = context_bounds(300, 0, 1000, 128, "causal")
    assert stop <= 301


def test_batch_bounds_match_scalar_bounds() -> None:
    centers = np.arange(200, 260)
    starts = np.full(centers.size, 200)
    stops = np.full(centers.size, 500)
    batch = context_bounds_batch(centers, starts, stops, 2048, "causal")
    for index, center in enumerate(centers):
        scalar = context_bounds(int(center), 200, 500, 2048, "causal")
        assert (batch[0][index], batch[1][index], batch[2][index]) == scalar


# ---------------------------------------------------------------- labels
def _annotation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(device_mac="A", session_id="S", code="STANDING", t_start_rel_ms=0, t_end_rel_ms=60000),
            dict(device_mac="A", session_id="S", code="URINATION", t_start_rel_ms=10000, t_end_rel_ms=30000),
            dict(device_mac="A", session_id="S", code="TAIL_RAISED", t_start_rel_ms=40000, t_end_rel_ms=45000),
        ]
    )


def test_tail_raised_policies() -> None:
    times = np.arange(0, 60000, 500, dtype=np.int64)
    annotations = _annotation_frame()
    tail = EVENT_CODES.index("TAIL_RAISED")
    inside = (times >= 10000) & (times < 30000)

    _, events, masks = label_centers(times, annotations, None, tail_raised_policy="legacy")
    assert events[inside, tail].sum() == 0 and masks[inside, tail].sum() > 0, (
        "legacy behaviour: urination is a confident tail-raise negative"
    )

    _, events, masks = label_centers(times, annotations, None, tail_raised_policy="derive")
    assert events[inside, tail].all(), "policy 'derive' must mark urination as tail-raised"
    assert masks[inside, tail].all()

    _, events, masks = label_centers(times, annotations, None, tail_raised_policy="exclude")
    assert events[inside, tail].sum() == 0
    assert masks[inside, tail].sum() == 0, "policy 'exclude' must unsupervise those points"


def test_body_overlap_still_raises() -> None:
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


# ---------------------------------------------------------------- metrics
def test_threshold_matches_the_reference_implementation() -> None:
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(3)
    for _ in range(5):
        n = 1500
        target = (rng.random(n) < 0.05).astype(np.int8)
        probability = np.clip(0.3 * target + rng.random(n) * 0.8, 0, 1)
        mask = np.ones(n, dtype=bool)
        candidates = np.unique(np.concatenate((np.linspace(0.05, 0.95, 37), probability)))
        reference = max(f1_score(target, probability >= t, zero_division=0) for t in candidates)
        chosen = choose_threshold(target, probability, mask)
        assert abs(f1_score(target, probability >= chosen, zero_division=0) - reference) < 1e-12


def test_threshold_is_stable_on_degenerate_input() -> None:
    assert choose_threshold(np.zeros(10, np.int8), np.random.rand(10), np.ones(10, bool)) == 0.5
    assert choose_threshold(np.ones(10, np.int8), np.random.rand(10), np.ones(10, bool)) == 0.5


def test_postprocess_merges_and_drops() -> None:
    merged = postprocess_intervals([(0, 20000), (21000, 35000)], "URINATION")
    assert merged == [(0, 35000)], "a 1 s dip must not split one urination in two"
    assert postprocess_intervals([(0, 500)], "URINATION") == [], "500 ms is not a urination"


def test_truth_intervals_come_from_annotations() -> None:
    annotations = _annotation_frame()
    intervals = truth_intervals_from_annotations(annotations, "A", "S", "URINATION")
    assert intervals == [(10000, 30000)]


def test_event_metrics_flag_insufficient_evidence() -> None:
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
    report = event_level_metrics(frame, "URINATION", 0.5, annotations=_annotation_frame())
    assert report["true_events"] == 1
    assert report["recall"] == 1.0
    assert report["precision_f1_claimable"] is False, "one cow, one event is not reportable"
    assert report["evidence_level"] == "not_evaluable"


def test_rate_plausibility_catches_the_fold2_failure() -> None:
    verdict = rate_plausibility("URINATION", 1372, 32.2)
    assert verdict["verdict"] == "implausible"
    assert rate_plausibility("URINATION", 10, 30.0)["verdict"] == "plausible"


def test_selection_score_ignores_unevaluable_events() -> None:
    score = selection_score(0.9, 0.8, {"URINATION": 0.5, "TAIL_WAGGING": None})
    assert score["events_used"] == ["URINATION"]
    assert 0.0 < float(score["selection_score"]) < 1.0


# ---------------------------------------------------------------- features
def _synthetic_segment(n: int = 50 * 300) -> np.ndarray:
    rng = np.random.default_rng(11)
    array = np.zeros((n, 13), dtype=np.float32)
    array[:, 0] = rng.normal(0.0, 0.02, n)
    array[:, 1] = 0.2 + rng.normal(0.0, 0.02, n)
    array[:, 2] = 0.97 + rng.normal(0.0, 0.02, n)
    array[:, 3:6] = rng.normal(0.0, 1.0, (n, 3))
    array[5000:6500, 0] = 0.85
    array[5000:6500, 2] = 0.50
    return array


def test_calibration_puts_the_resting_tilt_at_zero() -> None:
    array = _synthetic_segment()
    static, dynamic = gravity_split(array[:, 0:3].astype(np.float64), causal=False)
    reference = session_reference(static, np.linalg.norm(dynamic, axis=1))
    centers = np.arange(0, array.shape[0], 25)
    table = segment_features(array, centers, causal=False, reference=reference)
    tilt = table["tilt_mean_5s"].to_numpy()
    episode = (centers >= 5200) & (centers < 6300)
    assert tilt[~episode].mean() < 3.0, "resting tilt must be near zero after calibration"
    assert tilt[episode].mean() > 30.0, "the tail-raise episode must stand out"


def test_causal_features_do_not_see_the_future() -> None:
    array = _synthetic_segment()
    static, dynamic = gravity_split(array[:, 0:3].astype(np.float64), causal=True)
    reference = session_reference(static, np.linalg.norm(dynamic, axis=1))
    centers = np.arange(0, array.shape[0], 25)
    base = segment_features(array, centers, causal=True, reference=reference)
    edited = array.copy()
    rng = np.random.default_rng(5)
    edited[10000:] = rng.normal(0.0, 1.0, edited[10000:].shape).astype(np.float32)
    after = segment_features(edited, centers, causal=True, reference=reference)
    cut = int(np.searchsorted(centers, 9000))
    difference = np.nanmax(np.abs(base.iloc[:cut].to_numpy() - after.iloc[:cut].to_numpy()))
    assert difference == 0.0, "a causal feature changed when future samples changed"


def test_rolling_statistics_are_exact() -> None:
    values = np.arange(10.0)
    mean, _ = window_stats(values, -2, 1)
    assert np.allclose(mean, [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8])
    mean, std = window_stats(np.ones(5), -1, 2)
    assert np.allclose(mean, 1.0) and np.allclose(std, 0.0)


def test_features_have_no_nan() -> None:
    array = _synthetic_segment(50 * 60)
    table = segment_features(array, np.arange(0, array.shape[0], 25), causal=False)
    assert not table.isna().any().any()


# ---------------------------------------------------------------- fusion
def test_fusion_discards_a_useless_branch() -> None:
    rng = np.random.default_rng(4)
    n = 3000
    target = (rng.random(n) < 0.05).astype(np.int8)
    informative = np.clip(0.6 * target + rng.random(n) * 0.5, 0, 1)
    noise = rng.random(n)
    fitted = fit_fusion_weights({"deep": informative, "feature": noise}, target)
    assert fitted["weights"]["deep"] > fitted["weights"]["feature"]
    fused = apply_fusion({"deep": informative, "feature": noise}, fitted["weights"])
    assert fused.shape == (n,) and np.all((fused >= 0) & (fused <= 1))


# ---------------------------------------------------------------- torch part
def test_dataset_window_is_segment_safe() -> None:
    try:
        import torch  # noqa: F401
    except Exception:
        print("  [skip] torch unavailable")
        return
    from cattle_imu.dataset import WindowDataset, recompute_magnitudes

    root = Path(tempfile.mkdtemp())
    key = "sess"
    (root / key).mkdir(parents=True)
    array = np.zeros((500, 13), dtype=np.float32)
    array[:, 0] = np.arange(500)
    np.save(root / key / "features.npy", array)
    rows = []
    for center in (0, 199, 200, 260, 499):
        start, stop = (0, 200) if center < 200 else (200, 500)
        row = dict(
            sample_id=f"S{center}", cache_key=key, cow_id="c", device_mac="A", session_id="S",
            center_index=center, center_time_ms=center * 20, body_target=0,
            segment_start_index=start, segment_stop_index=stop,
        )
        for code in EVENT_CODES:
            row[f"event_{code}"] = 0
            row[f"mask_{code}"] = 1
        rows.append(row)
    samples = pd.DataFrame(rows)
    dataset = WindowDataset(
        samples, root, np.zeros(13, np.float32), np.ones(13, np.float32),
        context_samples=64, feature_mode="raw12",
    )
    raw = np.load(root / key / "features.npy", mmap_mode="r")
    for index, center in enumerate(samples["center_index"]):
        window, valid = dataset._context(raw, index)
        real = window[64 - valid :]
        assert abs(real[-1, 0] - center) < 1e-6
        assert real[0, 0] >= (0 if center < 200 else 200)

    source = np.ones((4, 13), np.float32)
    source[:, :9] *= 2.0
    recompute_magnitudes(source)
    assert np.allclose(source[0, 9:12], np.sqrt(12.0))


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"FAIL {test.__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
