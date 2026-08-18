# -*- coding: utf-8 -*-
"""End-to-end contracts for the three batch stages.

These build a small synthetic dataset, run cache -> features -> GBDT ->
predict -> mine, and assert the properties that were silently broken in
20260818: a bundle that forgets its thresholds, a bundle that forgets which
feature definition it was fitted to, a dense prediction file that cannot be
grouped by animal, and threshold selection that peeks at training rows.

Runs without torch.  Uses the sklearn booster backend so it needs no xgboost.

    python tests/test_pipelines.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cowmata.cache import open_cache  # noqa: E402
from cowmata.features import FEATURE_VERSION  # noqa: E402
from cowmata.inference import COWMATA  # noqa: E402
from cowmata.labels import EVENT_CODES  # noqa: E402
from cowmata.pipelines import build_features, read_table, train_gbdt  # noqa: E402
from cowmata.preprocessing import supervised_sample_frame  # noqa: E402
from cowmata.tools import mine_candidates  # noqa: E402

from make_synthetic_dataset import make_dataset  # noqa: E402

_STATE: dict[str, object] = {}


def _dataset() -> dict[str, object]:
    """Build the synthetic corpus once and reuse it across the tests."""

    if "root" in _STATE:
        return _STATE
    root = Path(tempfile.mkdtemp(prefix="cowmata_pipeline_"))
    make_dataset(root, cows=4, sessions_per_cow=2, minutes=6, seed=11)

    annotations = pd.read_csv(
        root / "annotations" / "annotations_adjudicated_minimal.csv", encoding="utf-8-sig"
    )
    sessions = pd.read_csv(root / "supervised_cache" / "sessions.csv", encoding="utf-8-sig")
    frames = []
    for _, row in sessions.iterrows():
        cache = open_cache(root / "supervised_cache" / "session_cache" / row["cache_key"])
        group = annotations[
            (annotations["device_mac"] == row["device_mac"])
            & (annotations["session_id"] == row["session_id"])
        ]
        frames.append(
            supervised_sample_frame(
                segments=cache.segments,
                annotations=group,
                cache_name=str(row["cache_key"]),
                cow_id=str(row["cow_id"]),
                device_key=str(row["device_key"]),
                device_mac=str(row["device_mac"]),
                session_id=str(row["session_id"]),
            )
        )
    samples = pd.concat(frames, ignore_index=True)
    samples.insert(0, "sample_id", [f"SMP-{i:07d}" for i in range(1, len(samples) + 1)])
    samples.to_csv(root / "supervised_cache" / "samples.csv", index=False, encoding="utf-8-sig")

    manifest = build_features(
        samples=root / "supervised_cache" / "samples.csv",
        session_cache=root / "supervised_cache" / "session_cache",
        out=root / "feature_table",
        causal=False,
    )
    report = train_gbdt(
        feature_table=manifest["path"],
        out=root / "gbdt",
        backend="sklearn",
        device="cpu",
        n_estimators=40,
        feature_version=manifest["feature_version"],
        project_root=root,
    )
    _STATE.update(
        {
            "root": root,
            "samples": samples,
            "sessions": sessions,
            "manifest": manifest,
            "report": report,
        }
    )
    return _STATE


# ------------------------------------------------------------------ stage 2
def test_feature_table_covers_every_supervised_centre() -> None:
    state = _dataset()
    manifest = state["manifest"]
    assert manifest["rows"] == len(state["samples"]), (
        "every supervised label centre must produce exactly one feature row; "
        "a centre silently dropped here is a label silently dropped from training"
    )
    assert manifest["feature_version"] == FEATURE_VERSION
    assert manifest["window_mode"] == "offline_centered"


def test_feature_table_records_its_own_window_mode() -> None:
    """Causal and offline tables are not interchangeable and must say which they are."""

    state = _dataset()
    root = Path(state["root"])
    causal = build_features(
        samples=root / "supervised_cache" / "samples.csv",
        session_cache=root / "supervised_cache" / "session_cache",
        out=root / "feature_table_causal",
        causal=True,
        limit_sessions=1,
    )
    assert causal["window_mode"] == "causal"
    offline = read_table(state["manifest"]["path"])
    causal_table = read_table(causal["path"])
    shared = sorted(set(offline.columns) & set(causal_table.columns))
    merged = offline.merge(causal_table, on="sample_id", suffixes=("_off", "_cau"))
    assert len(merged) > 0
    differing = [
        column
        for column in shared
        if column.startswith(("tilt", "vedba", "gyro", "acc"))
        and not np.allclose(
            merged[f"{column}_off"].to_numpy(float),
            merged[f"{column}_cau"].to_numpy(float),
            equal_nan=True,
        )
    ]
    assert differing, "a causal table identical to the offline one means the flag did nothing"


# ------------------------------------------------------------------ stage 3
def test_bundle_carries_thresholds_and_feature_version() -> None:
    """The 20260818 bundle carried neither, so inference fell back to 0.5."""

    import joblib

    state = _dataset()
    bundle = joblib.load(Path(state["report"]["output"]))
    assert "thresholds" in bundle and bundle["thresholds"], "bundle has no thresholds"
    assert bundle["feature_version"] == FEATURE_VERSION
    assert set(bundle["thresholds"]) <= set(bundle["models"])
    for name, value in bundle["thresholds"].items():
        assert 0.0 <= float(value) <= 1.0, f"{name}: threshold out of range"


def test_thresholds_come_from_cows_the_booster_never_saw() -> None:
    state = _dataset()
    report = state["report"]
    holdout = set(report["validation_cows"])
    assert holdout, "a cow-disjoint validation split must exist"
    trained = {name for name, task in report["tasks"].items() if task["status"] == "trained"}
    assert trained
    tuned = {
        name
        for name in trained
        if report["tasks"][name]["threshold_source"] == "validation_cows"
    }
    assert tuned, "no threshold was chosen on held-out cows"
    table = read_table(state["manifest"]["path"])
    all_cows = set(table["cow_id"].astype(str))
    assert holdout < all_cows, "validation may not consume every cow"


def test_a_default_threshold_is_labelled_as_a_default() -> None:
    """A guess and a tuned value must never be indistinguishable in the report."""

    state = _dataset()
    for name, task in state["report"]["tasks"].items():
        if task["status"] != "trained":
            continue
        source = task["threshold_source"]
        assert source in {"validation_cows", "default_no_validation_evidence"}
        if source == "default_no_validation_evidence":
            assert task["threshold"] == 0.5


def test_an_event_with_no_positives_is_skipped_not_faked() -> None:
    state = _dataset()
    tasks = state["report"]["tasks"]
    # MOUNTING has no synthetic annotations, so it must be absent, not fitted.
    assert tasks["MOUNTING"]["status"] == "skipped"
    import joblib

    bundle = joblib.load(Path(state["report"]["output"]))
    assert "MOUNTING" not in bundle["models"]
    assert "MOUNTING" not in bundle["thresholds"]


# ------------------------------------------------------------------ inference
def test_inference_uses_the_bundle_thresholds_not_one_half() -> None:
    state = _dataset()
    root = Path(state["root"])
    model = COWMATA(state["report"]["output"], data_root=root)
    key = str(state["sessions"].iloc[0]["cache_key"])
    result = model.predict(key, project=root / "pred", causal=False)
    stored = model.bundle_thresholds
    for name, value in stored.items():
        assert abs(result.thresholds[name] - value) < 1e-12, f"{name} threshold was overridden"
    # An event the bundle never saw still needs a number; 0.5 is the documented
    # fallback, and it must apply only to that event.
    assert result.thresholds["MOUNTING"] == 0.5
    assert result.feature_version == FEATURE_VERSION


def test_an_explicit_override_beats_the_bundle() -> None:
    state = _dataset()
    root = Path(state["root"])
    model = COWMATA(state["report"]["output"], data_root=root)
    key = str(state["sessions"].iloc[0]["cache_key"])
    result = model.predict(key, project=root / "pred_override", threshold=0.8, causal=False)
    assert set(result.thresholds.values()) == {0.8}


def test_dense_predictions_can_be_grouped_by_animal() -> None:
    """Two sessions concatenated must stay separable; in 20260818 they did not."""

    state = _dataset()
    root = Path(state["root"])
    model = COWMATA(state["report"]["output"], data_root=root)
    keys = [str(k) for k in state["sessions"]["cache_key"].head(3)]
    frames = [
        model.predict(key, project=root / "pred_many", causal=False).dense for key in keys
    ]
    for column in ("cache_key", "cow_id", "device_mac", "session_id"):
        assert all(column in frame.columns for frame in frames), f"dense output lost {column}"
    combined = pd.concat(frames, ignore_index=True)
    assert combined.groupby(["device_mac", "session_id"]).ngroups == len(keys)
    assert combined["cow_id"].nunique() >= 1


def test_run_json_records_what_produced_the_numbers() -> None:
    state = _dataset()
    root = Path(state["root"])
    model = COWMATA(state["report"]["output"], data_root=root)
    key = str(state["sessions"].iloc[0]["cache_key"])
    model.predict(key, project=root / "pred_meta", causal=False)
    payload = json.loads((root / "pred_meta" / f"{key}_run.json").read_text(encoding="utf-8"))
    for field in ("cache_key", "schema_version", "feature_version", "thresholds"):
        assert field in payload, f"run manifest omits {field}"


# ------------------------------------------------------------------ mining
def test_mining_consumes_the_dense_file_without_a_manual_join() -> None:
    state = _dataset()
    root = Path(state["root"])
    model = COWMATA(state["report"]["output"], data_root=root)
    frames = [
        model.predict(str(key), project=root / "pred_mine", causal=False).dense
        for key in state["sessions"]["cache_key"].head(4)
    ]
    dense = pd.concat(frames, ignore_index=True)
    queue, manifest = mine_candidates(dense, ["URINATION", "DEFECATION"], per_event=8)
    assert len(queue) > 0
    assert (queue["review_decision"] == "").all(), "mining must never emit a label"
    assert (queue["sampling_source"] == "random").any()
    assert queue["cow_id"].nunique() >= 2, (
        "the queue must span animals; the identity columns in the dense file are "
        "what make that possible"
    )
    assert manifest["rows"] == len(queue)


# ------------------------------------------------------------------ taxonomy
def test_every_trained_event_is_a_known_event_code() -> None:
    state = _dataset()
    known = set(EVENT_CODES) | {"POSTURE_LYING", "WALKING"}
    assert set(state["report"]["tasks"]) <= known


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
    root = _STATE.get("root")
    if root is not None:
        shutil.rmtree(root, ignore_errors=True)
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
