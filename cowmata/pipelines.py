"""Batch pipelines: raw JSON -> cache -> feature table -> GBDT bundle.

These were three standalone scripts in 20260818, each with its own argument
parser, its own copy of the channel list and its own idea of what a model
bundle contains.  They are library functions here for three reasons.

**The bundle now carries its own contract.**  ``train_gbdt`` writes
``feature_version`` and per-event ``thresholds`` into the joblib file.  The
20260818 trainer wrote neither, so inference fell back to a hard-coded 0.5 on
every event and would happily have scored the deployed model with a different
feature definition after the next feature change.  Thresholds are chosen on a
cow-disjoint validation split, never on the rows the boosters were fitted to.

**Cache writing goes through schema 2.**  Raw int16 counts plus a calibration
record, instead of thirteen float32 columns of which three were recomputable
norms.  See :mod:`cowmata.cache`.

**The dense label frame is written next to the supervised one.**  Dense
training needs every 2 Hz grid point with its mask, not only the supervised
subset; writing both at cache time means the deep branch never re-derives
labels with a slightly different rule than the GBDT branch.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .cache import Calibration, open_cache, write_cache_v2
from .features import (
    FEATURE_VERSION,
    estimate_reference,
    feature_columns,
    segment_features,
)
from .io import load_v2_json
from .labels import BODY_CODES, EVENT_CODES
from .metrics import choose_threshold
from .preprocessing import (
    DEFAULT_TAIL_RAISED_POLICY,
    cache_key,
    calibration_from_manifest,
    coverage_summary,
    dense_label_frame,
    resample_session,
    supervised_sample_frame,
)

IDENTITY_COLUMNS = (
    "sample_id",
    "cache_key",
    "cow_id",
    "device_key",
    "device_mac",
    "session_id",
    "center_index",
    "center_time_ms",
    "body_target",
    "segment_id",
    "segment_start_index",
    "segment_stop_index",
)

DECISION_HZ = 2.0


def _write_table(frame: pd.DataFrame, directory: Path, stem: str) -> Path:
    """Parquet when pyarrow exists, gzipped CSV otherwise.

    The container that produced this release has no pyarrow, so the CSV branch
    is the one that was exercised; both are read back by :func:`read_table`.
    """

    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{stem}.parquet"
    try:
        frame.to_parquet(target, index=False)
        return target
    except Exception:
        target = directory / f"{stem}.csv.gz"
        frame.to_csv(target, index=False, encoding="utf-8-sig", compression="gzip")
        return target


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8-sig")


# --------------------------------------------------------------------------
# stage 1: raw JSON -> schema 2 session cache + label frames
# --------------------------------------------------------------------------


def build_cache(
    *,
    annotations: str | Path,
    calibration_manifest: str | Path,
    output_root: str | Path,
    review_coverage: str | Path | None = None,
    tail_raised_policy: str = DEFAULT_TAIL_RAISED_POLICY,
    tail_position: str = "unknown",
    limit_sessions: int = 0,
) -> dict[str, Any]:
    """Resample every annotated session and write it as a schema 2 cache."""

    annotation_frame = pd.read_csv(annotations, encoding="utf-8-sig")
    coverage = (
        pd.read_csv(review_coverage, encoding="utf-8-sig")
        if review_coverage is not None
        else None
    )
    if coverage is not None and coverage.empty:
        coverage = None
    manifest = json.loads(Path(calibration_manifest).read_text(encoding="utf-8"))
    items = {
        (str(item["device_mac"]), str(item["session_id"])): item
        for item in manifest["sessions"]
    }

    run_dir = Path(output_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_root = run_dir / "session_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    groups = list(annotation_frame.groupby(["device_mac", "session_id"], sort=True))
    if limit_sessions:
        groups = groups[:limit_sessions]

    supervised_frames: list[pd.DataFrame] = []
    dense_frames: list[pd.DataFrame] = []
    session_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    covered_hours = {code: 0.0 for code in EVENT_CODES}

    for ordinal, ((device_mac, session_id), group) in enumerate(groups, start=1):
        key = (str(device_mac), str(session_id))
        item = items.get(key)
        if item is None:
            failures.append({**dict(zip(("device_mac", "session_id"), key)), "error": "missing calibration"})
            continue
        if not item.get("eligible_for_training", True):
            session_rows.append(
                {
                    "device_mac": device_mac,
                    "session_id": session_id,
                    "status": "excluded_sensor_qc",
                    "issues": ";".join(item.get("issues", [])),
                    "samples": 0,
                }
            )
            continue
        try:
            session = load_v2_json(item["path"])
            processed = resample_session(session)
            if processed.n_frames == 0:
                raise ValueError("no continuous run survived the gap filter")
            name = cache_key(str(device_mac), str(session_id), str(item["path"]))
            cows = sorted(v for v in group["cow_id"].dropna().astype(str).unique() if v)
            if len(cows) != 1:
                raise ValueError(f"expected exactly one normalised cow_id, got {cows}")
            devices = sorted(group["device_key"].astype(str).unique())
            if len(devices) != 1:
                raise ValueError(f"expected exactly one device_key, got {devices}")

            session_coverage = None
            if coverage is not None:
                session_coverage = coverage[
                    (coverage["device_mac"].astype(str) == str(device_mac))
                    & (coverage["session_id"].astype(str) == str(session_id))
                ]

            shared = dict(
                segments=processed.segments,
                annotations=group,
                cache_name=name,
                cow_id=cows[0],
                device_key=devices[0],
                device_mac=str(device_mac),
                session_id=str(session_id),
                review_coverage=session_coverage,
                tail_raised_policy=tail_raised_policy,
            )
            dense = dense_label_frame(**shared)
            supervised = supervised_sample_frame(**shared)

            write_cache_v2(
                cache_root / name,
                counts=processed.counts,
                segments=processed.segments,
                calibration=calibration_from_manifest(item),
                quality_flag=processed.quality_flag,
                metadata={
                    "cache_key": name,
                    "raw_path": str(item["path"]),
                    "cow_id": cows[0],
                    "device_key": devices[0],
                    "device_mac": str(device_mac),
                    "session_id": str(session_id),
                    "tail_position": str(item.get("tail_position", tail_position)),
                    "decision_hz": DECISION_HZ,
                    "tail_raised_policy": tail_raised_policy,
                    "label_contract": (
                        "annotation timestamps are mothers; the causal or centred "
                        "context is selected at training time, never baked in here"
                    ),
                },
                overwrite=True,
            )

            mask_columns = [f"mask_{code}" for code in EVENT_CODES]
            hours = coverage_summary(
                dense["center_time_ms"].to_numpy(np.int64),
                dense[mask_columns].to_numpy(np.float32),
            )
            for code, value in hours.items():
                covered_hours[code] += float(value)

            supervised_frames.append(supervised)
            dense_frames.append(dense)
            session_rows.append(
                {
                    "device_mac": device_mac,
                    "session_id": session_id,
                    "cow_id": cows[0],
                    "device_key": devices[0],
                    "cache_key": name,
                    "status": "included",
                    "issues": "",
                    "raw_frames": int(session.raw_values.shape[0]),
                    "resampled_frames": processed.n_frames,
                    "continuous_segments": len(processed.segments),
                    "hours": round(processed.n_frames / 50.0 / 3600.0, 4),
                    "samples": int(len(supervised)),
                    "dense_points": int(len(dense)),
                }
            )
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            failures.append(
                {
                    "device_mac": str(device_mac),
                    "session_id": str(session_id),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        print(f"[cache] {ordinal}/{len(groups)}", flush=True)

    samples = (
        pd.concat(supervised_frames, ignore_index=True) if supervised_frames else pd.DataFrame()
    )
    if not samples.empty:
        samples.insert(0, "sample_id", [f"SMP-{i:07d}" for i in range(1, len(samples) + 1)])
        samples.to_csv(run_dir / "samples.csv", index=False, encoding="utf-8-sig")
    dense = pd.concat(dense_frames, ignore_index=True) if dense_frames else pd.DataFrame()
    if not dense.empty:
        _write_table(dense, run_dir, "dense_labels")
    sessions = pd.DataFrame(session_rows)
    sessions.to_csv(run_dir / "sessions.csv", index=False, encoding="utf-8-sig")
    (run_dir / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    event_counts = {
        code: int(samples[f"event_{code}"].sum()) if not samples.empty else 0
        for code in EVENT_CODES
    }
    summary = {
        "run_dir": str(run_dir),
        "schema_version": 2,
        "annotation_sessions": len(groups),
        "included_sessions": int((sessions.get("status") == "included").sum()) if len(sessions) else 0,
        "excluded_sensor_qc_sessions": int(
            (sessions.get("status") == "excluded_sensor_qc").sum()
        )
        if len(sessions)
        else 0,
        "failures": len(failures),
        "supervised_points": int(len(samples)),
        "dense_points": int(len(dense)),
        "event_positive_points": event_counts,
        "supervised_hours_per_event": {k: round(v, 3) for k, v in covered_hours.items()},
        "tail_raised_policy": tail_raised_policy,
        "gap_policy": "split at dt<=0 or dt>40 ms; each run resampled independently to 50 Hz",
        "negative_policy": (
            "event negatives are supervised only inside exhaustive review_coverage ranges"
            if coverage is not None
            else "LEGACY PROXY: no review_coverage supplied, so no false-alarm rate may be "
            "quoted from this cache; negatives come from body-state intervals only"
        ),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


# --------------------------------------------------------------------------
# stage 2: cache + label frame -> hand-crafted feature table
# --------------------------------------------------------------------------


def _session_features(
    key: str,
    group: pd.DataFrame,
    cache_root: Path,
    *,
    causal: bool,
    calibrate: bool,
    reference_stride: int,
    feature_version: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache = open_cache(cache_root / key)
    signal = cache.physical()
    segments = [
        {"start_index": int(s.start_index), "stop_index": int(s.stop_index)}
        for s in cache.segments
    ] or [{"start_index": 0, "stop_index": int(signal.shape[0])}]

    reference = None
    if calibrate:
        blocks = [
            np.asarray(signal[s["start_index"] : s["stop_index"]], dtype=np.float64)
            for s in segments
            if s["stop_index"] - s["start_index"] >= 32
        ]
        reference = (
            estimate_reference(blocks, causal=causal, stride=reference_stride)
            if blocks
            else None
        )

    centers = group["center_index"].to_numpy(np.int64)
    owner = np.full(centers.size, -1, dtype=np.int64)
    for index, segment in enumerate(segments):
        inside = (centers >= segment["start_index"]) & (centers < segment["stop_index"])
        owner[inside] = index
    if np.any(owner < 0):
        raise ValueError(
            f"{key}: {int((owner < 0).sum())} label centres fall outside every segment"
        )

    frames: list[pd.DataFrame] = []
    for index, segment in enumerate(segments):
        selected = owner == index
        if not np.any(selected):
            continue
        start, stop = segment["start_index"], segment["stop_index"]
        block = np.asarray(signal[start:stop], dtype=np.float32)
        local = centers[selected] - start
        features = segment_features(
            block,
            local,
            causal=causal,
            reference=reference,
            feature_version=feature_version,
        )
        identity = group.loc[selected].reset_index(drop=True)
        identity = identity.assign(segment_start_index=start, segment_stop_index=stop)
        frames.append(pd.concat([identity, features.reset_index(drop=True)], axis=1))

    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    meta = {
        "cache_key": key,
        "segments": len(segments),
        "rows": int(len(table)),
        "calibration": reference.to_dict() if reference is not None else None,
    }
    return table, meta


def _feature_worker(payload: tuple) -> tuple[pd.DataFrame, dict[str, Any]]:
    key, records, root, causal, calibrate, stride, version = payload
    return _session_features(
        key,
        pd.DataFrame(records),
        Path(root),
        causal=causal,
        calibrate=calibrate,
        reference_stride=stride,
        feature_version=version,
    )


def build_features(
    *,
    samples: str | Path,
    session_cache: str | Path,
    out: str | Path,
    causal: bool = True,
    calibrate: bool = True,
    reference_stride: int = 10,
    workers: int = 1,
    feature_version: int = FEATURE_VERSION,
    limit_sessions: int = 0,
) -> dict[str, Any]:
    """Compute the tabular features for every labelled centre in ``samples``."""

    frame = read_table(samples)
    keep = [c for c in frame.columns if c in IDENTITY_COLUMNS]
    keep += [c for c in frame.columns if c.startswith(("event_", "mask_"))]
    frame = frame[keep]

    groups = list(frame.groupby("cache_key", sort=True))
    if limit_sessions:
        groups = groups[:limit_sessions]

    started = time.perf_counter()
    tables: list[pd.DataFrame] = []
    catalog: list[dict[str, Any]] = []
    root = Path(session_cache)

    if workers > 1:
        payloads = [
            (key, group.to_dict("list"), str(root), causal, calibrate, reference_stride, feature_version)
            for key, group in groups
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_feature_worker, p): p[0] for p in payloads}
            for done, future in enumerate(as_completed(futures), start=1):
                table, meta = future.result()
                tables.append(table)
                catalog.append(meta)
                if done % 20 == 0:
                    print(f"[features] {done}/{len(futures)} sessions", flush=True)
    else:
        for done, (key, group) in enumerate(groups, start=1):
            table, meta = _session_features(
                key,
                group.reset_index(drop=True),
                root,
                causal=causal,
                calibrate=calibrate,
                reference_stride=reference_stride,
                feature_version=feature_version,
            )
            tables.append(table)
            catalog.append(meta)
            if done % 20 == 0:
                print(f"[features] {done}/{len(groups)} sessions", flush=True)

    populated = [t for t in tables if len(t)]
    if not populated:
        raise ValueError("no session produced any feature row; check samples.csv and the cache root")
    table = pd.concat(populated, ignore_index=True)
    out_dir = Path(out)
    target = _write_table(table, out_dir, "feature_table")
    manifest = {
        "path": str(target),
        "rows": int(len(table)),
        "columns": int(table.shape[1]),
        "feature_columns": len(feature_columns(table)),
        "feature_version": int(feature_version),
        "window_mode": "causal" if causal else "offline_centered",
        "calibration": bool(calibrate),
        "sessions": catalog,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    (out_dir / "feature_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


# --------------------------------------------------------------------------
# stage 3: feature table -> GBDT bundle with thresholds
# --------------------------------------------------------------------------


def _validation_cows(cows: Iterable[str], fraction: float, seed: int) -> set[str]:
    ordered = sorted({str(c) for c in cows})
    if len(ordered) < 2:
        return set()
    count = max(1, int(round(len(ordered) * fraction)))
    count = min(count, len(ordered) - 1)
    rng = np.random.default_rng(seed)
    return set(rng.permutation(ordered)[:count].tolist())


def train_gbdt(
    *,
    feature_table: str | Path,
    out: str | Path | None = None,
    backend: str = "xgboost",
    device: str = "cuda",
    n_estimators: int = 400,
    feature_version: int = FEATURE_VERSION,  # what the table was built with
    validation_fraction: float = 0.25,
    seed: int = 20260819,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Fit one binary booster per task and record the thresholds it needs.

    The 20260818 trainer fitted on every labelled row and stopped there.  Two
    things are added: a cow-disjoint validation split, and the per-event
    threshold chosen on that split, stored in the bundle so inference stops
    falling back to 0.5.  An event with no validation evidence keeps 0.5 and
    says so in ``threshold_source``, so a default is never mistaken for a
    tuned value.
    """

    import joblib

    from .gbdt import BinaryBooster, BoosterConfig

    table = read_table(feature_table)
    features = feature_columns(table)
    matrix = np.nan_to_num(
        table[features].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    body = table["body_target"].to_numpy(np.int64)
    labelled = (body >= 0).astype(np.float32)

    cows = table["cow_id"].astype(str).to_numpy()
    holdout = _validation_cows(cows, validation_fraction, seed)
    is_validation = np.isin(cows, list(holdout)) if holdout else np.zeros(len(table), bool)
    is_train = ~is_validation

    tasks: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "POSTURE_LYING": ((body == BODY_CODES.index("LYING")).astype(np.int8), labelled),
        "WALKING": ((body == BODY_CODES.index("WALKING")).astype(np.int8), labelled),
    }
    for code in EVENT_CODES:
        tasks[code] = (
            table[f"event_{code}"].to_numpy(np.int8),
            table[f"mask_{code}"].to_numpy(np.float32),
        )

    root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
    output_dir = Path(out) if out else root / "runs" / "full_gbdt" / datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    models: dict[str, BinaryBooster] = {}
    thresholds: dict[str, float] = {}
    summary: dict[str, dict[str, Any]] = {}

    for task, (target, mask) in tasks.items():
        rows = mask > 0
        train_rows = rows & is_train
        valid_rows = rows & is_validation
        positives = int(target[train_rows].sum())
        if int(train_rows.sum()) < 50 or positives < 3:
            summary[task] = {"status": "skipped", "train_positives": positives}
            print(f"{task:16s} skipped: {positives} positive training rows")
            continue
        booster = BinaryBooster(
            BoosterConfig(device=device, n_estimators=n_estimators, random_state=seed),
            backend=backend,
        )
        booster.fit(matrix[train_rows], target[train_rows])
        models[task] = booster

        # No physiological rate ceiling is applied here on purpose.  The
        # ceiling in cowmata.labels is a rate per wall-clock hour, and these
        # rows are the *supervised subset*, which is enriched in positives by
        # roughly two orders of magnitude relative to a real day.  Constraining
        # the predicted-positive fraction of that subset would push every
        # threshold towards 1.0 for a reason that has nothing to do with
        # physiology.  The ceiling is enforced where the denominator is real
        # time instead: cowmata.metrics.rate_plausibility, on dense predictions.
        if int(valid_rows.sum()) >= 50 and int(target[valid_rows].sum()) >= 3:
            probability = booster.predict_proba(matrix[valid_rows])
            threshold = choose_threshold(
                target[valid_rows],
                probability,
                np.ones(int(valid_rows.sum()), dtype=bool),
            )
            source = "validation_cows"
        else:
            threshold = 0.5
            source = "default_no_validation_evidence"
        thresholds[task] = float(threshold)
        summary[task] = {
            "status": "trained",
            "train_positives": positives,
            "train_negatives": int(train_rows.sum() - positives),
            "validation_positives": int(target[valid_rows].sum()),
            "threshold": round(float(threshold), 6),
            "threshold_source": source,
        }
        print(f"{task:16s} trained: {positives} positives, threshold {threshold:.3f} ({source})")

    model_path = output_dir / "gbdt_full.joblib"
    joblib.dump(
        {
            "models": models,
            "features": features,
            "thresholds": thresholds,
            "feature_version": int(feature_version),
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "validation_cows": sorted(holdout),
        },
        model_path,
    )
    report = {
        "feature_table": str(feature_table),
        "backend": backend,
        "device": device,
        "feature_version": int(feature_version),
        "n_features": len(features),
        "validation_cows": sorted(holdout),
        "tasks": summary,
        "output": str(model_path),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
