# -*- coding: utf-8 -*-
"""Build the hand-crafted feature table used by the non-deep branch.

For every label centre in ``samples.csv`` this computes ~100 features:
multi-scale tilt / motion-intensity / band-energy statistics plus posture
transition geometry.  Everything is computed *inside* the enclosing contiguous
segment and, when ``--offline`` is used, with zero-phase filters and centred
windows.

Per-session self-calibration is on by default: the dominant resting gravity
direction of the session defines a body frame, so a 20 degree tail lift looks
the same on every animal regardless of how the ring happened to be mounted.
This is a software step applied to data you already have; no hardware action
is required.

Usage
-----
    python scripts/build_feature_table.py \
        --samples datasets/cowmata_imu/supervised_cache/samples.csv \
        --session-cache datasets/cowmata_imu/supervised_cache/session_cache \
        --out runs/feature_table \
        --offline --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cattle_imu.features import (  # noqa: E402
    gravity_split,
    segment_features,
    session_reference,
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


def load_segments(cache_root: Path, cache_key: str, length: int) -> list[dict[str, int]]:
    meta = cache_root / cache_key / "metadata.json"
    if meta.exists():
        with meta.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        segments = payload.get("segments") or []
        if segments:
            return [
                {"start_index": int(s["start_index"]), "stop_index": int(s["stop_index"])}
                for s in segments
            ]
    return [{"start_index": 0, "stop_index": int(length)}]


def process_session(
    cache_key: str,
    group: pd.DataFrame,
    cache_root: Path,
    causal: bool,
    calibrate: bool,
    reference_stride: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    array = np.load(cache_root / cache_key / "features.npy", mmap_mode="r")
    segments = load_segments(cache_root, cache_key, len(array))

    reference = None
    if calibrate:
        static_parts: list[np.ndarray] = []
        motion_parts: list[np.ndarray] = []
        for segment in segments:
            block = np.asarray(
                array[segment["start_index"] : segment["stop_index"], 0:3], dtype=np.float64
            )
            if block.shape[0] < 32:
                continue
            static, dynamic = gravity_split(block, causal=causal)
            static_parts.append(static[::reference_stride])
            motion_parts.append(np.linalg.norm(dynamic, axis=1)[::reference_stride])
        if static_parts:
            reference = session_reference(
                np.concatenate(static_parts, axis=0), np.concatenate(motion_parts)
            )

    centers = group["center_index"].to_numpy(np.int64)
    owner = np.full(centers.size, -1, dtype=np.int64)
    for index, segment in enumerate(segments):
        inside = (centers >= segment["start_index"]) & (centers < segment["stop_index"])
        owner[inside] = index
    if np.any(owner < 0):
        raise ValueError(f"{cache_key}: {int((owner < 0).sum())} centres fall outside every segment")

    frames: list[pd.DataFrame] = []
    for index, segment in enumerate(segments):
        selected = owner == index
        if not np.any(selected):
            continue
        start = segment["start_index"]
        stop = segment["stop_index"]
        block = np.asarray(array[start:stop], dtype=np.float32)
        local = centers[selected] - start
        features = segment_features(block, local, causal=causal, reference=reference)
        identity = group.loc[selected].reset_index(drop=True)
        identity = identity.assign(
            segment_start_index=start,
            segment_stop_index=stop,
        )
        frames.append(pd.concat([identity, features.reset_index(drop=True)], axis=1))

    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    meta = {
        "cache_key": cache_key,
        "segments": len(segments),
        "rows": int(len(table)),
        "calibration": reference.to_dict() if reference is not None else None,
    }
    return table, meta


def _worker(payload: tuple) -> tuple[pd.DataFrame, dict[str, object]]:
    cache_key, records, cache_root, causal, calibrate, stride = payload
    group = pd.DataFrame(records)
    return process_session(cache_key, group, Path(cache_root), causal, calibrate, stride)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hand-crafted feature table")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--session-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reference-stride", type=int, default=10)
    parser.add_argument("--no-calibration", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="zero-phase filters, centred windows")
    mode.add_argument("--causal", action="store_true", help="one-sided filters, trailing windows")
    parser.add_argument("--limit-sessions", type=int, default=0, help="debug: only N sessions")
    args = parser.parse_args()

    causal = bool(args.causal) or not bool(args.offline)
    calibrate = not args.no_calibration
    samples = pd.read_csv(args.samples, encoding="utf-8-sig")
    keep = [column for column in samples.columns if column in IDENTITY_COLUMNS]
    keep += [c for c in samples.columns if c.startswith("event_") or c.startswith("mask_")]
    samples = samples[keep]

    groups = list(samples.groupby("cache_key", sort=True))
    if args.limit_sessions:
        groups = groups[: args.limit_sessions]
    print(f"[info] sessions={len(groups)} rows={len(samples)} causal={causal} calibrate={calibrate}")

    started = time.perf_counter()
    tables: list[pd.DataFrame] = []
    catalog: list[dict[str, object]] = []
    if args.workers > 1:
        payloads = [
            (key, group.to_dict("list"), str(args.session_cache), causal, calibrate, args.reference_stride)
            for key, group in groups
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, payload): payload[0] for payload in payloads}
            for done, future in enumerate(as_completed(futures), start=1):
                table, meta = future.result()
                tables.append(table)
                catalog.append(meta)
                if done % 20 == 0:
                    print(f"[info] {done}/{len(futures)} sessions", flush=True)
    else:
        for done, (key, group) in enumerate(groups, start=1):
            table, meta = process_session(
                key, group.reset_index(drop=True), args.session_cache, causal, calibrate, args.reference_stride
            )
            tables.append(table)
            catalog.append(meta)
            if done % 20 == 0:
                print(f"[info] {done}/{len(groups)} sessions", flush=True)

    table = pd.concat([t for t in tables if len(t)], ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "feature_table.parquet"
    try:
        table.to_parquet(target, index=False)
    except Exception as error:  # pyarrow missing
        target = args.out / "feature_table.csv.gz"
        table.to_csv(target, index=False, encoding="utf-8-sig", compression="gzip")
        print(f"[warn] parquet unavailable ({error}); wrote {target.name} instead")
    manifest = {
        "rows": int(len(table)),
        "columns": int(table.shape[1]),
        "window_mode": "causal" if causal else "offline_centered",
        "calibration": calibrate,
        "sessions": catalog,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    with (args.out / "feature_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"[done] {target} rows={len(table)} cols={table.shape[1]} in {manifest['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
