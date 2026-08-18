# -*- coding: utf-8 -*-
"""Generate a small synthetic dataset with the exact on-disk layout.

Used by ``tests/test_v2_contracts.py`` and by anyone who wants to smoke-test
the pipeline without touching the real data.  The signal is deliberately
simple: a tail-raise plateau during URINATION, a gravity flip during
LYING_DOWN / STANDING_UP, elevated gyro energy during WALKING.  A correct
pipeline must reach high AP on it; a broken one will not.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_RATE = 50
EVENT_CODES = (
    "STANDING_UP",
    "LYING_DOWN",
    "URINATION",
    "DEFECATION",
    "TAIL_RAISED",
    "TAIL_WAGGING",
)
BODY_CODES = ("STANDING", "LYING", "WALKING", "FEEDING")


def _mount_rotation(rng: np.random.Generator) -> np.ndarray:
    """A random but fixed mounting orientation for one animal."""

    angles = np.deg2rad(rng.uniform(-60, 60, size=3))
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    rx = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def build_session(rng: np.random.Generator, rotation: np.ndarray, minutes: int, n_segments: int):
    """Return (features array, segments, annotation intervals in session ms)."""

    per_segment = minutes * 60 * SAMPLE_RATE // n_segments
    blocks: list[np.ndarray] = []
    segments: list[dict[str, int]] = []
    records: list[dict[str, object]] = []
    offset = 0
    time_offset_ms = 0
    for _ in range(n_segments):
        n = per_segment
        local: list[tuple[str, int, int]] = []
        gravity = np.tile(np.asarray([0.0, 0.0, 1.0]), (n, 1))
        gyro = rng.normal(0.0, 0.8, (n, 3))

        lie_start = int(n * 0.40)
        lie_stop = int(n * 0.60)
        gravity[lie_start:lie_stop] = np.asarray([0.0, 0.90, 0.44])
        local.append(("LYING", lie_start, lie_stop))
        local.append(("LYING_DOWN", lie_start - 5 * SAMPLE_RATE, lie_start + 5 * SAMPLE_RATE))
        local.append(("STANDING_UP", lie_stop - 5 * SAMPLE_RATE, lie_stop + 5 * SAMPLE_RATE))
        gyro[max(0, lie_start - 3 * SAMPLE_RATE) : lie_start + 3 * SAMPLE_RATE] *= 6.0
        gyro[max(0, lie_stop - 3 * SAMPLE_RATE) : lie_stop + 3 * SAMPLE_RATE] *= 6.0

        for fraction in (0.10, 0.75):
            start = int(n * fraction)
            stop = start + 40 * SAMPLE_RATE
            if stop >= n:
                continue
            phase = np.arange(stop - start) / SAMPLE_RATE
            gyro[start:stop] += 12.0 * np.sin(2 * np.pi * 1.8 * phase)[:, None]
            local.append(("WALKING", start, stop))

        for fraction in (0.25, 0.85):
            start = int(n * fraction)
            stop = start + 35 * SAMPLE_RATE
            if stop >= n:
                continue
            gravity[start:stop] = np.asarray([0.72, 0.0, 0.69])
            local.append(("URINATION", start, stop))

        start = int(n * 0.55)
        stop = start + 18 * SAMPLE_RATE
        if stop < n:
            gravity[start:stop] = np.asarray([0.62, 0.10, 0.78])
            gyro[max(0, stop - 2 * SAMPLE_RATE) : stop] *= 8.0
            local.append(("DEFECATION", start, stop))

        start = int(n * 0.66)
        stop = start + 20 * SAMPLE_RATE
        if stop < n:
            gravity[start:stop] = np.asarray([0.70, -0.05, 0.71])
            local.append(("TAIL_RAISED", start, stop))

        acc = (gravity + rng.normal(0.0, 0.02, (n, 3))) @ rotation.T
        spun = gyro @ rotation.T
        mag = rng.normal(0.0, 0.1, (n, 3))
        block = np.zeros((n, 13), dtype=np.float32)
        block[:, 0:3] = acc
        block[:, 3:6] = spun
        block[:, 6:9] = mag
        block[:, 9] = np.linalg.norm(acc, axis=1)
        block[:, 10] = np.linalg.norm(spun, axis=1)
        block[:, 11] = np.linalg.norm(mag, axis=1)
        blocks.append(block)
        segments.append({"segment_id": len(segments), "start_index": offset, "stop_index": offset + n})

        for code, start, stop in local:
            start = max(0, start)
            stop = min(n, stop)
            if stop <= start:
                continue
            records.append(
                {
                    "code": code,
                    "t_start_rel_ms": int(time_offset_ms + start * 1000 / SAMPLE_RATE),
                    "t_end_rel_ms": int(time_offset_ms + stop * 1000 / SAMPLE_RATE),
                }
            )
        offset += n
        time_offset_ms += int(n * 1000 / SAMPLE_RATE) + 3 * 3600 * 1000
    return np.concatenate(blocks, axis=0), segments, records


def make_dataset(root: Path, *, cows: int = 4, sessions_per_cow: int = 2, minutes: int = 20, seed: int = 7):
    rng = np.random.default_rng(seed)
    cache_root = root / "supervised_cache" / "session_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    sample_rows: list[pd.DataFrame] = []
    annotation_rows: list[dict[str, object]] = []

    for cow_index in range(cows):
        cow_id = f"cow-{cow_index + 1}"
        device_mac = f"MAC{cow_index:04d}"
        rotation = _mount_rotation(rng)
        for session_index in range(sessions_per_cow):
            session_id = f"2026-08-{session_index + 1:02d} 10_00_00"
            cache_key = f"{device_mac}_{session_index}"
            n_segments = 1 + (session_index + cow_index) % 3
            array, segments, session_records = build_session(rng, rotation, minutes, n_segments)
            directory = cache_root / cache_key
            directory.mkdir(parents=True, exist_ok=True)
            np.save(directory / "features.npy", array)
            times = np.zeros(len(array), dtype=np.int64)
            cursor = 0
            for index, segment in enumerate(segments):
                length = segment["stop_index"] - segment["start_index"]
                times[segment["start_index"] : segment["stop_index"]] = (
                    cursor + np.arange(length, dtype=np.int64) * 20
                )
                cursor += length * 20 + 3 * 3600 * 1000
            np.save(directory / "times_ms.npy", times)
            with (directory / "metadata.json").open("w", encoding="utf-8") as handle:
                json.dump({"segments": segments}, handle)

            for record in session_records:
                annotation_rows.append(
                    {
                        "device_mac": device_mac,
                        "session_id": session_id,
                        "cow_id": cow_id,
                        "sensor_eligible": True,
                        **record,
                    }
                )

            centers = []
            for segment in segments:
                centers.append(
                    np.arange(segment["start_index"], segment["stop_index"], 25, dtype=np.int64)
                )
            centers_all = np.concatenate(centers)
            frame = pd.DataFrame(
                {
                    "cache_key": cache_key,
                    "cow_id": cow_id,
                    "device_key": device_mac,
                    "device_mac": device_mac,
                    "session_id": session_id,
                    "center_index": centers_all,
                    "center_time_ms": times[centers_all],
                }
            )
            starts = np.asarray([s["start_index"] for s in segments])
            stops = np.asarray([s["stop_index"] for s in segments])
            position = np.searchsorted(starts, centers_all, side="right") - 1
            frame["segment_id"] = position
            frame["segment_start_index"] = starts[position]
            frame["segment_stop_index"] = stops[position]
            sample_rows.append(frame)

    annotations = pd.DataFrame(annotation_rows)
    samples = pd.concat(sample_rows, ignore_index=True)

    # label the samples from the annotation table
    samples["body_target"] = -1
    for code in EVENT_CODES:
        samples[f"event_{code}"] = 0
        samples[f"mask_{code}"] = 0
    for (device, session), group in annotations.groupby(["device_mac", "session_id"]):
        rows = (samples["device_mac"] == device) & (samples["session_id"] == session)
        times = samples.loc[rows, "center_time_ms"].to_numpy(np.int64)
        body = np.full(times.size, -1, dtype=np.int64)
        events = {code: np.zeros(times.size, dtype=np.uint8) for code in EVENT_CODES}
        for record in group.itertuples():
            inside = (times >= record.t_start_rel_ms) & (times < record.t_end_rel_ms)
            if record.code in BODY_CODES:
                body[inside] = BODY_CODES.index(record.code)
            elif record.code in EVENT_CODES:
                events[record.code][inside] = 1
        body[body < 0] = 0  # everything else is STANDING here
        samples.loc[rows, "body_target"] = body
        for code in EVENT_CODES:
            samples.loc[rows, f"event_{code}"] = events[code]
            samples.loc[rows, f"mask_{code}"] = 1

    samples.insert(0, "sample_id", [f"SMP-{i:07d}" for i in range(1, len(samples) + 1)])
    (root / "supervised_cache").mkdir(parents=True, exist_ok=True)
    samples.to_csv(root / "supervised_cache" / "samples.csv", index=False, encoding="utf-8-sig")
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    annotations.to_csv(
        root / "annotations" / "annotations.csv", index=False, encoding="utf-8-sig"
    )

    # leave-one-cow-out splits in the real format
    keys = (samples["device_mac"].astype(str) + "|" + samples["session_id"].astype(str)).unique().tolist()
    by_cow: dict[str, list[str]] = {}
    for key in keys:
        cow = samples.loc[
            (samples["device_mac"].astype(str) + "|" + samples["session_id"].astype(str)) == key,
            "cow_id",
        ].iloc[0]
        by_cow.setdefault(str(cow), []).append(key)
    folds = []
    cow_ids = sorted(by_cow)
    for index, cow in enumerate(cow_ids, start=1):
        others = [c for c in cow_ids if c != cow]
        validation_cow = others[index % len(others)]
        train = [k for c in others if c != validation_cow for k in by_cow[c]]
        folds.append(
            {
                "fold": index,
                "test_cow": cow,
                "train_sessions": train,
                "validation_sessions": by_cow[validation_cow],
                "test_sessions": by_cow[cow],
                "counts": {},
            }
        )
    (root / "loco_splits").mkdir(parents=True, exist_ok=True)
    with (root / "loco_splits" / "loco_splits.json").open("w", encoding="utf-8") as handle:
        json.dump({"protocol": "synthetic_loco", "folds": folds}, handle, ensure_ascii=False, indent=2)
    return {
        "samples": len(samples),
        "annotations": len(annotations),
        "sessions": len(keys),
        "cows": len(cow_ids),
    }


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_data")
    target.mkdir(parents=True, exist_ok=True)
    print(make_dataset(target))
