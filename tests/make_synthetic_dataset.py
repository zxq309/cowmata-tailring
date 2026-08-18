# -*- coding: utf-8 -*-
"""Generate a small synthetic dataset with the exact 20260819 on-disk layout.

The signal is deliberately simple and physically motivated: a tail-raise plateau
during URINATION, a gravity flip during LYING_DOWN / STANDING_UP, elevated gyro
energy during WALKING, and a short impact burst for MOUNTED_BY.  Each animal is
given a *different random mounting rotation and a different lever arm*, so a
pipeline that has lost its orientation or amplitude calibration fails on this
dataset instead of passing and failing later on real cows.

Usage::

    python tests/make_synthetic_dataset.py /tmp/synthetic
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cowmata.cache import Calibration, Segment, write_cache_v2  # noqa: E402
from cowmata.labels import EVENT_CODES, STATE_ANNOTATION_CODES  # noqa: E402

SAMPLE_RATE = 50
ACC_DIVISOR = 4096.0
GYRO_DIVISOR = 32.0
MAG_DIVISOR = 1000.0


def mount_rotation(rng: np.random.Generator) -> np.ndarray:
    """A random but per-animal-fixed mounting orientation, full yaw range."""

    yaw = rng.uniform(-np.pi, np.pi)
    tilt = np.deg2rad(rng.uniform(-40, 40))
    roll = np.deg2rad(rng.uniform(-40, 40))
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(tilt), np.sin(tilt)
    cx, sx = np.cos(roll), np.sin(roll)
    rz = np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1.0]])
    ry = np.asarray([[cy, 0, sy], [0, 1.0, 0], [-sy, 0, cy]])
    rx = np.asarray([[1.0, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return rz @ ry @ rx


def build_segment(
    rng: np.random.Generator, rotation: np.ndarray, lever: float, minutes: int
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    n = minutes * 60 * SAMPLE_RATE
    labels: list[tuple[str, int, int]] = []
    gravity = np.tile(np.asarray([0.0, 0.0, 1.0]), (n, 1))
    gyro = rng.normal(0.0, 0.8, (n, 3))

    lie_start, lie_stop = int(n * 0.40), int(n * 0.60)
    gravity[lie_start:lie_stop] = np.asarray([0.0, 0.90, 0.44])
    labels.append(("LYING", lie_start, lie_stop))
    labels.append(("LYING_DOWN", lie_start - 5 * SAMPLE_RATE, lie_start + 5 * SAMPLE_RATE))
    labels.append(("STANDING_UP", lie_stop - 5 * SAMPLE_RATE, lie_stop + 5 * SAMPLE_RATE))
    gyro[max(0, lie_start - 3 * SAMPLE_RATE) : lie_start + 3 * SAMPLE_RATE] *= 6.0
    gyro[max(0, lie_stop - 3 * SAMPLE_RATE) : lie_stop + 3 * SAMPLE_RATE] *= 6.0

    for fraction in (0.10, 0.75):
        start, stop = int(n * fraction), int(n * fraction) + 40 * SAMPLE_RATE
        if stop < n:
            phase = np.arange(stop - start) / SAMPLE_RATE
            gyro[start:stop] += 12.0 * np.sin(2 * np.pi * 1.8 * phase)[:, None]
            labels.append(("WALKING", start, stop))

    for fraction in (0.25, 0.85):
        start, stop = int(n * fraction), int(n * fraction) + 35 * SAMPLE_RATE
        if stop < n:
            gravity[start:stop] = np.asarray([0.72, 0.0, 0.69])
            labels.append(("URINATION", start, stop))

    start, stop = int(n * 0.55), int(n * 0.55) + 18 * SAMPLE_RATE
    if stop < n:
        gravity[start:stop] = np.asarray([0.62, 0.10, 0.78])
        gyro[max(0, stop - 2 * SAMPLE_RATE) : stop] *= 8.0
        labels.append(("DEFECATION", start, stop))

    start, stop = int(n * 0.66), int(n * 0.66) + 20 * SAMPLE_RATE
    if stop < n:
        gravity[start:stop] = np.asarray([0.70, -0.05, 0.71])
        labels.append(("TAIL_RAISED", start, stop))

    # Being mounted: a 3 s impact with a large vertical transient.
    start, stop = int(n * 0.32), int(n * 0.32) + 3 * SAMPLE_RATE
    if stop < n:
        gravity[start:stop] += np.asarray([0.0, 0.0, 0.55])
        gyro[start:stop] *= 25.0
        labels.append(("MOUNTED_BY", start, stop))

    acc = (gravity + rng.normal(0.0, 0.02, (n, 3))) @ rotation.T
    spun = (gyro * lever) @ rotation.T
    mag = rng.normal(0.0, 0.1, (n, 3)) @ rotation.T
    counts = np.column_stack(
        (acc * ACC_DIVISOR, spun * GYRO_DIVISOR, mag * MAG_DIVISOR)
    )
    counts = np.clip(counts, np.iinfo(np.int16).min, np.iinfo(np.int16).max)
    return np.rint(counts).astype(np.int16), labels


def make_dataset(
    root: str | Path,
    *,
    cows: int = 4,
    sessions_per_cow: int = 2,
    minutes: int = 10,
    segments_per_session: int = 2,
    seed: int = 7,
) -> dict[str, object]:
    root = Path(root)
    cache_root = root / "supervised_cache" / "session_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    annotation_rows: list[dict[str, object]] = []
    session_rows: list[dict[str, object]] = []

    for cow_index in range(cows):
        cow_id = f"cow-{cow_index + 1}"
        device_mac = f"MAC{cow_index:04d}"
        rotation = mount_rotation(rng)
        lever = float(rng.uniform(0.5, 2.5))  # tail root vs mid-tail
        for session_index in range(sessions_per_cow):
            session_id = f"2026-08-{session_index + 1:02d} 10_00_00"
            cache_key = f"{device_mac}_{session_index}"
            blocks: list[np.ndarray] = []
            segments: list[Segment] = []
            offset = 0
            time_offset = 0
            for segment_index in range(segments_per_session):
                counts, labels = build_segment(rng, rotation, lever, minutes)
                blocks.append(counts)
                segments.append(
                    Segment(
                        segment_index,
                        offset,
                        offset + counts.shape[0],
                        time_offset,
                        time_offset + (counts.shape[0] - 1) * 20,
                    )
                )
                for code, start, stop in labels:
                    start = max(0, start)
                    stop = min(counts.shape[0], stop)
                    if stop <= start:
                        continue
                    annotation_rows.append(
                        {
                            "device_mac": device_mac,
                            "device_key": device_mac,
                            "session_id": session_id,
                            "cow_id": cow_id,
                            "code": code,
                            "t_start_rel_ms": int(time_offset + start * 20),
                            "t_end_rel_ms": int(time_offset + stop * 20),
                            "sensor_eligible": True,
                        }
                    )
                offset += counts.shape[0]
                time_offset += counts.shape[0] * 20 + 3 * 3600 * 1000
            array = np.concatenate(blocks, axis=0)
            write_cache_v2(
                cache_root / cache_key,
                counts=array,
                segments=segments,
                calibration=Calibration(ACC_DIVISOR, GYRO_DIVISOR, MAG_DIVISOR),
                quality_flag=np.zeros(array.shape[0], dtype=np.float32),
                metadata={
                    "cache_key": cache_key,
                    "cow_id": cow_id,
                    "device_mac": device_mac,
                    "session_id": session_id,
                    "tail_position": ("root", "mid", "tip")[cow_index % 3],
                },
                overwrite=True,
            )
            session_rows.append(
                {
                    "device_mac": device_mac,
                    "device_key": device_mac,
                    "session_id": session_id,
                    "cow_id": cow_id,
                    "cache_key": cache_key,
                    "frames": int(array.shape[0]),
                    "status": "included",
                }
            )

    annotations = pd.DataFrame(annotation_rows)
    annotations.insert(0, "event_id", [f"EVT-{i:05d}" for i in range(1, len(annotations) + 1)])
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    annotations.to_csv(
        root / "annotations" / "annotations_adjudicated_minimal.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sessions = pd.DataFrame(session_rows)
    sessions.to_csv(root / "supervised_cache" / "sessions.csv", index=False, encoding="utf-8-sig")
    return {
        "root": str(root),
        "cows": cows,
        "sessions": len(sessions),
        "annotations": len(annotations),
        "codes": sorted(set(annotations["code"])),
        "known_codes": sorted(set(EVENT_CODES) | set(STATE_ANNOTATION_CODES)),
    }


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_data")
    print(make_dataset(target))
