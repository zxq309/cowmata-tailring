"""Gap-safe resampling and supervised-window indexing."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .annotations import BODY_CODES, EVENT_CODES
from .io import ImuSession, contiguous_slices


TARGET_HZ = 50.0
TARGET_DT_MS = 20
WINDOW_SAMPLES = 256
WINDOW_LEFT = WINDOW_SAMPLES // 2
WINDOW_RIGHT = WINDOW_SAMPLES - WINDOW_LEFT
TARGET_STRIDE_SAMPLES = 25  # 0.5 s / 2 Hz decisions
GAP_THRESHOLD_MS = 40.0


@dataclass(frozen=True)
class ProcessedSession:
    features: np.ndarray
    times_ms: np.ndarray
    segment_ids: np.ndarray
    segments: list[dict[str, int]]


def cache_key(device_mac: str, session_id: str, raw_path: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", f"{device_mac}_{session_id}").strip("_")
    suffix = hashlib.sha256(str(raw_path).encode("utf-8")).hexdigest()[:10]
    return f"{safe}_{suffix}"


def _quality_flag(raw_time: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Mark resampled points whose surrounding raw interval is not near 20 ms."""

    right = np.searchsorted(raw_time, grid, side="left")
    right = np.clip(right, 1, len(raw_time) - 1)
    local_dt = raw_time[right] - raw_time[right - 1]
    return (np.abs(local_dt - TARGET_DT_MS) > 5).astype(np.float32)


def resample_session(
    session: ImuSession,
    *,
    acc_divisor: float,
    acc_bias_counts: tuple[float, float, float] | list[float],
    gyro_divisor: float,
    gyro_bias_counts: tuple[float, float, float] | list[float],
    mag_divisor: float,
    gap_threshold_ms: float = GAP_THRESHOLD_MS,
) -> ProcessedSession:
    """Resample each continuous run independently; no interpolation crosses a gap."""

    physical = session.physical_values(
        acc_divisor=acc_divisor,
        acc_bias_counts=np.asarray(acc_bias_counts, dtype=np.float32),
        gyro_divisor=gyro_divisor,
        gyro_bias_counts=np.asarray(gyro_bias_counts, dtype=np.float32),
        mag_divisor=mag_divisor,
    )
    feature_chunks: list[np.ndarray] = []
    time_chunks: list[np.ndarray] = []
    segment_chunks: list[np.ndarray] = []
    segments: list[dict[str, int]] = []
    output_offset = 0
    segment_id = 0
    for run in contiguous_slices(session.elapsed_ms, gap_threshold_ms=gap_threshold_ms):
        raw_time = session.elapsed_ms[run].astype(np.float64)
        raw_values = physical[run]
        if raw_time.size < WINDOW_SAMPLES:
            continue
        start = int(math.ceil(raw_time[0] / TARGET_DT_MS) * TARGET_DT_MS)
        stop = int(math.floor(raw_time[-1] / TARGET_DT_MS) * TARGET_DT_MS)
        if stop - start < (WINDOW_SAMPLES - 1) * TARGET_DT_MS:
            continue
        grid = np.arange(start, stop + 1, TARGET_DT_MS, dtype=np.int64)
        channels = np.column_stack(
            [np.interp(grid, raw_time, raw_values[:, index]) for index in range(raw_values.shape[1])]
        ).astype(np.float32)
        magnitudes = np.column_stack(
            (
                np.linalg.norm(channels[:, 0:3], axis=1),
                np.linalg.norm(channels[:, 3:6], axis=1),
                np.linalg.norm(channels[:, 6:9], axis=1),
            )
        ).astype(np.float32)
        quality = _quality_flag(raw_time, grid)[:, None]
        features = np.concatenate((channels, magnitudes, quality), axis=1).astype(np.float32)
        feature_chunks.append(features)
        time_chunks.append(grid)
        segment_chunks.append(np.full(grid.size, segment_id, dtype=np.int32))
        segments.append(
            {
                "segment_id": segment_id,
                "start_index": output_offset,
                "stop_index": output_offset + grid.size,
                "start_ms": int(grid[0]),
                "stop_ms": int(grid[-1]),
            }
        )
        output_offset += grid.size
        segment_id += 1
    if not feature_chunks:
        return ProcessedSession(
            features=np.empty((0, 13), dtype=np.float32),
            times_ms=np.empty(0, dtype=np.int64),
            segment_ids=np.empty(0, dtype=np.int32),
            segments=[],
        )
    return ProcessedSession(
        features=np.concatenate(feature_chunks, axis=0),
        times_ms=np.concatenate(time_chunks),
        segment_ids=np.concatenate(segment_chunks),
        segments=segments,
    )


def candidate_centers(processed: ProcessedSession) -> np.ndarray:
    centers: list[np.ndarray] = []
    for segment in processed.segments:
        # Mother labels are independent of a later model context. Causal models
        # may left-pad at segment start and never require future samples.
        first = int(segment["start_index"])
        last_exclusive = int(segment["stop_index"])
        if last_exclusive <= first:
            continue
        centers.append(np.arange(first, last_exclusive, TARGET_STRIDE_SAMPLES, dtype=np.int64))
    return np.concatenate(centers) if centers else np.empty(0, dtype=np.int64)


def segment_bounds_for_centers(
    processed: ProcessedSession, centers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``[start, stop)`` bounds of the segment owning each centre."""

    centers = np.asarray(centers, dtype=np.int64)
    if centers.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if not processed.segments:
        raise ValueError("processed session has no segments but centres were requested")
    starts = np.asarray([int(s["start_index"]) for s in processed.segments], dtype=np.int64)
    stops = np.asarray([int(s["stop_index"]) for s in processed.segments], dtype=np.int64)
    position = np.searchsorted(starts, centers, side="right") - 1
    if np.any(position < 0):
        raise ValueError("centre index precedes the first segment")
    start = starts[position]
    stop = stops[position]
    if np.any(centers >= stop):
        raise ValueError("centre index falls in a gap between segments")
    return start, stop


def coverage_summary(
    center_times_ms: np.ndarray,
    event_masks: np.ndarray,
) -> dict[str, float]:
    """Supervised hours per event code, for the cache summary.

    A session missing from ``review_coverage`` silently loses every negative
    sample; printing the covered hours per code makes that visible instead of
    looking like a data change.
    """

    times = np.asarray(center_times_ms, dtype=np.int64)
    masks = np.asarray(event_masks)
    if times.size == 0:
        return {code: 0.0 for code in EVENT_CODES}
    step_hours = 0.5 / 3600.0
    return {
        code: float(masks[:, index].sum() * step_hours)
        for index, code in enumerate(EVENT_CODES)
    }


TAIL_RAISED_POLICIES = ("derive", "exclude", "legacy")
DEFAULT_TAIL_RAISED_POLICY = "derive"


def _apply_tail_raised_policy(
    events: np.ndarray,
    event_masks: np.ndarray,
    policy: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconcile TAIL_RAISED with the excretion codes.

    URINATION is annotated as the *whole* event including the preparatory tail
    lift, yet no URINATION or DEFECATION interval in the adjudicated table is
    also annotated TAIL_RAISED.  Under the legacy mask rule those 104 intervals
    became confident *negatives* for the tail-raise head, i.e. the model was
    told that a raised tail during urination is not a raised tail.

    ``derive``  - excretion intervals inherit TAIL_RAISED=1 (hierarchical view:
                  tail-raise is the parent state of urination / defecation).
    ``exclude`` - excretion intervals become unsupervised for TAIL_RAISED
                  (mask 0) instead of counting as negatives.
    ``legacy``  - previous behaviour, kept only for reproducing old runs.
    """

    if policy not in TAIL_RAISED_POLICIES:
        raise ValueError(f"tail_raised_policy must be one of {TAIL_RAISED_POLICIES}")
    if policy == "legacy" or "TAIL_RAISED" not in EVENT_CODES:
        return events, event_masks
    tail = EVENT_CODES.index("TAIL_RAISED")
    excretion = [
        EVENT_CODES.index(code) for code in ("URINATION", "DEFECATION") if code in EVENT_CODES
    ]
    if not excretion:
        return events, event_masks
    active = np.any(events[:, excretion] > 0, axis=1)
    if policy == "derive":
        events[active, tail] = 1
        event_masks[active, tail] = 1
    else:  # exclude
        unlabelled = active & (events[:, tail] == 0)
        event_masks[unlabelled, tail] = 0
    return events, event_masks


def label_centers(
    center_times_ms: np.ndarray,
    annotations: pd.DataFrame,
    review_coverage: pd.DataFrame | None = None,
    *,
    tail_raised_policy: str = DEFAULT_TAIL_RAISED_POLICY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return body class, event multi-hot, and conservative event masks."""

    times = np.asarray(center_times_ms, dtype=np.int64)
    body = np.full(times.size, -1, dtype=np.int8)
    events = np.zeros((times.size, len(EVENT_CODES)), dtype=np.uint8)
    for row in annotations.itertuples(index=False):
        mask = (times >= float(row.t_start_rel_ms)) & (times < float(row.t_end_rel_ms))
        if row.code in BODY_CODES:
            class_index = BODY_CODES.index(row.code)
            conflict = mask & (body >= 0) & (body != class_index)
            if np.any(conflict):
                raise ValueError(f"body-state overlap at {row.device_mac}/{row.session_id}")
            body[mask] = class_index
        elif row.code in EVENT_CODES:
            events[mask, EVENT_CODES.index(row.code)] = 1
    event_masks = np.zeros((times.size, len(EVENT_CODES)), dtype=np.uint8)
    if review_coverage is None:
        # Backward-compatible proxy for historical caches. New datasets should
        # supply event-specific exhaustive video-review ranges.
        event_masks[:] = np.repeat((body >= 0)[:, None], len(EVENT_CODES), axis=1)
    else:
        for row in review_coverage.itertuples(index=False):
            reviewed = str(row.exhaustive_reviewed).strip().lower() in {"1", "true", "yes", "y"}
            if not reviewed:
                continue
            covered = (times >= float(row.t_start_rel_ms)) & (times < float(row.t_end_rel_ms))
            code = str(row.event_code).strip().upper()
            indices = range(len(EVENT_CODES)) if code in {"ALL", "*"} else [EVENT_CODES.index(code)]
            for index in indices:
                event_masks[covered, index] = 1
    event_masks[events.astype(bool)] = 1
    events, event_masks = _apply_tail_raised_policy(events, event_masks, tail_raised_policy)
    return body, events, event_masks


def supervised_sample_frame(
    *,
    processed: ProcessedSession,
    annotations: pd.DataFrame,
    cache_name: str,
    cow_id: str,
    device_key: str,
    device_mac: str,
    session_id: str,
    review_coverage: pd.DataFrame | None = None,
    tail_raised_policy: str = DEFAULT_TAIL_RAISED_POLICY,
) -> pd.DataFrame:
    centers = candidate_centers(processed)
    center_times = processed.times_ms[centers]
    body, events, event_masks = label_centers(
        center_times, annotations, review_coverage, tail_raised_policy=tail_raised_policy
    )
    supervised = (body >= 0) | np.any(events > 0, axis=1) | np.any(event_masks > 0, axis=1)
    centers = centers[supervised]
    center_times = center_times[supervised]
    body = body[supervised]
    events = events[supervised]
    event_masks = event_masks[supervised]
    # Segment bounds travel with every row so a model context window can be
    # clipped to its own contiguous run.  ``features.npy`` stores all segments
    # end to end, so without these columns a window at the start of segment k
    # silently reads the tail of segment k-1, which may be hours earlier.
    segment_start, segment_stop = segment_bounds_for_centers(processed, centers)
    frame = pd.DataFrame(
        {
            "cache_key": cache_name,
            "cow_id": cow_id,
            "device_key": device_key,
            "device_mac": device_mac,
            "session_id": session_id,
            "center_index": centers,
            "center_time_ms": center_times,
            "body_target": body,
            "segment_id": processed.segment_ids[centers],
            "segment_start_index": segment_start,
            "segment_stop_index": segment_stop,
        }
    )
    for index, code in enumerate(EVENT_CODES):
        frame[f"event_{code}"] = events[:, index]
        frame[f"mask_{code}"] = event_masks[:, index]
    return frame


def save_processed_session(
    processed: ProcessedSession,
    directory: Path,
    metadata: dict[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    np.save(directory / "features.npy", processed.features, allow_pickle=False)
    np.save(directory / "times_ms.npy", processed.times_ms, allow_pickle=False)
    np.save(directory / "segment_ids.npy", processed.segment_ids, allow_pickle=False)
    with (directory / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump({**metadata, "segments": processed.segments}, handle, ensure_ascii=False, indent=2)
