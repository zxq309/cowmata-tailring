"""Gap-safe resampling and label construction.

Two things changed against 20260818.

**Resampling writes counts.**  ``resample_session`` now interpolates the raw
int16 device counts and writes them straight into the schema-2 cache.  This is
not an approximation: the counts-to-physical map is affine per axis (bias,
divisor) and the magnetometer correction is a signed permutation, so
interpolating before or after that map gives the same trajectory.  The only new
quantity is a half-count rounding error, which is by definition below the
device's own resolution.  In exchange the calibration is no longer baked into
terabytes of cache and can be corrected without a rebuild.

**Labels come in two shapes.**  ``supervised_sample_frame`` keeps the sparse
2 Hz table the gradient-boosting branch and the feature table consume.
``dense_label_frame`` returns the *whole* 2 Hz grid of a session, supervised or
not, with a mask column per task.  Dense training needs the second: the model
consumes a contiguous stretch of signal and is supervised wherever the mask is
1, instead of re-encoding a 40 s window once per label point.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .cache import Calibration, Segment, TARGET_DT_MS, TARGET_HZ
from .io import ImuSession, contiguous_slices
from .labels import EVENT_CODES, STATE_ANNOTATION_CODES

#: 0.5 s decision step. Everything downstream assumes 2 Hz labels.
TARGET_STRIDE_SAMPLES = 25
DECISION_HZ = TARGET_HZ / TARGET_STRIDE_SAMPLES
GAP_THRESHOLD_MS = 40.0
MIN_SEGMENT_SAMPLES = 256

TAIL_RAISED_POLICIES = ("derive", "exclude", "legacy")
DEFAULT_TAIL_RAISED_POLICY = "derive"


@dataclass(frozen=True)
class ProcessedSession:
    """Resampled session, ready for :func:`cowmata.cache.write_cache_v2`."""

    counts: np.ndarray  # (N, 9) int16 on the 50 Hz grid
    quality_flag: np.ndarray  # (N,) float32, 1 where the raw cadence was off
    segments: list[Segment] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return int(self.counts.shape[0])

    def times_ms(self) -> np.ndarray:
        out = np.zeros(self.n_frames, dtype=np.int64)
        for segment in self.segments:
            span = np.arange(segment.length, dtype=np.int64)
            out[segment.start_index : segment.stop_index] = (
                segment.start_ms + span * TARGET_DT_MS
            )
        return out


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
    gap_threshold_ms: float = GAP_THRESHOLD_MS,
) -> ProcessedSession:
    """Resample each continuous run independently; no interpolation crosses a gap."""

    counts_chunks: list[np.ndarray] = []
    quality_chunks: list[np.ndarray] = []
    segments: list[Segment] = []
    offset = 0
    raw_values = session.raw_values.astype(np.float64, copy=False)
    for run in contiguous_slices(session.elapsed_ms, gap_threshold_ms=gap_threshold_ms):
        raw_time = session.elapsed_ms[run].astype(np.float64)
        block = raw_values[run]
        if raw_time.size < MIN_SEGMENT_SAMPLES:
            continue
        start = int(math.ceil(raw_time[0] / TARGET_DT_MS) * TARGET_DT_MS)
        stop = int(math.floor(raw_time[-1] / TARGET_DT_MS) * TARGET_DT_MS)
        if stop - start < (MIN_SEGMENT_SAMPLES - 1) * TARGET_DT_MS:
            continue
        grid = np.arange(start, stop + 1, TARGET_DT_MS, dtype=np.int64)
        resampled = np.column_stack(
            [np.interp(grid, raw_time, block[:, index]) for index in range(block.shape[1])]
        )
        counts_chunks.append(np.rint(resampled).astype(np.int16))
        quality_chunks.append(_quality_flag(raw_time, grid))
        segments.append(
            Segment(
                segment_id=len(segments),
                start_index=offset,
                stop_index=offset + grid.size,
                start_ms=int(grid[0]),
                stop_ms=int(grid[-1]),
            )
        )
        offset += grid.size
    if not counts_chunks:
        return ProcessedSession(
            counts=np.empty((0, 9), dtype=np.int16),
            quality_flag=np.empty(0, dtype=np.float32),
            segments=[],
        )
    return ProcessedSession(
        counts=np.concatenate(counts_chunks, axis=0),
        quality_flag=np.concatenate(quality_chunks),
        segments=segments,
    )


def calibration_from_manifest(item: dict[str, Any]) -> Calibration:
    return Calibration(
        acc_divisor=float(item["acc_divisor"]),
        gyro_divisor=float(item.get("gyro_divisor", 32.0)),
        mag_divisor=float(item.get("mag_divisor", 1000.0)),
        acc_bias_counts=tuple(float(v) for v in item.get("acc_bias_counts", (0, 0, 0))),
        gyro_bias_counts=tuple(float(v) for v in item.get("gyro_bias_counts", (0, 0, 0))),
    )


# --------------------------------------------------------------------------
# label grid
# --------------------------------------------------------------------------
def label_grid(
    segments: list[Segment],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(center_index, center_time_ms, segment_start, segment_stop)``.

    The grid covers every segment at the 2 Hz decision step.  Segment bounds
    travel with each point so any consumer can clip a context to its own
    contiguous run; a cache stores all segments end to end, so without them a
    window at the start of segment *k* silently reads the tail of segment
    *k - 1*, which may be hours earlier.
    """

    centers: list[np.ndarray] = []
    times: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    stops: list[np.ndarray] = []
    for segment in segments:
        local = np.arange(0, segment.length, TARGET_STRIDE_SAMPLES, dtype=np.int64)
        if local.size == 0:
            continue
        centers.append(segment.start_index + local)
        times.append(segment.start_ms + local * TARGET_DT_MS)
        starts.append(np.full(local.size, segment.start_index, dtype=np.int64))
        stops.append(np.full(local.size, segment.stop_index, dtype=np.int64))
    if not centers:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty, empty
    return (
        np.concatenate(centers),
        np.concatenate(times),
        np.concatenate(starts),
        np.concatenate(stops),
    )


def _apply_tail_raised_policy(
    events: np.ndarray, event_masks: np.ndarray, policy: str
) -> tuple[np.ndarray, np.ndarray]:
    """Reconcile TAIL_RAISED with the excretion codes.

    URINATION is annotated as the *whole* event including the preparatory tail
    lift, yet no URINATION or DEFECATION interval in the adjudicated table is
    also annotated TAIL_RAISED.  Under the legacy mask rule those intervals
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
    """Return body class, event multi-hot, and conservative event masks.

    Annotation codes outside :data:`cowmata.labels.EVENT_CODES` (currently only
    ``TAIL_WAGGING``) are read without error and then ignored, so the historical
    table stays loadable while the deprecated head is gone from the model.
    """

    times = np.asarray(center_times_ms, dtype=np.int64)
    body = np.full(times.size, -1, dtype=np.int8)
    events = np.zeros((times.size, len(EVENT_CODES)), dtype=np.uint8)
    for row in annotations.itertuples(index=False):
        code = str(row.code)
        mask = (times >= float(row.t_start_rel_ms)) & (times < float(row.t_end_rel_ms))
        if code in STATE_ANNOTATION_CODES:
            class_index = STATE_ANNOTATION_CODES.index(code)
            conflict = mask & (body >= 0) & (body != class_index)
            if np.any(conflict):
                raise ValueError(
                    f"body-state overlap at {getattr(row, 'device_mac', '?')}/"
                    f"{getattr(row, 'session_id', '?')}"
                )
            body[mask] = class_index
        elif code in EVENT_CODES:
            events[mask, EVENT_CODES.index(code)] = 1
    event_masks = np.zeros((times.size, len(EVENT_CODES)), dtype=np.uint8)
    if review_coverage is None or len(review_coverage) == 0:
        # Backward-compatible proxy for historical caches. New datasets must
        # supply event-specific exhaustive video-review ranges; without them no
        # precision or false-alarm number is claimable (see docs/METRICS.md).
        event_masks[:] = np.repeat((body >= 0)[:, None], len(EVENT_CODES), axis=1)
    else:
        for row in review_coverage.itertuples(index=False):
            reviewed = str(row.exhaustive_reviewed).strip().lower() in {"1", "true", "yes", "y"}
            if not reviewed:
                continue
            covered = (times >= float(row.t_start_rel_ms)) & (times < float(row.t_end_rel_ms))
            code = str(row.event_code).strip().upper()
            if code in {"ALL", "*"}:
                indices: list[int] = list(range(len(EVENT_CODES)))
            elif code in EVENT_CODES:
                indices = [EVENT_CODES.index(code)]
            else:
                continue
            for index in indices:
                event_masks[covered, index] = 1
    event_masks[events.astype(bool)] = 1
    events, event_masks = _apply_tail_raised_policy(events, event_masks, tail_raised_policy)
    return body, events, event_masks


def _label_frame(
    *,
    centers: np.ndarray,
    center_times: np.ndarray,
    segment_start: np.ndarray,
    segment_stop: np.ndarray,
    body: np.ndarray,
    events: np.ndarray,
    event_masks: np.ndarray,
    cache_name: str,
    cow_id: str,
    device_key: str,
    device_mac: str,
    session_id: str,
) -> pd.DataFrame:
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
            "segment_start_index": segment_start,
            "segment_stop_index": segment_stop,
        }
    )
    for index, code in enumerate(EVENT_CODES):
        frame[f"event_{code}"] = events[:, index]
        frame[f"mask_{code}"] = event_masks[:, index]
    return frame


def dense_label_frame(
    *,
    segments: list[Segment],
    annotations: pd.DataFrame,
    cache_name: str,
    cow_id: str,
    device_key: str,
    device_mac: str,
    session_id: str,
    review_coverage: pd.DataFrame | None = None,
    tail_raised_policy: str = DEFAULT_TAIL_RAISED_POLICY,
) -> pd.DataFrame:
    """Every 2 Hz grid point of the session, supervised or not."""

    centers, times, starts, stops = label_grid(segments)
    body, events, masks = label_centers(
        times, annotations, review_coverage, tail_raised_policy=tail_raised_policy
    )
    return _label_frame(
        centers=centers,
        center_times=times,
        segment_start=starts,
        segment_stop=stops,
        body=body,
        events=events,
        event_masks=masks,
        cache_name=cache_name,
        cow_id=cow_id,
        device_key=device_key,
        device_mac=device_mac,
        session_id=session_id,
    )


def supervised_sample_frame(
    *,
    segments: list[Segment],
    annotations: pd.DataFrame,
    cache_name: str,
    cow_id: str,
    device_key: str,
    device_mac: str,
    session_id: str,
    review_coverage: pd.DataFrame | None = None,
    tail_raised_policy: str = DEFAULT_TAIL_RAISED_POLICY,
) -> pd.DataFrame:
    """The supervised subset of :func:`dense_label_frame`."""

    frame = dense_label_frame(
        segments=segments,
        annotations=annotations,
        cache_name=cache_name,
        cow_id=cow_id,
        device_key=device_key,
        device_mac=device_mac,
        session_id=session_id,
        review_coverage=review_coverage,
        tail_raised_policy=tail_raised_policy,
    )
    mask_columns = [f"mask_{code}" for code in EVENT_CODES]
    event_columns = [f"event_{code}" for code in EVENT_CODES]
    supervised = (
        (frame["body_target"].to_numpy() >= 0)
        | (frame[event_columns].to_numpy().sum(axis=1) > 0)
        | (frame[mask_columns].to_numpy().sum(axis=1) > 0)
    )
    return frame.loc[supervised].reset_index(drop=True)


def coverage_summary(center_times_ms: np.ndarray, event_masks: np.ndarray) -> dict[str, float]:
    """Supervised hours per event code, for the cache summary.

    A session missing from ``review_coverage`` silently loses every negative
    sample; printing the covered hours per code makes that visible instead of
    looking like a data change.
    """

    times = np.asarray(center_times_ms, dtype=np.int64)
    masks = np.asarray(event_masks)
    if times.size == 0:
        return {code: 0.0 for code in EVENT_CODES}
    step_hours = (1.0 / DECISION_HZ) / 3600.0
    return {
        code: float(masks[:, index].sum() * step_hours) for index, code in enumerate(EVENT_CODES)
    }
