"""Turning a probability series into events, and events into a timeline.

Two stages live here.

**Interval assembly.**  The 20260818 path thresholded each event probability
independently and merged whatever survived.  That is why a 35 s urination whose
probability dipped for one 0.5 s step was scored as two predictions, one of
which became a false alarm.  :func:`assemble_intervals` keeps that behaviour as
its base and adds the two things the segmentation literature uses instead:

* **hysteresis** - a high threshold to *start* an event and a lower one to
  *continue* it.  One threshold has to be simultaneously strict enough to avoid
  spurious onsets and loose enough not to chop a real event in half, and it
  cannot be both.
* **boundary snapping** - when the model's ASRF-style boundary head is
  available, an assembled interval is trimmed to the nearest boundary peaks.
  This is what converts a good frame-level model into a good *event-level* one,
  and it is the mechanism that should move event precision, not a bigger
  backbone.

**Posture state machine.**  Ported from ``cattle_imu/state_machine.py``
unchanged in behaviour.  It is a hand-written two-state HMM with a confirmation
count and a dwell time; it is tuned on validation and applied once to test.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .labels import EVENT_POSTPROCESS

DECISION_STEP_MS = 500


# ==========================================================================
# interval assembly
# ==========================================================================
def hysteresis_mask(
    scores: np.ndarray, *, high: float, low: float | None = None
) -> np.ndarray:
    """Start a run where ``scores >= high``, continue it while ``scores >= low``.

    A run is kept only if it contains at least one sample above ``high``, so the
    low threshold can never invent an event on its own.
    """

    values = np.asarray(scores, dtype=np.float64)
    high = float(high)
    low = float(high if low is None else low)
    if low > high:
        raise ValueError("the continuation threshold must not exceed the onset threshold")
    above_low = values >= low
    if not above_low.any():
        return np.zeros(values.size, dtype=bool)
    padded = np.concatenate(([False], above_low, [False]))
    change = np.flatnonzero(padded[1:] != padded[:-1])
    keep = np.zeros(values.size, dtype=bool)
    for start, stop in zip(change[0::2], change[1::2]):
        if np.any(values[start:stop] >= high):
            keep[start:stop] = True
    return keep


def snap_to_boundaries(
    interval: tuple[int, int],
    times_ms: np.ndarray,
    boundary: np.ndarray,
    *,
    search_ms: int = 3000,
    threshold: float = 0.5,
) -> tuple[int, int]:
    """Trim an interval to the nearest confident boundary peaks.

    The search window is deliberately narrow: a boundary further away than
    ``search_ms`` belongs to a different event, and snapping to it would fuse
    two behaviours into one.
    """

    times = np.asarray(times_ms, dtype=np.int64)
    scores = np.asarray(boundary, dtype=np.float64)
    if times.size == 0 or scores.size != times.size:
        return interval
    start, stop = int(interval[0]), int(interval[1])

    def nearest(anchor: int) -> int | None:
        window = (times >= anchor - search_ms) & (times <= anchor + search_ms)
        if not np.any(window):
            return None
        candidates = np.flatnonzero(window & (scores >= threshold))
        if candidates.size == 0:
            return None
        best = candidates[np.argmax(scores[candidates])]
        return int(times[best])

    snapped_start = nearest(start)
    snapped_stop = nearest(stop)
    new_start = snapped_start if snapped_start is not None else start
    new_stop = snapped_stop if snapped_stop is not None else stop
    if new_stop - new_start < DECISION_STEP_MS:
        return start, stop
    return new_start, new_stop


def merge_and_filter(
    intervals: list[tuple[int, int]], code: str, *, enabled: bool = True
) -> list[tuple[int, int]]:
    """Merge near-adjacent intervals and drop implausibly short ones."""

    if not enabled or not intervals:
        return list(intervals)
    rule = EVENT_POSTPROCESS.get(str(code).upper())
    if rule is None:
        return list(intervals)
    ordered = sorted(intervals)
    merged: list[list[int]] = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start - merged[-1][1] <= rule["merge_gap_ms"]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(int(s), int(e)) for s, e in merged if e - s >= rule["min_ms"]]


def assemble_intervals(
    times_ms: np.ndarray,
    scores: np.ndarray,
    code: str,
    *,
    threshold: float,
    low_threshold: float | None = None,
    boundary: np.ndarray | None = None,
    boundary_threshold: float = 0.5,
    boundary_search_ms: int = 3000,
    max_gap_ms: int = 750,
    postprocess: bool = True,
) -> list[tuple[int, int]]:
    """Full probability-series to event-interval pipeline for one event code.

    ``low_threshold`` defaults to ``0.6 * threshold``, i.e. mild hysteresis.
    Pass ``low_threshold=threshold`` to reproduce the 20260818 single-threshold
    behaviour exactly.
    """

    times = np.asarray(times_ms, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if times.size != values.size:
        raise ValueError("times and scores must have the same length")
    if times.size == 0:
        return []
    low = float(threshold) * 0.6 if low_threshold is None else float(low_threshold)
    selected = hysteresis_mask(values, high=float(threshold), low=low)
    if not selected.any():
        return []

    chosen_times = times[selected]
    raw: list[tuple[int, int]] = []
    start = previous = int(chosen_times[0])
    for value in chosen_times[1:]:
        current = int(value)
        if current - previous > max_gap_ms:
            raw.append((start, previous + DECISION_STEP_MS))
            start = current
        previous = current
    raw.append((start, previous + DECISION_STEP_MS))

    if boundary is not None:
        raw = [
            snap_to_boundaries(
                item,
                times,
                boundary,
                search_ms=boundary_search_ms,
                threshold=boundary_threshold,
            )
            for item in raw
        ]
    return merge_and_filter(raw, code, enabled=postprocess)


# ==========================================================================
# posture state machine
# ==========================================================================
@dataclass(frozen=True)
class StateMachineConfig:
    confirm_points: int = 4
    min_dwell_points: int = 4
    margin: float = 0.10
    transition_event_weight: float = 0.50
    max_sequence_gap_ms: int = 750

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _run_sequence(frame: pd.DataFrame, config: StateMachineConfig) -> np.ndarray:
    lying = frame["prob_posture_LYING"].to_numpy(float)
    upright = frame["prob_posture_UPRIGHT"].to_numpy(float)
    standing_up = frame["prob_STANDING_UP"].to_numpy(float)
    lying_down = frame["prob_LYING_DOWN"].to_numpy(float)
    states = np.empty(len(frame), dtype=np.int8)
    state = int(lying[0] > upright[0])
    candidate = -1
    candidate_count = 0
    dwell = config.min_dwell_points
    for index in range(len(frame)):
        if index == 0:
            states[index] = state
            continue
        to_lying = lying[index] + config.transition_event_weight * lying_down[index]
        to_upright = upright[index] + config.transition_event_weight * standing_up[index]
        proposed = state
        if state == 0 and to_lying > to_upright + config.margin:
            proposed = 1
        elif state == 1 and to_upright > to_lying + config.margin:
            proposed = 0
        if proposed == state:
            candidate = -1
            candidate_count = 0
        else:
            candidate_count = candidate_count + 1 if candidate == proposed else 1
            candidate = proposed
            if dwell >= config.min_dwell_points and candidate_count >= config.confirm_points:
                state = proposed
                dwell = 0
                candidate = -1
                candidate_count = 0
        dwell += 1
        states[index] = state
    return states


def apply_state_machine(predictions: pd.DataFrame, config: StateMachineConfig) -> pd.DataFrame:
    required = {
        "device_mac",
        "session_id",
        "center_time_ms",
        "prob_posture_UPRIGHT",
        "prob_posture_LYING",
        "prob_STANDING_UP",
        "prob_LYING_DOWN",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"state-machine input is missing columns: {sorted(missing)}")
    output = predictions.copy()
    output["state_raw"] = (
        output["prob_posture_LYING"].to_numpy(float)
        > output["prob_posture_UPRIGHT"].to_numpy(float)
    ).astype(np.int8)
    output["state_sm"] = -1
    for _, group in output.groupby(["device_mac", "session_id"], sort=False):
        ordered = group.sort_values("center_time_ms", kind="stable")
        times = ordered["center_time_ms"].to_numpy(np.int64)
        boundaries = np.flatnonzero(np.diff(times) > config.max_sequence_gap_ms) + 1
        starts = np.concatenate(([0], boundaries))
        stops = np.concatenate((boundaries, [len(ordered)]))
        for start, stop in zip(starts, stops):
            if stop > start:
                block = ordered.iloc[start:stop]
                output.loc[block.index, "state_sm"] = _run_sequence(block, config)
    return output


def state_machine_metrics(frame: pd.DataFrame) -> dict[str, object]:
    valid = frame["posture_target"].to_numpy(np.int64) >= 0
    if not np.any(valid):
        return {"evaluated": 0, "state_macro_f1_raw": None, "state_macro_f1_sm": None}
    target = frame.loc[valid, "posture_target"].to_numpy(np.int64)
    raw = frame.loc[valid, "state_raw"].to_numpy(np.int64)
    smoothed = frame.loc[valid, "state_sm"].to_numpy(np.int64)
    return {
        "evaluated": int(valid.sum()),
        "state_macro_f1_raw": float(
            f1_score(target, raw, labels=[0, 1], average="macro", zero_division=0)
        ),
        "state_macro_f1_sm": float(
            f1_score(target, smoothed, labels=[0, 1], average="macro", zero_division=0)
        ),
        "raw_predicted_state_changes": int(
            np.sum(np.diff(frame["state_raw"].to_numpy(np.int8)) != 0)
        ),
        "sm_predicted_state_changes": int(
            np.sum(np.diff(frame["state_sm"].to_numpy(np.int8)) != 0)
        ),
    }


def tune_state_machine(validation: pd.DataFrame) -> tuple[StateMachineConfig, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    best: tuple[tuple[float, float], StateMachineConfig] | None = None
    for confirm in (2, 4, 6, 10):
        for dwell in (2, 4, 10, 20):
            for margin in (0.0, 0.10, 0.20, 0.30):
                for event_weight in (0.0, 0.25, 0.50, 1.0):
                    config = StateMachineConfig(confirm, dwell, margin, event_weight)
                    metrics = state_machine_metrics(apply_state_machine(validation, config))
                    rank = (float(metrics["state_macro_f1_sm"]), -float(confirm + dwell))
                    rows.append({**config.to_dict(), **metrics})
                    if best is None or rank > best[0]:
                        best = (rank, config)
    assert best is not None
    return best[1], pd.DataFrame(rows).sort_values("state_macro_f1_sm", ascending=False)
