"""Causal upright/lying state machine tuned without test-set access."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


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
        "posture_target",
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
        "state_macro_f1_raw": float(f1_score(target, raw, labels=[0, 1], average="macro", zero_division=0)),
        "state_macro_f1_sm": float(f1_score(target, smoothed, labels=[0, 1], average="macro", zero_division=0)),
        "raw_predicted_state_changes": int(np.sum(np.diff(frame["state_raw"].to_numpy(np.int8)) != 0)),
        "sm_predicted_state_changes": int(np.sum(np.diff(frame["state_sm"].to_numpy(np.int8)) != 0)),
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
