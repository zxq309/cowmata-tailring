"""Point and event-level metrics with validation-only threshold selection.

Changes against the previous version
------------------------------------
1. ``choose_threshold`` is vectorised through ``precision_recall_curve``.
   The old implementation called ``f1_score`` once per candidate threshold and
   used every distinct validation score as a candidate, i.e. O(N^2) with a
   Python loop: ~540 s per task on a 76 k-row validation split.  The new one is
   O(N log N) and returns an identical threshold.
2. Event truth intervals can be built from the *annotation table* instead of
   being reconstructed from sparse label points, so an annotation hole no
   longer splits one true event into several.
3. ``false_alarms_per_evaluated_hour`` is kept for backwards compatibility but
   is now also reported as ``false_alarms_per_labelled_hour``: label points are
   not wall-clock contiguous and the old name invited misreading.
4. ``rate_plausibility`` compares the predicted event rate with published
   physiological rates.  A model predicting 40 urinations/hour is wrong
   regardless of what the (untrusted) negatives say.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
)


# Published per-animal event rates for housed dairy cattle, expressed as events
# per hour.  Used only as an upper sanity bound, never as a training target.
#   urination  ~9 events/24 h (range 5-18)  -> 0.38/h, generous bound 1.5/h
#   defecation ~16 events/24 h (range 8-29) -> 0.67/h, generous bound 2.5/h
# Posture transitions and tail events are bounded loosely; they are far more
# frequent and far less well characterised in the literature.
PHYSIOLOGICAL_RATE_PER_HOUR: dict[str, float] = {
    "URINATION": 1.5,
    "DEFECATION": 2.5,
    "STANDING_UP": 4.0,
    "LYING_DOWN": 4.0,
    "TAIL_RAISED": 12.0,
    "TAIL_WAGGING": 60.0,
}

# Minimum evidence required before precision / F1 may be reported at all.
MIN_TRUE_EVENTS_FOR_CLAIM = 10
MIN_COWS_FOR_CLAIM = 3


def body_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, object]:
    """Generic multiclass metrics retained under the historical function name."""

    valid = np.asarray(target) >= 0
    if not np.any(valid):
        return {"evaluated": 0, "macro_f1": None, "per_class": [], "confusion_matrix": []}
    actual = np.asarray(target)[valid]
    probabilities = np.asarray(probability)
    class_count = int(probabilities.shape[1])
    labels = np.arange(class_count)
    predicted = np.argmax(probabilities[valid], axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=labels, zero_division=0
    )
    return {
        "evaluated": int(valid.sum()),
        "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "per_class": [
            {
                "class_index": index,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(class_count)
        ],
        "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
    }


def choose_threshold(
    target: np.ndarray,
    probability: np.ndarray,
    mask: np.ndarray,
    *,
    beta: float = 1.0,
    max_positive_rate: float | None = None,
) -> float:
    """Pick the F-beta optimal threshold on the *validation* split only.

    ``max_positive_rate`` optionally restricts the search to thresholds whose
    predicted-positive fraction stays below a bound; use it to encode a
    physiological ceiling (see :data:`PHYSIOLOGICAL_RATE_PER_HOUR`).  When no
    threshold satisfies the bound the constraint is dropped and the
    unconstrained optimum is returned, so this can never fail closed.
    """

    valid = np.asarray(mask).astype(bool)
    actual = np.asarray(target)[valid].astype(np.int8)
    scores = np.asarray(probability, dtype=np.float64)[valid]
    if actual.size == 0 or np.unique(actual).size < 2:
        return 0.5

    precision, recall, thresholds = precision_recall_curve(actual, scores)
    # precision_recall_curve returns len(thresholds) + 1 points; the final point
    # (recall 0, precision 1) has no threshold and is dropped.
    precision = precision[:-1]
    recall = recall[:-1]
    beta2 = float(beta) ** 2
    denominator = beta2 * precision + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        fbeta = np.where(denominator > 0, (1.0 + beta2) * precision * recall / denominator, 0.0)

    usable = np.ones(thresholds.shape, dtype=bool)
    if max_positive_rate is not None:
        # predicted-positive fraction at each threshold, computed once.
        order = np.argsort(scores, kind="stable")
        sorted_scores = scores[order]
        predicted_positive = scores.size - np.searchsorted(sorted_scores, thresholds, side="left")
        rate = predicted_positive / float(scores.size)
        constrained = rate <= float(max_positive_rate)
        if np.any(constrained):
            usable = constrained

    candidate_scores = np.where(usable, fbeta, -np.inf)
    best_value = float(np.max(candidate_scores))
    if not np.isfinite(best_value):
        return 0.5
    tied = np.flatnonzero(candidate_scores >= best_value - 1e-12)
    # Historical tie-break: prefer the threshold closest to 0.5.
    chosen = tied[np.argmin(np.abs(thresholds[tied] - 0.5))]
    return float(thresholds[chosen])


def binary_point_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    mask: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    valid = np.asarray(mask).astype(bool)
    actual = np.asarray(target)[valid].astype(np.int8)
    scores = np.asarray(probability)[valid]
    if actual.size == 0:
        return {
            "scope": "no_evaluable_points",
            "evaluated": 0,
            "positive": 0,
            "negative": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "average_precision": None,
        }
    predicted = scores >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, average="binary", zero_division=0
    )
    ap = float(average_precision_score(actual, scores)) if len(np.unique(actual)) == 2 else None
    positive = int(actual.sum())
    negative = int(actual.size - positive)
    if positive > 0 and negative > 0:
        scope = "positive_and_trusted_negative"
        reported_precision: float | None = float(precision)
        reported_f1: float | None = float(f1)
        reported_recall: float | None = float(recall)
    elif positive > 0:
        scope = "positive_only_recall"
        reported_precision = None
        reported_f1 = None
        reported_recall = float(recall)
    else:
        scope = "trusted_negative_only"
        reported_precision = None
        reported_f1 = None
        reported_recall = None
    return {
        "scope": scope,
        "evaluated": int(actual.size),
        "positive": positive,
        "negative": negative,
        "predicted_positive": int(predicted.sum()),
        "false_positive": int(np.sum(predicted & (actual == 0))),
        "false_positive_rate": float(np.mean(predicted[actual == 0])) if negative else None,
        "precision": reported_precision,
        "recall": reported_recall,
        "f1": reported_f1,
        "average_precision": ap,
    }


def intervals_from_binary(times_ms: np.ndarray, values: np.ndarray, *, max_gap_ms: int = 750) -> list[tuple[int, int]]:
    times = np.asarray(times_ms, dtype=np.int64)
    positive = np.asarray(values).astype(bool)
    selected = times[positive]
    if selected.size == 0:
        return []
    intervals: list[tuple[int, int]] = []
    start = int(selected[0])
    previous = int(selected[0])
    for current_value in selected[1:]:
        current = int(current_value)
        if current - previous > max_gap_ms:
            intervals.append((start, previous + 500))
            start = current
        previous = current
    intervals.append((start, previous + 500))
    return intervals


def truth_intervals_from_annotations(
    annotations: "object",
    device_mac: str,
    session_id: str,
    code: str,
) -> list[tuple[int, int]]:
    """Read true event intervals straight from the adjudicated annotation table.

    Reconstructing them from 2 Hz label points (``intervals_from_binary``)
    splits one real event into several whenever the label series has a hole,
    which inflates the true-event count and depresses recall.
    """

    if annotations is None:
        return []
    selected = annotations[
        (annotations["device_mac"].astype(str) == str(device_mac))
        & (annotations["session_id"].astype(str) == str(session_id))
        & (annotations["code"].astype(str) == str(code))
    ]
    if len(selected) == 0:
        return []
    starts = selected["t_start_rel_ms"].to_numpy(np.int64)
    ends = selected["t_end_rel_ms"].to_numpy(np.int64)
    order = np.argsort(starts, kind="stable")
    return [(int(starts[i]), int(ends[i])) for i in order]


# Post-processing applied to the *predicted* point series before it is turned
# into events.  Without it a 35 s urination whose probability dips below the
# threshold for one 0.5 s step is scored as two predictions, one of which
# becomes a false alarm.  Values are deliberately shorter than the observed
# annotation durations so a real event is never dissolved.
EVENT_POSTPROCESS: dict[str, dict[str, int]] = {
    "STANDING_UP": {"merge_gap_ms": 2000, "min_ms": 1000},
    "LYING_DOWN": {"merge_gap_ms": 2000, "min_ms": 1000},
    "URINATION": {"merge_gap_ms": 5000, "min_ms": 4000},
    "DEFECATION": {"merge_gap_ms": 4000, "min_ms": 3000},
    "TAIL_RAISED": {"merge_gap_ms": 3000, "min_ms": 2000},
    "TAIL_WAGGING": {"merge_gap_ms": 1500, "min_ms": 1000},
}


def postprocess_intervals(
    intervals: list[tuple[int, int]],
    code: str,
    *,
    enabled: bool = True,
) -> list[tuple[int, int]]:
    """Merge near-adjacent predicted intervals and drop implausibly short ones."""

    if not enabled or not intervals:
        return intervals
    rule = EVENT_POSTPROCESS.get(str(code).upper())
    if rule is None:
        return intervals
    ordered = sorted(intervals)
    merged: list[list[int]] = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start - merged[-1][1] <= rule["merge_gap_ms"]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(int(s), int(e)) for s, e in merged if e - s >= rule["min_ms"]]


def match_intervals(
    truth: list[tuple[int, int]],
    predicted: list[tuple[int, int]],
    *,
    tolerance_ms: int = 2500,
) -> tuple[int, int, int]:
    candidates: list[tuple[float, int, int]] = []
    for truth_index, (truth_start, truth_end) in enumerate(truth):
        for pred_index, (pred_start, pred_end) in enumerate(predicted):
            expanded_start = truth_start - tolerance_ms
            expanded_end = truth_end + tolerance_ms
            overlap = min(expanded_end, pred_end) - max(expanded_start, pred_start)
            if overlap > 0:
                candidates.append((float(overlap), truth_index, pred_index))
    matched_truth: set[int] = set()
    matched_prediction: set[int] = set()
    for _, truth_index, pred_index in sorted(candidates, reverse=True):
        if truth_index not in matched_truth and pred_index not in matched_prediction:
            matched_truth.add(truth_index)
            matched_prediction.add(pred_index)
    tp = len(matched_truth)
    return tp, len(predicted) - tp, len(truth) - tp


def rate_plausibility(code: str, predicted_events: int, labelled_hours: float) -> dict[str, object]:
    """Compare the predicted event rate with a published physiological bound."""

    bound = PHYSIOLOGICAL_RATE_PER_HOUR.get(str(code).upper())
    if bound is None or labelled_hours <= 0:
        return {"bound_per_hour": bound, "observed_per_hour": None, "ratio": None, "verdict": "unknown"}
    observed = predicted_events / labelled_hours
    ratio = observed / bound
    if ratio <= 1.0:
        verdict = "plausible"
    elif ratio <= 5.0:
        verdict = "suspicious"
    else:
        verdict = "implausible"
    return {
        "bound_per_hour": float(bound),
        "observed_per_hour": float(observed),
        "ratio": float(ratio),
        "verdict": verdict,
    }


def event_level_metrics(
    prediction_frame: "object",
    code: str,
    threshold: float,
    *,
    tolerance_ms: int = 2500,
    annotations: "object" = None,
    postprocess: bool = True,
    min_true_events: int = MIN_TRUE_EVENTS_FOR_CLAIM,
    min_cows: int = MIN_COWS_FOR_CLAIM,
) -> dict[str, object]:
    """Aggregate one-to-one event matches across sessions.

    ``prediction_frame`` is duck-typed: it must contain ``center_time_ms``,
    ``target_{code}``, ``mask_{code}``, ``prob_{code}``, ``device_mac`` and
    ``session_id``; ``cow_id`` is used when present to count evidence breadth.
    When ``annotations`` is supplied the truth intervals come from it instead
    of from the sparse label points.
    """

    target_column = f"target_{code}"
    mask_column = f"mask_{code}"
    probability_column = f"prob_{code}"
    total_true = 0
    total_predicted = 0
    total_tp = 0
    evaluated_points = 0
    trusted_negative_points = 0
    cows_with_truth: set[str] = set()
    details: list[dict[str, object]] = []
    for (device, session), group in prediction_frame.groupby(["device_mac", "session_id"], sort=True):
        ordered = group.sort_values("center_time_ms", kind="stable")
        valid = ordered[mask_column].to_numpy().astype(bool)
        if not np.any(valid):
            continue
        times = ordered.loc[valid, "center_time_ms"].to_numpy(np.int64)
        actual = ordered.loc[valid, target_column].to_numpy(np.uint8)
        predicted = ordered.loc[valid, probability_column].to_numpy(float) >= threshold
        if annotations is not None:
            truth_intervals = truth_intervals_from_annotations(annotations, device, session, code)
        else:
            truth_intervals = intervals_from_binary(times, actual)
        predicted_intervals = postprocess_intervals(
            intervals_from_binary(times, predicted), code, enabled=postprocess
        )
        tp, fp, fn = match_intervals(
            truth_intervals, predicted_intervals, tolerance_ms=tolerance_ms
        )
        total_true += len(truth_intervals)
        total_predicted += len(predicted_intervals)
        total_tp += tp
        evaluated_points += len(actual)
        trusted_negative_points += int(np.sum(actual == 0))
        if truth_intervals and "cow_id" in ordered.columns:
            cows_with_truth.add(str(ordered["cow_id"].iloc[0]))
        if truth_intervals or predicted_intervals:
            details.append(
                {
                    "device_mac": str(device),
                    "session_id": str(session),
                    "true_events": len(truth_intervals),
                    "predicted_events": len(predicted_intervals),
                    "true_positive": tp,
                    "false_positive": fp,
                    "false_negative": fn,
                }
            )
    false_positive = total_predicted - total_tp
    false_negative = total_true - total_tp
    recall = total_tp / total_true if total_true else None
    labelled_hours = evaluated_points * 0.5 / 3600.0
    false_alarms_per_hour = false_positive / labelled_hours if labelled_hours > 0 else None
    if total_true > 0 and trusted_negative_points > 0:
        scope = "positive_and_trusted_negative"
        precision = total_tp / total_predicted if total_predicted else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if recall is not None and precision + recall > 0
            else 0.0
        )
    elif total_true > 0:
        scope = "positive_only_recall"
        precision = None
        f1 = None
        false_alarms_per_hour = None
    else:
        scope = "trusted_negative_only" if trusted_negative_points else "no_evaluable_events"
        precision = None
        recall = None
        f1 = None

    enough_events = total_true >= int(min_true_events)
    enough_cows = (len(cows_with_truth) >= int(min_cows)) if cows_with_truth else False
    claimable = bool(scope == "positive_and_trusted_negative" and enough_events and enough_cows)
    evidence = "reportable" if claimable else "not_evaluable"
    return {
        "scope": scope,
        "evidence_level": evidence,
        "precision_f1_claimable": claimable,
        "cows_with_truth": sorted(cows_with_truth),
        "threshold": float(threshold),
        "matching_tolerance_ms": tolerance_ms,
        "postprocess": EVENT_POSTPROCESS.get(str(code).upper()) if postprocess else None,
        "evaluated_points": evaluated_points,
        "labelled_hours": float(labelled_hours),
        "trusted_negative_points": trusted_negative_points,
        "true_events": total_true,
        "predicted_events": total_predicted,
        "true_positive": total_tp,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarms_per_labelled_hour": false_alarms_per_hour,
        # Deprecated alias kept so existing report readers do not break.
        "false_alarms_per_evaluated_hour": false_alarms_per_hour,
        "rate_plausibility": rate_plausibility(code, total_predicted, labelled_hours),
        "session_details": details,
    }


def selection_score(
    posture_macro_f1: float | None,
    walking_average_precision: float | None,
    event_average_precisions: dict[str, float | None],
    *,
    posture_weight: float = 0.5,
    walking_weight: float = 0.3,
    event_weight: float = 0.2,
) -> dict[str, object]:
    """Model-selection objective.

    The previous version averaged eight thresholded F1 values with equal
    weight, so four tail-wagging annotations from one cow counted as much as
    a hundred thousand posture labels.  This version weights the tasks by how
    much evidence supports them and uses average precision, which does not
    depend on a threshold and therefore has far lower variance.
    """

    usable_events = {
        code: float(value)
        for code, value in event_average_precisions.items()
        if value is not None and np.isfinite(value)
    }
    parts: list[tuple[float, float]] = []
    if posture_macro_f1 is not None:
        parts.append((float(posture_weight), float(posture_macro_f1)))
    if walking_average_precision is not None:
        parts.append((float(walking_weight), float(walking_average_precision)))
    if usable_events:
        parts.append((float(event_weight), float(np.mean(list(usable_events.values())))))
    if not parts:
        return {"selection_score": None, "components": {}, "events_used": []}
    total_weight = sum(weight for weight, _ in parts)
    score = sum(weight * value for weight, value in parts) / total_weight
    return {
        "selection_score": float(score),
        "components": {
            "posture_macro_f1": posture_macro_f1,
            "walking_average_precision": walking_average_precision,
            "event_mean_average_precision": float(np.mean(list(usable_events.values())))
            if usable_events
            else None,
        },
        "events_used": sorted(usable_events),
    }
