"""The COWMATA metric suite.

Five layers, all reported internally, only the fifth quoted to customers.
The layering exists because a single number cannot answer the question the
project actually has to answer, which is not "is the classifier good" but "how
often does this product wake a farmer up for nothing".

===== ==================================================================
layer content
===== ==================================================================
1     frame level: per-class P / R / F1, macro-F1, MoF
2     segment level: ``F1@tIoU``, ``edit score``, plus the historical
      2.5 s-tolerance matching kept for continuity
3     deployment level: false alarms per cow per 24 h, miss rate,
      onset localisation error (median and P90)
4     generalisation: every metric grouped **per cow**, plus a bootstrap
      confidence interval resampled over cows
5     day level: sensitivity, specificity, PPV and lead time for
      oestrus / calving
===== ==================================================================

Why layer 2 was added
---------------------
The 20260818 pipeline matched events with a fixed 2.5 s tolerance.  That rule
is lenient for a 1 s stand-up and strict for a 35 s defecation, so the numbers
of two different events are not on the same scale, and no reviewer outside this
project recognises the protocol.  ``F1@{10,25,50}`` and ``edit score`` are what
the temporal-action-segmentation literature reports, and the edit score in
particular quantifies over-segmentation directly - which is the diagnosed cause
of the low event-level precision, so it belongs in the report rather than in a
commit message.

Why layer 4 is mandatory
------------------------
In the 20260818 data one animal contributed 71.6% of all supervised samples.
A pooled cross-cow number computed over that is an average of one cow.  Every
public function that aggregates therefore also accepts ``group`` and reports
per-group values, and :func:`bootstrap_ci` resamples **cows**, not rows.
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

from .labels import (
    EVENT_POSTPROCESS,
    MIN_COWS_FOR_CLAIM,
    MIN_TRUE_EVENTS_FOR_CLAIM,
    PHYSIOLOGICAL_RATE_PER_HOUR,
)

DECISION_STEP_SECONDS = 0.5
DEFAULT_TIOU_THRESHOLDS = (0.10, 0.25, 0.50)


# ==========================================================================
# layer 1 - frame level
# ==========================================================================
def multiclass_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, object]:
    """Per-class and macro metrics for a softmax head."""

    valid = np.asarray(target) >= 0
    if not np.any(valid):
        return {
            "evaluated": 0,
            "macro_f1": None,
            "mof": None,
            "per_class": [],
            "confusion_matrix": [],
        }
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
        "macro_f1": float(
            f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "mof": float(np.mean(actual == predicted)),
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


#: Historical name kept so old report readers and scripts do not break.
body_metrics = multiclass_metrics


def mof(target: np.ndarray, predicted: np.ndarray) -> float | None:
    """Mean-over-frames accuracy, the standard segmentation frame metric."""

    actual = np.asarray(target)
    valid = actual >= 0
    if not np.any(valid):
        return None
    return float(np.mean(actual[valid] == np.asarray(predicted)[valid]))


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
    physiological ceiling.  When no threshold satisfies the bound the constraint
    is dropped and the unconstrained optimum is returned, so this can never fail
    closed.  Vectorised through ``precision_recall_curve``: the pre-20260818
    implementation called ``f1_score`` once per candidate threshold, which cost
    minutes per task on a 76k-row split and returned the identical answer.
    """

    valid = np.asarray(mask).astype(bool)
    actual = np.asarray(target)[valid].astype(np.int8)
    scores = np.asarray(probability, dtype=np.float64)[valid]
    if actual.size == 0 or np.unique(actual).size < 2:
        return 0.5

    precision, recall, thresholds = precision_recall_curve(actual, scores)
    precision = precision[:-1]
    recall = recall[:-1]
    beta2 = float(beta) ** 2
    denominator = beta2 * precision + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        fbeta = np.where(denominator > 0, (1.0 + beta2) * precision * recall / denominator, 0.0)

    usable = np.ones(thresholds.shape, dtype=bool)
    if max_positive_rate is not None:
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
    chosen = tied[np.argmin(np.abs(thresholds[tied] - 0.5))]
    return float(thresholds[chosen])


def binary_point_metrics(
    target: np.ndarray, probability: np.ndarray, mask: np.ndarray, threshold: float
) -> dict[str, object]:
    """Frame metrics for one binary head, with an explicit evidence scope."""

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


# ==========================================================================
# interval construction
# ==========================================================================
def intervals_from_binary(
    times_ms: np.ndarray, values: np.ndarray, *, max_gap_ms: int = 750
) -> list[tuple[int, int]]:
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
    annotations: "object", device_mac: str, session_id: str, code: str
) -> list[tuple[int, int]]:
    """Read true event intervals straight from the adjudicated annotation table.

    Reconstructing them from 2 Hz label points splits one real event into
    several whenever the label series has a hole, which inflates the true-event
    count and depresses recall.
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


def postprocess_intervals(
    intervals: list[tuple[int, int]], code: str, *, enabled: bool = True
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


# ==========================================================================
# layer 2 - segment level
# ==========================================================================
def match_intervals(
    truth: list[tuple[int, int]],
    predicted: list[tuple[int, int]],
    *,
    tolerance_ms: int = 2500,
) -> tuple[int, int, int]:
    """Historical one-to-one matching with a fixed absolute tolerance."""

    candidates: list[tuple[float, int, int]] = []
    for truth_index, (truth_start, truth_end) in enumerate(truth):
        for pred_index, (pred_start, pred_end) in enumerate(predicted):
            overlap = min(truth_end + tolerance_ms, pred_end) - max(
                truth_start - tolerance_ms, pred_start
            )
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


def temporal_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    intersection = min(a[1], b[1]) - max(a[0], b[0])
    if intersection <= 0:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return float(intersection) / float(max(union, 1))


def match_intervals_tiou(
    truth: list[tuple[int, int]],
    predicted: list[tuple[int, int]],
    threshold: float,
) -> tuple[int, int, int, list[tuple[int, int]]]:
    """Greedy one-to-one matching by temporal IoU.

    Returns ``(tp, fp, fn, pairs)``.  This is the protocol behind ``F1@k`` in
    the action-segmentation literature, and unlike a fixed second tolerance it
    scales with the length of the behaviour, so a 1 s stand-up and a 35 s
    defecation are judged on the same footing.
    """

    scored: list[tuple[float, int, int]] = []
    for t_index, t_interval in enumerate(truth):
        for p_index, p_interval in enumerate(predicted):
            value = temporal_iou(t_interval, p_interval)
            if value >= threshold:
                scored.append((value, t_index, p_index))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_truth: set[int] = set()
    used_pred: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, t_index, p_index in scored:
        if t_index in used_truth or p_index in used_pred:
            continue
        used_truth.add(t_index)
        used_pred.add(p_index)
        pairs.append((t_index, p_index))
    tp = len(pairs)
    return tp, len(predicted) - tp, len(truth) - tp, pairs


def f1_at_tiou(
    truth: list[tuple[int, int]],
    predicted: list[tuple[int, int]],
    thresholds: tuple[float, ...] = DEFAULT_TIOU_THRESHOLDS,
) -> dict[str, object]:
    """``F1@k`` for each requested tIoU threshold."""

    out: dict[str, object] = {}
    for threshold in thresholds:
        tp, fp, fn, _ = match_intervals_tiou(truth, predicted, threshold)
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        if precision is None or recall is None or precision + recall == 0:
            f1: float | None = 0.0 if (tp + fp + fn) else None
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        out[f"f1@{int(round(threshold * 100)):02d}"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        }
    return out


def run_length_labels(sequence: np.ndarray) -> list[int]:
    """Collapse a frame label sequence into its run labels."""

    values = np.asarray(sequence).ravel()
    if values.size == 0:
        return []
    keep = np.concatenate(([True], values[1:] != values[:-1]))
    return [int(v) for v in values[keep]]


def levenshtein(left: list[int], right: list[int]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        current = [i]
        for j, b in enumerate(right, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def edit_score(truth_sequence: np.ndarray, predicted_sequence: np.ndarray) -> float:
    """Segmental edit score in ``[0, 100]``; 100 means no over-segmentation.

    Computed on the *run* sequences, so it is insensitive to how long each
    segment lasts and sensitive only to how many segments there are and in what
    order.  A model whose frame accuracy is fine but which chops every real
    event into three fragments scores badly here and nowhere else.
    """

    truth_runs = run_length_labels(truth_sequence)
    predicted_runs = run_length_labels(predicted_sequence)
    if not truth_runs and not predicted_runs:
        return 100.0
    distance = levenshtein(truth_runs, predicted_runs)
    denominator = max(len(truth_runs), len(predicted_runs), 1)
    return float((1.0 - distance / denominator) * 100.0)


# ==========================================================================
# layer 3 - deployment level
# ==========================================================================
def localisation_error(
    truth: list[tuple[int, int]],
    predicted: list[tuple[int, int]],
    pairs: list[tuple[int, int]],
) -> dict[str, float | None]:
    """Onset error of matched events, in milliseconds."""

    if not pairs:
        return {"median_ms": None, "p90_ms": None, "matched": 0}
    deltas = np.asarray(
        [abs(int(predicted[p][0]) - int(truth[t][0])) for t, p in pairs], dtype=np.float64
    )
    return {
        "median_ms": float(np.median(deltas)),
        "p90_ms": float(np.percentile(deltas, 90)),
        "matched": int(deltas.size),
    }


def rate_plausibility(code: str, predicted_events: int, labelled_hours: float) -> dict[str, object]:
    """Compare the predicted event rate with a published physiological bound."""

    bound = PHYSIOLOGICAL_RATE_PER_HOUR.get(str(code).upper())
    if bound is None or labelled_hours <= 0:
        return {
            "bound_per_hour": bound,
            "observed_per_hour": None,
            "ratio": None,
            "verdict": "unknown",
        }
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


# ==========================================================================
# layer 4 - generalisation
# ==========================================================================
def bootstrap_ci(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    statistic=np.mean,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 20260819,
) -> dict[str, float | None]:
    """Confidence interval obtained by resampling **groups**, not rows.

    Resampling rows would treat 246,487 samples from one animal as 246,487
    independent observations and produce an interval an order of magnitude too
    narrow.  The unit of independence in this project is the cow.
    """

    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    finite = np.isfinite(values)
    values, groups = values[finite], groups[finite]
    unique = np.unique(groups)
    if unique.size == 0:
        return {"point": None, "low": None, "high": None, "groups": 0}
    per_group = np.asarray([statistic(values[groups == g]) for g in unique], dtype=np.float64)
    point = float(statistic(per_group))
    if unique.size < 2:
        return {"point": point, "low": None, "high": None, "groups": int(unique.size)}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, unique.size, size=(int(n_resamples), unique.size))
    samples = np.asarray([float(statistic(per_group[row])) for row in draws])
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "point": point,
        "low": float(np.quantile(samples, alpha)),
        "high": float(np.quantile(samples, 1.0 - alpha)),
        "groups": int(unique.size),
    }


# ==========================================================================
# the event report
# ==========================================================================
def event_level_metrics(
    prediction_frame: "object",
    code: str,
    threshold: float,
    *,
    tolerance_ms: int = 2500,
    annotations: "object" = None,
    postprocess: bool = True,
    tiou_thresholds: tuple[float, ...] = DEFAULT_TIOU_THRESHOLDS,
    min_true_events: int = MIN_TRUE_EVENTS_FOR_CLAIM,
    min_cows: int = MIN_COWS_FOR_CLAIM,
) -> dict[str, object]:
    """Event report for one code: layers 2, 3 and 4 in a single pass.

    ``prediction_frame`` is duck-typed: it must contain ``center_time_ms``,
    ``target_{code}``, ``mask_{code}``, ``prob_{code}``, ``device_mac`` and
    ``session_id``; ``cow_id`` is used to produce the per-cow breakdown and the
    bootstrap interval, and its absence downgrades the evidence level rather
    than silently pooling.
    """

    target_column = f"target_{code}"
    mask_column = f"mask_{code}"
    probability_column = f"prob_{code}"

    total_true = total_predicted = total_tp = 0
    evaluated_points = trusted_negative_points = 0
    tiou_counts = {t: [0, 0, 0] for t in tiou_thresholds}
    edit_scores: list[float] = []
    onset_deltas: list[float] = []
    per_cow: dict[str, dict[str, float]] = {}
    cows_with_truth: set[str] = set()
    details: list[dict[str, object]] = []

    for (device, session), group in prediction_frame.groupby(
        ["device_mac", "session_id"], sort=True
    ):
        ordered = group.sort_values("center_time_ms", kind="stable")
        valid = ordered[mask_column].to_numpy().astype(bool)
        if not np.any(valid):
            continue
        times = ordered.loc[valid, "center_time_ms"].to_numpy(np.int64)
        actual = ordered.loc[valid, target_column].to_numpy(np.uint8)
        predicted = ordered.loc[valid, probability_column].to_numpy(float) >= threshold
        cow = str(ordered["cow_id"].iloc[0]) if "cow_id" in ordered.columns else "unknown"

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
        evaluated_points += int(actual.size)
        trusted_negative_points += int(np.sum(actual == 0))

        for tiou in tiou_thresholds:
            k_tp, k_fp, k_fn, pairs = match_intervals_tiou(
                truth_intervals, predicted_intervals, tiou
            )
            tiou_counts[tiou][0] += k_tp
            tiou_counts[tiou][1] += k_fp
            tiou_counts[tiou][2] += k_fn
            if tiou == min(tiou_thresholds) and pairs:
                onset_deltas.extend(
                    abs(int(predicted_intervals[p][0]) - int(truth_intervals[t][0]))
                    for t, p in pairs
                )
        edit_scores.append(edit_score(actual, predicted.astype(np.uint8)))

        record = per_cow.setdefault(
            cow,
            {"true_events": 0, "predicted_events": 0, "true_positive": 0, "labelled_hours": 0.0},
        )
        record["true_events"] += len(truth_intervals)
        record["predicted_events"] += len(predicted_intervals)
        record["true_positive"] += tp
        record["labelled_hours"] += float(actual.size) * DECISION_STEP_SECONDS / 3600.0

        if truth_intervals:
            cows_with_truth.add(cow)
        if truth_intervals or predicted_intervals:
            details.append(
                {
                    "device_mac": str(device),
                    "session_id": str(session),
                    "cow_id": cow,
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
    labelled_hours = evaluated_points * DECISION_STEP_SECONDS / 3600.0
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

    tiou_report: dict[str, object] = {}
    for tiou, (tp, fp, fn) in tiou_counts.items():
        p = tp / (tp + fp) if (tp + fp) else None
        r = tp / (tp + fn) if (tp + fn) else None
        value = (
            2.0 * p * r / (p + r)
            if p is not None and r is not None and p + r > 0
            else (0.0 if (tp + fp + fn) else None)
        )
        tiou_report[f"f1@{int(round(tiou * 100)):02d}"] = {
            "precision": p,
            "recall": r,
            "f1": value,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        }

    cow_f1: list[float] = []
    cow_ids: list[str] = []
    for cow, record in per_cow.items():
        if record["true_events"] <= 0:
            continue
        p = (
            record["true_positive"] / record["predicted_events"]
            if record["predicted_events"]
            else 0.0
        )
        r = record["true_positive"] / record["true_events"]
        cow_f1.append(2.0 * p * r / (p + r) if p + r > 0 else 0.0)
        cow_ids.append(cow)
        record["precision"] = p
        record["recall"] = r
        record["f1"] = cow_f1[-1]
        record["false_alarms_per_24h"] = (
            (record["predicted_events"] - record["true_positive"])
            / record["labelled_hours"]
            * 24.0
            if record["labelled_hours"] > 0
            else None
        )

    enough_events = total_true >= int(min_true_events)
    enough_cows = len(cows_with_truth) >= int(min_cows)
    claimable = bool(scope == "positive_and_trusted_negative" and enough_events and enough_cows)

    return {
        "scope": scope,
        "evidence_level": "reportable" if claimable else "not_evaluable",
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
        # layer 2
        "segment_level": tiou_report,
        "edit_score": float(np.mean(edit_scores)) if edit_scores else None,
        # layer 3
        "false_alarms_per_labelled_hour": false_alarms_per_hour,
        "false_alarms_per_cow_per_24h": (
            false_alarms_per_hour * 24.0 if false_alarms_per_hour is not None else None
        ),
        "onset_error_ms": {
            "median": float(np.median(onset_deltas)) if onset_deltas else None,
            "p90": float(np.percentile(onset_deltas, 90)) if onset_deltas else None,
            "matched": len(onset_deltas),
        },
        # layer 4
        "per_cow": per_cow,
        "cow_f1_bootstrap": (
            bootstrap_ci(np.asarray(cow_f1), np.asarray(cow_ids))
            if cow_f1
            else {"point": None, "low": None, "high": None, "groups": 0}
        ),
        "rate_plausibility": rate_plausibility(code, total_predicted, labelled_hours),
        "session_details": details,
        # deprecated alias kept so existing report readers do not break
        "false_alarms_per_evaluated_hour": false_alarms_per_hour,
    }


# ==========================================================================
# layer 5 - day level
# ==========================================================================
def day_level_metrics(
    truth_days: np.ndarray,
    predicted_days: np.ndarray,
    *,
    lead_time_hours: np.ndarray | None = None,
    cow_ids: np.ndarray | None = None,
) -> dict[str, object]:
    """Sensitivity / specificity / PPV / lead time for oestrus or calving.

    One row is one cow-day.  ``lead_time_hours`` is the interval between the
    alert and the reference event, defined only for true positives; its median
    and quartiles are what a farm actually buys, because an alert that arrives
    after the fact has a sensitivity of 1 and a value of 0.
    """

    truth = np.asarray(truth_days).astype(bool)
    predicted = np.asarray(predicted_days).astype(bool)
    if truth.size != predicted.size:
        raise ValueError("truth and prediction must have the same number of cow-days")
    tp = int(np.sum(truth & predicted))
    fp = int(np.sum(~truth & predicted))
    fn = int(np.sum(truth & ~predicted))
    tn = int(np.sum(~truth & ~predicted))
    sensitivity = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    ppv = tp / (tp + fp) if (tp + fp) else None

    lead: dict[str, float | None] = {"median": None, "q25": None, "q75": None, "n": 0}
    if lead_time_hours is not None:
        values = np.asarray(lead_time_hours, dtype=np.float64)[truth & predicted]
        values = values[np.isfinite(values)]
        if values.size:
            lead = {
                "median": float(np.median(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
                "n": int(values.size),
            }

    per_cow: dict[str, object] = {}
    weekly_false_alarms: dict[str, float] = {}
    if cow_ids is not None:
        ids = np.asarray(cow_ids)
        for cow in np.unique(ids):
            selected = ids == cow
            cow_days = int(selected.sum())
            cow_fp = int(np.sum(~truth[selected] & predicted[selected]))
            per_cow[str(cow)] = {
                "cow_days": cow_days,
                "true_positive": int(np.sum(truth[selected] & predicted[selected])),
                "false_positive": cow_fp,
                "false_negative": int(np.sum(truth[selected] & ~predicted[selected])),
            }
            weekly_false_alarms[str(cow)] = cow_fp / cow_days * 7.0 if cow_days else 0.0

    return {
        "cow_days": int(truth.size),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "lead_time_hours": lead,
        "false_alarms_per_cow_per_week": (
            float(np.mean(list(weekly_false_alarms.values()))) if weekly_false_alarms else None
        ),
        "per_cow": per_cow,
    }


# ==========================================================================
# model selection
# ==========================================================================
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

    The pre-20260819 deep training script averaged eight thresholded F1 values
    with equal weight, so four tail-wagging annotations from one cow counted as
    much as a hundred thousand posture labels.  This version weights tasks by
    how much evidence supports them and uses average precision, which needs no
    threshold and therefore has far lower variance.  It is now the objective
    every trainer calls; the old equal-weight version is gone rather than merely
    deprecated, because having two disagreeing definitions in one repository was
    the actual defect.
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
            "event_mean_average_precision": (
                float(np.mean(list(usable_events.values()))) if usable_events else None
            ),
        },
        "events_used": sorted(usable_events),
    }
