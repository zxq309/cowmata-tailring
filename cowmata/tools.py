"""Dataset diagnostics, integrity checks and the human-review queue.

These were three separate scripts in the 20260818 baseline
(``diagnose_dataset.py``, ``verify_dataset.py``, ``mine_candidates.py``), each
with its own argument parser and its own copy of the label list.  They are
library functions here and the scripts became four-line ``main()`` shims, so the
CLI can call them directly instead of launching a subprocess it cannot pass an
object to and whose traceback it loses.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cache import BYTES_PER_FRAME_V1, BYTES_PER_FRAME_V2, estimate_storage_bytes, open_cache
from .labels import (
    ALL_ANNOTATION_CODES,
    DEPRECATED_EVENT_CODES,
    EVENT_CODES,
    MIN_COWS_FOR_CLAIM,
    MIN_TRUE_EVENTS_FOR_CLAIM,
    PHYSIOLOGICAL_RATE_PER_HOUR,
)

SAMPLE_RATE_HZ = 50.0
EXCRETION_CODES = ("URINATION", "DEFECATION")
REVIEW_COLUMNS = ("review_decision", "reviewer", "reviewed_at", "notes")
POINT_MS = 500


def _read_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None or not Path(path).exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


def _read_json(path: Path | None) -> Any:
    if path is None or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ==========================================================================
# diagnostics
# ==========================================================================
def segment_report(cache_root: Path | None) -> dict[str, object]:
    per_session: dict[str, list[dict[str, int]]] = {}
    if cache_root is not None and Path(cache_root).exists():
        for directory in sorted(Path(cache_root).iterdir()):
            if not directory.is_dir():
                continue
            try:
                cache = open_cache(directory)
            except Exception:
                continue
            per_session[directory.name] = [s.to_dict() for s in cache.segments]
    if not per_session:
        return {"available": False, "reason": "no session cache found"}
    counts = Counter(len(segments) for segments in per_session.values())
    lengths = [
        int(s["stop_index"]) - int(s["start_index"])
        for segments in per_session.values()
        for s in segments
    ]
    array = np.asarray(lengths, dtype=np.float64) if lengths else np.zeros(1)
    return {
        "available": True,
        "sessions": len(per_session),
        "segments_total": int(sum(int(k) * int(v) for k, v in counts.items())),
        "segments_per_session_histogram": {int(k): int(v) for k, v in sorted(counts.items())},
        "sessions_with_one_segment": int(counts.get(1, 0)),
        "sessions_multi_segment": int(sum(v for k, v in counts.items() if k > 1)),
        "segment_seconds": {
            "min": float(array.min() / SAMPLE_RATE_HZ),
            "median": float(np.median(array) / SAMPLE_RATE_HZ),
            "mean": float(array.mean() / SAMPLE_RATE_HZ),
            "max": float(array.max() / SAMPLE_RATE_HZ),
        },
        "_per_session": per_session,
    }


def annotation_report(annotations: pd.DataFrame | None) -> dict[str, object]:
    if annotations is None:
        return {"available": False, "reason": "annotation table not found"}
    frame = annotations.copy()
    frame["duration_ms"] = frame["t_end_rel_ms"].astype(np.int64) - frame[
        "t_start_rel_ms"
    ].astype(np.int64)
    unknown = sorted(set(frame["code"].astype(str)) - set(ALL_ANNOTATION_CODES))
    per_code = []
    for code, group in frame.groupby("code"):
        per_code.append(
            {
                "code": str(code),
                "intervals": int(len(group)),
                "cows": int(group["cow_id"].nunique()) if "cow_id" in group else None,
                "sessions": int(group["session_id"].nunique()) if "session_id" in group else None,
                "duration_s_mean": float(group["duration_ms"].mean() / 1000.0),
                "duration_s_median": float(group["duration_ms"].median() / 1000.0),
                "duration_s_max": float(group["duration_ms"].max() / 1000.0),
                "total_minutes": float(group["duration_ms"].sum() / 60000.0),
                "trained": str(code) in EVENT_CODES,
                "deprecated": str(code) in DEPRECATED_EVENT_CODES,
            }
        )

    conflicts = []
    tail = frame[frame["code"] == "TAIL_RAISED"]
    for code in EXCRETION_CODES:
        subset = frame[frame["code"] == code]
        overlapping = 0
        for _, row in subset.iterrows():
            same = tail[
                (tail["device_mac"].astype(str) == str(row["device_mac"]))
                & (tail["session_id"].astype(str) == str(row["session_id"]))
                & (tail["t_start_rel_ms"] < row["t_end_rel_ms"])
                & (tail["t_end_rel_ms"] > row["t_start_rel_ms"])
            ]
            if len(same):
                overlapping += 1
        conflicts.append(
            {
                "code": code,
                "intervals": int(len(subset)),
                "also_annotated_tail_raised": int(overlapping),
                "would_be_negative_under_legacy_policy": int(len(subset) - overlapping),
            }
        )

    evaluability = []
    for code in EVENT_CODES:
        group = frame[frame["code"] == code]
        cows = int(group["cow_id"].nunique()) if "cow_id" in group and len(group) else 0
        evaluability.append(
            {
                "code": code,
                "intervals": int(len(group)),
                "cows": cows,
                "status": (
                    "reportable"
                    if (len(group) >= MIN_TRUE_EVENTS_FOR_CLAIM and cows >= MIN_COWS_FOR_CLAIM)
                    else "not_evaluable"
                ),
            }
        )
    return {
        "available": True,
        "rows": int(len(frame)),
        "unknown_codes": unknown,
        "per_code": per_code,
        "tail_conflicts": conflicts,
        "evaluability": evaluability,
    }


def cow_balance_report(samples: pd.DataFrame | None) -> dict[str, object]:
    """How concentrated the supervised data is in a handful of animals.

    This is the number that decides whether a pooled cross-cow metric means
    anything.  In the 20260818 dataset one animal held 71.6% of all supervised
    samples, so a pooled figure was effectively that animal's score.
    """

    if samples is None or "cow_id" not in samples.columns:
        return {"available": False, "reason": "sample table not found"}
    counts = samples["cow_id"].astype(str).value_counts()
    total = int(counts.sum())
    share = (counts / max(total, 1)).to_dict()
    ordered = sorted(share.items(), key=lambda item: -item[1])
    cumulative = np.cumsum([value for _, value in ordered])
    effective = float(1.0 / np.sum(np.square(list(share.values())))) if total else 0.0
    return {
        "available": True,
        "cows": int(counts.size),
        "samples": total,
        "per_cow": {cow: {"samples": int(counts[cow]), "share": float(share[cow])} for cow in counts.index},
        "largest_share": float(ordered[0][1]) if ordered else None,
        "cows_for_80_percent": int(np.searchsorted(cumulative, 0.8) + 1) if total else 0,
        "effective_cow_count": effective,
        "note": (
            "effective_cow_count is the inverse Simpson index of the per-cow sample "
            "shares: the number of equally-sized animals this dataset is worth."
        ),
    }


def coverage_report(
    coverage: pd.DataFrame | None, samples: pd.DataFrame | None
) -> dict[str, object]:
    if coverage is None or len(coverage) == 0:
        hours, covered_sessions, detail = 0.0, 0, []
    else:
        frame = coverage.copy()
        flag = frame.get("exhaustive_reviewed")
        if flag is not None:
            keep = flag.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})
            frame = frame[keep]
        frame["hours"] = (
            frame["t_end_rel_ms"].astype(np.int64) - frame["t_start_rel_ms"].astype(np.int64)
        ) / 3600000.0
        hours = float(frame["hours"].sum())
        covered_sessions = int(frame.groupby(["device_mac", "session_id"]).ngroups) if len(frame) else 0
        detail = [
            {"event_code": str(code), "hours": float(group["hours"].sum()), "rows": int(len(group))}
            for code, group in frame.groupby("event_code")
        ]
    total_sessions = (
        int(samples.groupby(["device_mac", "session_id"]).ngroups) if samples is not None else None
    )
    return {
        "exhaustively_reviewed_hours": hours,
        "covered_sessions": covered_sessions,
        "total_sessions": total_sessions,
        "per_code": detail,
        "verdict": (
            "no exhaustive review coverage: event precision and false-alarm rates are NOT claimable"
            if hours <= 0
            else "partial coverage: restrict precision claims to covered ranges"
        ),
    }


def fold_report(splits: object) -> dict[str, object]:
    if not isinstance(splits, dict) or "folds" not in splits:
        return {"available": False, "reason": "split manifest not found or malformed"}
    rows = []
    for fold in splits["folds"]:
        counts = fold.get("counts", {})
        test = counts.get("test", {})
        rows.append(
            {
                "fold": fold.get("fold"),
                "test_cow": fold.get("test_cow") or fold.get("test_cows"),
                "train_sessions": len(fold.get("train_sessions", [])),
                "validation_sessions": len(fold.get("validation_sessions", [])),
                "test_sessions": len(fold.get("test_sessions", [])),
                "test_samples": test.get("samples"),
            }
        )
    usable = [row for row in rows if (row["test_samples"] or 0) >= 1000]
    return {
        "available": True,
        "protocol": splits.get("protocol"),
        "folds": rows,
        "folds_total": len(rows),
        "folds_with_usable_test": len(usable),
        "note": "folds with <1000 test samples produce noise, not evidence; use pooled results",
    }


def diagnose(root: str | Path, *, out: str | Path | None = None) -> dict[str, object]:
    """Full dataset diagnostic. Writes JSON and Markdown when ``out`` is given."""

    root = Path(root)
    data = root / "datasets" / "cowmata_imu"
    paths = {
        "annotations": data / "annotations" / "annotations_adjudicated_minimal.csv",
        "samples": data / "supervised_cache" / "samples.csv",
        "session_cache": data / "supervised_cache" / "session_cache",
        "review_coverage": data / "annotations" / "review_coverage_template.csv",
        "splits": data / "loco_splits" / "loco_splits.json",
    }
    annotations = _read_csv(paths["annotations"])
    samples = _read_csv(paths["samples"])
    coverage = _read_csv(paths["review_coverage"])
    splits = _read_json(paths["splits"])

    segments = segment_report(paths["session_cache"])
    segments.pop("_per_session", None)
    payload = {
        "inputs": {key: str(value) for key, value in paths.items()},
        "segments": segments,
        "annotations": annotation_report(annotations),
        "cow_balance": cow_balance_report(samples),
        "coverage": coverage_report(coverage, samples),
        "folds": fold_report(splits),
    }
    if out is not None:
        target = Path(out)
        target.mkdir(parents=True, exist_ok=True)
        (target / "diagnostics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
        )
        (target / "diagnostics.md").write_text(render_diagnostics(payload), encoding="utf-8")
    return payload


def render_diagnostics(payload: dict[str, object]) -> str:
    lines: list[str] = ["# Dataset diagnostic report", ""]
    balance = payload["cow_balance"]
    lines.append("## Per-cow concentration")
    if not balance.get("available"):
        lines.append(f"- unavailable: {balance.get('reason')}")
    else:
        lines.append(
            f"- {balance['cows']} cows, {balance['samples']} supervised samples; "
            f"largest single share {balance['largest_share']:.1%}; "
            f"effective cow count {balance['effective_cow_count']:.2f}"
        )
        lines.append("")
        lines.append("| Cow | Samples | Share |")
        lines.append("|---|---:|---:|")
        for cow, item in balance["per_cow"].items():
            lines.append(f"| {cow} | {item['samples']} | {item['share']:.2%} |")
    lines.append("")

    annotations = payload["annotations"]
    lines.append("## Annotation audit")
    if not annotations.get("available"):
        lines.append(f"- unavailable: {annotations.get('reason')}")
    else:
        lines.append("| Code | Intervals | Cows | Sessions | Median (s) | Trained |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for item in annotations["per_code"]:
            lines.append(
                f"| {item['code']} | {item['intervals']} | {item['cows']} | "
                f"{item['sessions']} | {item['duration_s_median']:.1f} | "
                f"{'yes' if item['trained'] else 'no'} |"
            )
        lines.append("")
        lines.append("### Tail-raise label conflicts")
        lines.append("| Code | Intervals | Also tail-raised | Negative under legacy policy |")
        lines.append("|---|---:|---:|---:|")
        for item in annotations["tail_conflicts"]:
            lines.append(
                f"| {item['code']} | {item['intervals']} | "
                f"{item['also_annotated_tail_raised']} | "
                f"{item['would_be_negative_under_legacy_policy']} |"
            )
    lines.append("")

    coverage = payload["coverage"]
    lines.append("## Exhaustive-review coverage")
    lines.append(f"- reviewed duration: {coverage['exhaustively_reviewed_hours']:.2f} h")
    lines.append(
        f"- covered sessions: {coverage['covered_sessions']} / {coverage['total_sessions']}"
    )
    lines.append(f"- verdict: {coverage['verdict']}")
    return "\n".join(lines) + "\n"


# ==========================================================================
# verification
# ==========================================================================
def verify(root: str | Path, *, full_cache_scan: bool = False) -> dict[str, object]:
    """Structural preflight over annotations, sessions, caches and splits."""

    root = Path(root)
    data = root / "datasets" / "cowmata_imu"
    problems: list[str] = []
    warnings: list[str] = []

    annotations = _read_csv(data / "annotations" / "annotations_adjudicated_minimal.csv")
    sessions = _read_csv(data / "supervised_cache" / "sessions.csv")
    samples = _read_csv(data / "supervised_cache" / "samples.csv")
    cache_root = data / "supervised_cache" / "session_cache"

    if annotations is None:
        problems.append("annotation table missing")
    else:
        if "event_id" in annotations and annotations["event_id"].duplicated().any():
            problems.append("duplicate annotation event_id")
        if (annotations["t_end_rel_ms"] <= annotations["t_start_rel_ms"]).any():
            problems.append("non-positive annotation interval")
        unknown = sorted(set(annotations["code"].astype(str)) - set(ALL_ANNOTATION_CODES))
        if unknown:
            problems.append(f"unsupported annotation codes: {unknown}")
        deprecated = [
            code for code in DEPRECATED_EVENT_CODES if (annotations["code"] == code).any()
        ]
        if deprecated:
            warnings.append(
                f"deprecated codes present and readable but excluded from training: {deprecated}"
            )

    if samples is not None:
        required = {"cow_id", "cache_key", "center_index", "body_target"}
        required.update(f"event_{code}" for code in EVENT_CODES)
        required.update(f"mask_{code}" for code in EVENT_CODES)
        missing = sorted(required - set(samples.columns))
        if missing:
            problems.append(f"sample table missing columns: {missing[:8]}")
        if "cow_id" in samples and (samples["cow_id"].astype(str) == "unknown").any():
            warnings.append("some supervised rows have cow_id=unknown; exclude them from splits")

    checked = 0
    absent = 0
    if sessions is not None:
        for item in sessions.to_dict("records"):
            directory = cache_root / str(item.get("cache_key", ""))
            if not directory.exists():
                # The supervised cache is deliberately outside Git.  Listing all
                # 132 missing directories buries any real problem, so it is one
                # counted line.
                absent += 1
                continue
            if full_cache_scan or checked < 5:
                try:
                    cache = open_cache(directory)
                    block = cache.physical(0, min(cache.n_frames, 100_000))
                    if not np.isfinite(block).all():
                        problems.append(f"non-finite cache values: {directory.name}")
                except Exception as error:
                    problems.append(f"cache unreadable ({type(error).__name__}): {directory.name}")
                checked += 1
    if absent:
        warnings.append(
            f"{absent} of {0 if sessions is None else len(sessions)} session caches are not "
            "present locally; recover them per docs/DATA_ACCESS.md before retraining"
        )

    split_reports: list[dict[str, object]] = []
    for path in (
        data / "loco_splits" / "loco_splits.json",
        data / "development_split" / "development_all.json",
    ):
        manifest = _read_json(path)
        if manifest is None:
            warnings.append(f"split manifest not found: {path.name}")
            continue
        for fold in manifest["folds"]:
            train = set(fold.get("train_sessions", []))
            validation = set(fold.get("validation_sessions", []))
            test = set(fold.get("test_sessions", []))
            if train & validation or train & test or validation & test:
                problems.append(f"split session overlap in {path.name}, fold {fold.get('fold')}")
        split_reports.append(
            {"path": str(path), "protocol": manifest.get("protocol"), "folds": len(manifest["folds"])}
        )

    return {
        "status": "PASS" if not problems else "FAIL",
        "root": str(root),
        "annotations": int(len(annotations)) if annotations is not None else 0,
        "sessions": int(len(sessions)) if sessions is not None else 0,
        "samples": int(len(samples)) if samples is not None else 0,
        "caches_checked": checked,
        "caches_absent": absent,
        "split_reports": split_reports,
        "warnings": warnings,
        "problems": problems,
    }


# ==========================================================================
# storage planning
# ==========================================================================
def plan_storage(cows: int, days: float) -> dict[str, object]:
    """Compare schema 1 and schema 2 footprints for a planned collection."""

    new = estimate_storage_bytes(cows, days, schema=2)
    old = estimate_storage_bytes(cows, days, schema=1)
    return {
        "cows": int(cows),
        "days_per_cow": float(days),
        "cow_days": float(cows) * float(days),
        "hours": float(cows) * float(days) * 24.0,
        "frames": new["frames"],
        "schema1_gigabytes": old["gigabytes"],
        "schema2_gigabytes": new["gigabytes"],
        "saving_ratio": old["gigabytes"] / max(new["gigabytes"], 1e-9),
        "bytes_per_frame": {"schema1": BYTES_PER_FRAME_V1, "schema2": BYTES_PER_FRAME_V2},
    }


# ==========================================================================
# candidate mining
# ==========================================================================
def _merge_points(
    times_ms: np.ndarray, scores: np.ndarray, selected: np.ndarray, *, max_gap_ms: int, pad_ms: int
) -> list[dict[str, object]]:
    indices = np.flatnonzero(selected)
    if indices.size == 0:
        return []
    out: list[dict[str, object]] = []
    start = previous = int(indices[0])
    for index in indices[1:]:
        if times_ms[index] - times_ms[previous] > max_gap_ms:
            out.append(_interval(times_ms, scores, start, previous, pad_ms))
            start = int(index)
        previous = int(index)
    out.append(_interval(times_ms, scores, start, previous, pad_ms))
    return out


def _interval(times_ms, scores, start, stop, pad_ms) -> dict[str, object]:
    window = slice(int(start), int(stop) + 1)
    peak = int(start) + int(np.argmax(scores[window]))
    return {
        "t_start_rel_ms": int(times_ms[start]) - pad_ms,
        "t_end_rel_ms": int(times_ms[stop]) + POINT_MS + pad_ms,
        "peak_time_ms": int(times_ms[peak]),
        "max_score": float(scores[window].max()),
        "mean_score": float(scores[window].mean()),
        "points": int(stop - start + 1),
    }


def mine_candidates(
    predictions: pd.DataFrame,
    events: list[str],
    *,
    per_event: int = 40,
    random_fraction: float = 0.3,
    top_fraction: float = 0.4,
    boundary_fraction: float = 0.4,
    candidate_quantile: float = 0.995,
    merge_gap_ms: int = 5000,
    pad_ms: int = 5000,
    daily_cap_multiplier: float = 2.0,
    seed: int = 20260819,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build the human review queue for the annotation bootstrap loop.

    Four rules are enforced in code because they are what makes the loop
    converge instead of drift:

    1. a random control group is always included, because precision and recall
       estimated on model-selected candidates are meaningless;
    2. selection mixes exploitation, uncertainty and diversity;
    3. a per-cow-per-session cap derived from published excretion rates keeps
       one noisy session from eating the whole budget;
    4. nothing here is a label - every row leaves with ``review_decision`` empty.
    """

    if random_fraction < 0.1:
        raise ValueError(
            "random_fraction below 0.1 is refused: without a random control group no "
            "unbiased precision or recall can be estimated from this round"
        )
    frame = predictions.copy()
    if "cow_id" not in frame.columns:
        frame["cow_id"] = ""
    rng = np.random.default_rng(seed)
    queues: list[pd.DataFrame] = []
    summary: list[dict[str, object]] = []

    for event in [str(code).strip().upper() for code in events]:
        column = f"prob_{event}"
        if column not in frame.columns:
            continue
        usable = frame[np.isfinite(frame[column].to_numpy(dtype=float))]
        rows: list[dict[str, object]] = []
        for (device, session), group in usable.groupby(["device_mac", "session_id"], sort=True):
            ordered = group.sort_values("center_time_ms", kind="stable")
            times = ordered["center_time_ms"].to_numpy(np.int64)
            scores = ordered[column].to_numpy(float)
            if scores.size < 3:
                continue
            cut = float(np.quantile(scores, candidate_quantile))
            for interval in _merge_points(
                times, scores, scores >= cut, max_gap_ms=merge_gap_ms, pad_ms=pad_ms
            ):
                rows.append(
                    {
                        "device_mac": str(device),
                        "session_id": str(session),
                        "cow_id": str(ordered["cow_id"].iloc[0]),
                        "event_code": event,
                        "session_cut_score": cut,
                        **interval,
                    }
                )
        candidates = pd.DataFrame(rows)
        if candidates.empty:
            continue
        bound = PHYSIOLOGICAL_RATE_PER_HOUR.get(event)
        if bound is not None:
            limit = max(1, int(round(bound * 24.0 * daily_cap_multiplier)))
            candidates = (
                candidates.sort_values("max_score", ascending=False, kind="stable")
                .groupby(["cow_id", "session_id"], sort=False, group_keys=False)
                .head(limit)
            )
        candidates = candidates.reset_index(drop=True)

        budget = int(per_event)
        n_top = int(round(budget * top_fraction))
        n_boundary = int(round(budget * boundary_fraction))
        n_diverse = max(0, budget - n_top - n_boundary)
        pool = candidates.copy()
        top = pool.sort_values("max_score", ascending=False, kind="stable").head(n_top)
        pool = pool.drop(index=top.index)
        if len(pool):
            reference = float(np.median(candidates["session_cut_score"]))
            pool = pool.assign(_uncertainty=(pool["max_score"] - reference).abs())
            boundary = pool.sort_values("_uncertainty", kind="stable").head(n_boundary)
            pool = pool.drop(index=boundary.index).drop(columns=["_uncertainty"])
            boundary = boundary.drop(columns=["_uncertainty"])
        else:
            boundary = pool
        diverse = _diverse_subset(pool, n_diverse) if len(pool) else pool

        for block, name in (
            (top, "active_top"),
            (boundary, "active_boundary"),
            (diverse, "active_diverse"),
        ):
            if len(block):
                queues.append(block.assign(selection_strategy=name, sampling_source="active"))

        n_control = int(round(budget * random_fraction))
        control = pd.DataFrame()
        if n_control > 0 and len(usable):
            picked = usable.sample(
                n=min(n_control, len(usable)), random_state=int(rng.integers(1 << 31))
            )
            control = pd.DataFrame(
                {
                    "device_mac": picked["device_mac"].astype(str).to_numpy(),
                    "session_id": picked["session_id"].astype(str).to_numpy(),
                    "cow_id": picked["cow_id"].astype(str).to_numpy(),
                    "event_code": event,
                    "t_start_rel_ms": picked["center_time_ms"].to_numpy(np.int64) - pad_ms,
                    "t_end_rel_ms": picked["center_time_ms"].to_numpy(np.int64)
                    + POINT_MS
                    + pad_ms,
                    "peak_time_ms": picked["center_time_ms"].to_numpy(np.int64),
                    "max_score": picked[column].to_numpy(float),
                    "mean_score": picked[column].to_numpy(float),
                    "points": 1,
                    "selection_strategy": "random_control",
                    "sampling_source": "random",
                }
            )
            queues.append(control)
        summary.append(
            {
                "event_code": event,
                "candidates_after_cap": int(len(candidates)),
                "active_selected": int(len(top) + len(boundary) + len(diverse)),
                "random_control": int(len(control)),
                "cows_covered": int(candidates["cow_id"].nunique()),
            }
        )

    if not queues:
        raise ValueError("nothing to review; check the prediction frame and the event codes")
    queue = pd.concat(queues, ignore_index=True)
    queue.insert(0, "candidate_id", [f"CND-{i:05d}" for i in range(1, len(queue) + 1)])
    for column in REVIEW_COLUMNS:
        queue[column] = ""
    queue["duration_s"] = (queue["t_end_rel_ms"] - queue["t_start_rel_ms"]) / 1000.0
    manifest = {
        "per_event_budget": per_event,
        "fractions": {
            "top": top_fraction,
            "boundary": boundary_fraction,
            "diverse": round(1.0 - top_fraction - boundary_fraction, 4),
            "random_control": random_fraction,
        },
        "events": summary,
        "rows": int(len(queue)),
        "rules": [
            "only rows with sampling_source=random may be used to estimate precision/recall",
            "review_decision must be filled by a human before any row enters the truth table",
            "keep this file; the next round's report must cite the round it came from",
        ],
    }
    return queue, manifest


def _diverse_subset(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    """Greedy spread over cows, then score."""

    if len(frame) <= count:
        return frame
    picked: list[int] = []
    remaining = frame.copy()
    remaining["_cow_seen"] = 0
    while len(picked) < count and len(remaining):
        remaining = remaining.sort_values(
            ["_cow_seen", "max_score"], ascending=[True, False], kind="stable"
        )
        row = remaining.iloc[0]
        picked.append(int(row.name))
        remaining = remaining.drop(index=row.name)
        same_cow = remaining["cow_id"] == row["cow_id"]
        remaining.loc[same_cow, "_cow_seen"] = remaining.loc[same_cow, "_cow_seen"] + 1
    return frame.loc[picked]
