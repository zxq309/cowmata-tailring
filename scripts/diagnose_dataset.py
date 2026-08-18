# -*- coding: utf-8 -*-
"""Dataset diagnostics. Run this first; it needs no GPU and no torch.

It answers, with numbers instead of guesses:

A. Segment distribution - how many contiguous runs each session is cut into.
   This decides whether the segment-boundary contamination bug materially
   affected the previous models or not.
B. Contamination estimate - how many training samples had a context window
   that reached into the previous segment under the old code.
C. Annotation audit - counts per code and per cow, duration statistics,
   and the URINATION/DEFECATION vs TAIL_RAISED overlap conflict.
D. Review coverage - exhaustively reviewed hours per session and per code,
   i.e. how much of the data can legitimately support a precision claim.
E. Fold balance - test sessions and samples per LOCO fold.

Usage
-----
    python scripts/diagnose_dataset.py --root "E:/.../20260815" --out reports/

Every input is optional; missing files are reported and skipped.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

CONTEXT_SAMPLES_DEFAULT = 2048
SAMPLE_RATE_HZ = 50.0
EVENT_CODES = (
    "STANDING_UP",
    "LYING_DOWN",
    "URINATION",
    "DEFECATION",
    "TAIL_RAISED",
    "TAIL_WAGGING",
)
EXCRETION_CODES = ("URINATION", "DEFECATION")


def read_csv(path: Path) -> pd.DataFrame | None:
    if path is None or not Path(path).exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


def read_json(path: Path) -> object | None:
    if path is None or not Path(path).exists():
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------- A + B
def segment_report(cache_root: Path) -> dict[str, object]:
    """Segment counts per session from the supervised session cache."""

    per_session: dict[str, list[dict[str, int]]] = {}
    if cache_root is not None and Path(cache_root).exists():
        for directory in sorted(Path(cache_root).iterdir()):
            meta = directory / "metadata.json"
            if not meta.exists():
                continue
            with meta.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            per_session[directory.name] = list(payload.get("segments", []))
    if not per_session:
        return {"available": False, "reason": "no supervised session_cache metadata"}

    counts = Counter(len(segments) for segments in per_session.values())
    lengths = [
        int(segment["stop_index"]) - int(segment["start_index"])
        for segments in per_session.values()
        for segment in segments
    ]
    lengths_array = np.asarray(lengths, dtype=np.float64) if lengths else np.zeros(1)
    return {
        "available": True,
        "sessions": len(per_session),
        "segments_total": int(sum(int(k) * int(v) for k, v in counts.items())),
        "segments_per_session_histogram": {int(k): int(v) for k, v in sorted(counts.items())},
        "sessions_with_one_segment": int(counts.get(1, 0)),
        "sessions_multi_segment": int(sum(v for k, v in counts.items() if k > 1)),
        "segment_seconds": {
            "min": float(lengths_array.min() / SAMPLE_RATE_HZ),
            "median": float(np.median(lengths_array) / SAMPLE_RATE_HZ),
            "mean": float(lengths_array.mean() / SAMPLE_RATE_HZ),
            "max": float(lengths_array.max() / SAMPLE_RATE_HZ),
        },
        "_per_session": per_session,
    }


def contamination_report(
    samples: pd.DataFrame | None,
    per_session: dict[str, list[dict[str, int]]],
    context_samples: int,
) -> dict[str, object]:
    """How many samples had a leaky context window under the old code."""

    if samples is None or not per_session:
        return {"available": False, "reason": "needs samples.csv and segment metadata"}
    if "cache_key" not in samples.columns or "center_index" not in samples.columns:
        return {"available": False, "reason": "samples.csv lacks cache_key/center_index"}

    contaminated = 0
    partially = 0
    total = 0
    for cache_key, group in samples.groupby("cache_key", sort=False):
        segments = per_session.get(str(cache_key))
        if not segments:
            continue
        starts = np.asarray([int(s["start_index"]) for s in segments], dtype=np.int64)
        stops = np.asarray([int(s["stop_index"]) for s in segments], dtype=np.int64)
        centers = group["center_index"].to_numpy(np.int64)
        position = np.clip(np.searchsorted(starts, centers, side="right") - 1, 0, len(starts) - 1)
        segment_start = starts[position]
        old_start = np.maximum(0, centers + 1 - context_samples)
        leak = old_start < segment_start
        total += centers.size
        contaminated += int(leak.sum())
        # fully-poisoned windows: the whole context predates this segment
        partially += int(np.sum(leak & (segment_start > centers + 1 - context_samples)))
    return {
        "available": True,
        "samples_total": int(total),
        "samples_with_cross_segment_context": int(contaminated),
        "fraction": float(contaminated / total) if total else None,
        "context_samples": int(context_samples),
        "note": "counts are for the OLD code path; the patched dataset clips to the segment",
    }


# ---------------------------------------------------------------- C
def annotation_report(annotations: pd.DataFrame | None) -> dict[str, object]:
    if annotations is None:
        return {"available": False, "reason": "annotations csv not found"}
    frame = annotations.copy()
    frame["duration_ms"] = frame["t_end_rel_ms"].astype(np.int64) - frame["t_start_rel_ms"].astype(np.int64)
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
            }
        )

    # Overlap conflict: excretion intervals that are not also annotated as a
    # raised tail.  With the "whole event including the preparatory tail lift"
    # definition, every one of these is a false negative for the tail head
    # under the legacy mask rule.
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
                "silently_negative_for_tail_head": int(len(subset) - overlapping),
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
                "status": "reportable" if (len(group) >= 10 and cows >= 3) else "not_evaluable",
            }
        )
    return {
        "available": True,
        "rows": int(len(frame)),
        "per_code": per_code,
        "tail_conflicts": conflicts,
        "evaluability": evaluability,
    }


# ---------------------------------------------------------------- D
def coverage_report(coverage: pd.DataFrame | None, samples: pd.DataFrame | None) -> dict[str, object]:
    if coverage is None or len(coverage) == 0:
        covered_sessions = 0
        detail: list[dict[str, object]] = []
        hours = 0.0
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
        covered_sessions = int(frame.groupby(["device_mac", "session_id"]).ngroups)
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


# ---------------------------------------------------------------- E
def fold_report(splits: object) -> dict[str, object]:
    if not isinstance(splits, dict) or "folds" not in splits:
        return {"available": False, "reason": "splits json not found or malformed"}
    rows = []
    for fold in splits["folds"]:
        counts = fold.get("counts", {})
        test = counts.get("test", {})
        rows.append(
            {
                "fold": fold.get("fold"),
                "test_cow": fold.get("test_cow"),
                "train_sessions": len(fold.get("train_sessions", [])),
                "validation_sessions": len(fold.get("validation_sessions", [])),
                "test_sessions": len(fold.get("test_sessions", [])),
                "test_samples": test.get("samples"),
                "test_event_positives": {
                    code: value.get("positive") for code, value in (test.get("events") or {}).items()
                },
            }
        )
    usable = [row for row in rows if (row["test_samples"] or 0) >= 1000]
    return {
        "available": True,
        "protocol": splits.get("protocol"),
        "folds": rows,
        "folds_total": len(rows),
        "folds_with_usable_test": len(usable),
        "note": "folds with <1000 test samples produce noise, not evidence; use pooled LOCO",
    }


# ---------------------------------------------------------------- report
def render_markdown(payload: dict[str, object]) -> str:
    lines: list[str] = ["# 数据集诊断报告", ""]

    segments = payload["segments"]
    lines.append("## A. 段（segment）分布")
    if not segments.get("available"):
        lines.append(f"- 不可用：{segments.get('reason')}")
    else:
        lines.append(f"- 会话数：{segments['sessions']}，总段数：{segments['segments_total']}")
        lines.append(
            f"- 单段会话：{segments['sessions_with_one_segment']}，多段会话：{segments['sessions_multi_segment']}"
        )
        lines.append(f"- 每会话段数直方图：{segments['segments_per_session_histogram']}")
        seconds = segments["segment_seconds"]
        lines.append(
            "- 段时长（秒）：min {min:.1f} / median {median:.1f} / mean {mean:.1f} / max {max:.1f}".format(**seconds)
        )
    lines.append("")

    contamination = payload["contamination"]
    lines.append("## B. 旧代码的跨段污染量")
    if not contamination.get("available"):
        lines.append(f"- 不可用：{contamination.get('reason')}")
    else:
        fraction = contamination["fraction"]
        lines.append(
            f"- 样本总数 {contamination['samples_total']}，其中 {contamination['samples_with_cross_segment_context']} "
            f"个（{fraction:.2%}）的因果上下文越过了段边界"
        )
        lines.append(
            "- 判读：占比 <1% 可只重跑评估；>5% 说明既有模型结论需要重训后重新确认"
        )
    lines.append("")

    annotations = payload["annotations"]
    lines.append("## C. 标注审计")
    if not annotations.get("available"):
        lines.append(f"- 不可用：{annotations.get('reason')}")
    else:
        lines.append("| 代码 | 区间数 | 牛数 | 会话数 | 均时长(s) | 中位(s) | 总时长(min) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for item in annotations["per_code"]:
            lines.append(
                "| {code} | {intervals} | {cows} | {sessions} | {duration_s_mean:.1f} | "
                "{duration_s_median:.1f} | {total_minutes:.1f} |".format(**item)
            )
        lines.append("")
        lines.append("### 抬尾标签冲突（口径 A 修复的对象）")
        lines.append("| 代码 | 区间数 | 同时标了抬尾 | 被当成抬尾负例 |")
        lines.append("|---|---:|---:|---:|")
        for item in annotations["tail_conflicts"]:
            lines.append(
                "| {code} | {intervals} | {also_annotated_tail_raised} | "
                "{silently_negative_for_tail_head} |".format(**item)
            )
        lines.append("")
        lines.append("### 可报告性（≥10 区间且 ≥3 头牛）")
        lines.append("| 事件 | 区间数 | 牛数 | 结论 |")
        lines.append("|---|---:|---:|---|")
        for item in annotations["evaluability"]:
            lines.append("| {code} | {intervals} | {cows} | {status} |".format(**item))
    lines.append("")

    coverage = payload["coverage"]
    lines.append("## D. 穷尽复核覆盖")
    lines.append(f"- 覆盖时长：{coverage['exhaustively_reviewed_hours']:.2f} 小时")
    lines.append(f"- 覆盖会话：{coverage['covered_sessions']} / {coverage['total_sessions']}")
    lines.append(f"- 结论：{coverage['verdict']}")
    lines.append("")

    folds = payload["folds"]
    lines.append("## E. LOCO 折平衡")
    if not folds.get("available"):
        lines.append(f"- 不可用：{folds.get('reason')}")
    else:
        lines.append("| 折 | 留出牛 | 训练会话 | 验证会话 | 测试会话 | 测试样本 |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for row in folds["folds"]:
            lines.append(
                "| {fold} | {test_cow} | {train_sessions} | {validation_sessions} | "
                "{test_sessions} | {test_samples} |".format(**row)
            )
        lines.append("")
        lines.append(
            f"- 共 {folds['folds_total']} 折，其中测试样本 ≥1000 的只有 {folds['folds_with_usable_test']} 折"
        )
        lines.append(f"- {folds['note']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Cattle IMU dataset diagnostics")
    parser.add_argument("--root", type=Path, default=Path("."), help="project root")
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--samples", type=Path, default=None)
    parser.add_argument("--session-cache", type=Path, default=None)
    parser.add_argument("--review-coverage", type=Path, default=None)
    parser.add_argument("--splits", type=Path, default=None)
    parser.add_argument("--context-samples", type=int, default=CONTEXT_SAMPLES_DEFAULT)
    parser.add_argument("--out", type=Path, default=Path("diagnostics"))
    args = parser.parse_args()

    root = args.root
    data = root / "datasets" / "cowmata_imu"
    annotations_path = args.annotations or data / "annotations/annotations_adjudicated_minimal.csv"
    samples_path = args.samples or data / "supervised_cache/samples.csv"
    cache_path = args.session_cache or data / "supervised_cache/session_cache"
    coverage_path = args.review_coverage or data / "annotations/review_coverage_template.csv"
    splits_path = args.splits or data / "loco_splits/loco_splits.json"

    annotations = read_csv(annotations_path)
    samples = read_csv(samples_path)
    coverage = read_csv(coverage_path)
    splits = read_json(splits_path)

    segments = segment_report(cache_path)
    per_session = segments.pop("_per_session", {}) if segments.get("available") else {}
    payload = {
        "inputs": {
            "annotations": str(annotations_path),
            "samples": str(samples_path),
            "session_cache": str(cache_path),
            "review_coverage": str(coverage_path),
            "splits": str(splits_path),
        },
        "segments": segments,
        "contamination": contamination_report(samples, per_session, args.context_samples),
        "annotations": annotation_report(annotations),
        "coverage": coverage_report(coverage, samples),
        "folds": fold_report(splits),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    markdown = render_markdown(payload)
    with (args.out / "diagnostics.md").open("w", encoding="utf-8") as handle:
        handle.write(markdown)
    print(markdown)
    print(f"[written] {args.out / 'diagnostics.json'}")


if __name__ == "__main__":
    main()
