# -*- coding: utf-8 -*-
"""Build the human review queue for the annotation bootstrap loop.

Four rules are enforced in code, because they are what makes the loop
converge instead of drifting:

1. **A random control group is always included.**  Model-selected candidates
   are a biased sample; precision and recall estimated on them mean nothing.
   Only the ``random_control`` rows may be used to compute metrics.  The
   default is 30% of the active budget and it cannot be switched off below
   10%.
2. **Selection mixes exploitation, uncertainty and diversity** (default
   40/40/20).  Reviewing only high-scoring windows teaches the model nothing
   it does not already know.
3. **A per-cow-per-day cap** derived from published excretion rates keeps one
   noisy session from eating the whole budget.
4. **Nothing here is a label.**  Every row leaves with ``review_decision``
   empty.  Candidates must never be merged into the truth table before a human
   fills that column in.

Usage
-----
    python scripts/mine_candidates.py \
        --predictions runs/feature_model/predictions_URINATION.csv \
        --events URINATION,DEFECATION,TAIL_RAISED \
        --per-event 40 --out runs/review_round_02
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cattle_imu.metrics import PHYSIOLOGICAL_RATE_PER_HOUR  # noqa: E402

POINT_MS = 500
DEFAULT_MERGE_GAP_MS = 5000
REVIEW_COLUMNS = ("review_decision", "reviewer", "reviewed_at", "notes")


def merge_intervals(
    times_ms: np.ndarray,
    scores: np.ndarray,
    selected: np.ndarray,
    *,
    max_gap_ms: int,
    pad_ms: int,
) -> list[dict[str, object]]:
    """Merge selected points into candidate intervals with a peak score."""

    indices = np.flatnonzero(selected)
    if indices.size == 0:
        return []
    out: list[dict[str, object]] = []
    start = indices[0]
    previous = indices[0]
    for index in indices[1:]:
        if times_ms[index] - times_ms[previous] > max_gap_ms:
            out.append(_interval(times_ms, scores, start, previous, pad_ms))
            start = index
        previous = index
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


def candidates_for_event(
    frame: pd.DataFrame,
    event: str,
    *,
    quantile: float,
    merge_gap_ms: int,
    pad_ms: int,
) -> pd.DataFrame:
    column = f"prob_{event}"
    if column not in frame.columns:
        return pd.DataFrame()
    # Prediction files are concatenated, so rows coming from another event's
    # file have a NaN here; they must not poison the session quantile.
    frame = frame[np.isfinite(frame[column].to_numpy(dtype=float))]
    rows: list[dict[str, object]] = []
    for (device, session), group in frame.groupby(["device_mac", "session_id"], sort=True):
        ordered = group.sort_values("center_time_ms", kind="stable")
        times = ordered["center_time_ms"].to_numpy(np.int64)
        scores = ordered[column].to_numpy(float)
        if scores.size < 3:
            continue
        cut = float(np.quantile(scores, quantile))
        for interval in merge_intervals(
            times, scores, scores >= cut, max_gap_ms=merge_gap_ms, pad_ms=pad_ms
        ):
            rows.append(
                {
                    "device_mac": str(device),
                    "session_id": str(session),
                    "cow_id": str(ordered["cow_id"].iloc[0]) if "cow_id" in ordered else "",
                    "event_code": event,
                    "session_cut_score": cut,
                    **interval,
                }
            )
    return pd.DataFrame(rows)


def diverse_subset(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    """Greedy spread over cows, then sessions, then score deciles."""

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


def apply_daily_cap(frame: pd.DataFrame, event: str, cap_multiplier: float) -> pd.DataFrame:
    """Keep at most ``rate * 24 * multiplier`` candidates per cow per session."""

    bound = PHYSIOLOGICAL_RATE_PER_HOUR.get(event.upper())
    if bound is None:
        return frame
    limit = max(1, int(round(bound * 24.0 * cap_multiplier)))
    return (
        frame.sort_values("max_score", ascending=False, kind="stable")
        .groupby(["cow_id", "session_id"], sort=False, group_keys=False)
        .head(limit)
    )


def random_control(frame: pd.DataFrame, event: str, count: int, seed: int, pad_ms: int) -> pd.DataFrame:
    """Uniformly sampled windows, independent of any model score."""

    column = f"prob_{event}"
    if column in frame.columns:
        frame = frame[np.isfinite(frame[column].to_numpy(dtype=float))]
    if count <= 0 or len(frame) == 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    picked = frame.sample(n=min(count, len(frame)), random_state=int(rng.integers(1 << 31)))
    return pd.DataFrame(
        {
            "device_mac": picked["device_mac"].astype(str).to_numpy(),
            "session_id": picked["session_id"].astype(str).to_numpy(),
            "cow_id": picked["cow_id"].astype(str).to_numpy() if "cow_id" in picked else "",
            "event_code": event,
            "t_start_rel_ms": picked["center_time_ms"].to_numpy(np.int64) - pad_ms,
            "t_end_rel_ms": picked["center_time_ms"].to_numpy(np.int64) + POINT_MS + pad_ms,
            "peak_time_ms": picked["center_time_ms"].to_numpy(np.int64),
            "max_score": picked[column].to_numpy(float) if column in picked else np.nan,
            "mean_score": picked[column].to_numpy(float) if column in picked else np.nan,
            "points": 1,
            "selection_strategy": "random_control",
            "sampling_source": "random",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an active-learning review queue")
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--events", type=str, required=True, help="comma separated event codes")
    parser.add_argument("--per-event", type=int, default=40, help="active budget per event")
    parser.add_argument("--random-fraction", type=float, default=0.3)
    parser.add_argument("--top-fraction", type=float, default=0.4)
    parser.add_argument("--boundary-fraction", type=float, default=0.4)
    parser.add_argument("--candidate-quantile", type=float, default=0.995)
    parser.add_argument("--merge-gap-ms", type=int, default=DEFAULT_MERGE_GAP_MS)
    parser.add_argument("--pad-ms", type=int, default=5000)
    parser.add_argument("--daily-cap-multiplier", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.random_fraction < 0.1:
        raise SystemExit(
            "--random-fraction below 0.1 is refused: without a random control group "
            "no unbiased precision or recall can be estimated from this round."
        )
    frames = [pd.read_csv(path, encoding="utf-8-sig") for path in args.predictions]
    frame = pd.concat(frames, ignore_index=True)
    if "cow_id" not in frame.columns:
        frame["cow_id"] = ""

    events = [code.strip().upper() for code in args.events.split(",") if code.strip()]
    queues: list[pd.DataFrame] = []
    summary: list[dict[str, object]] = []
    for event in events:
        column = f"prob_{event}"
        if column not in frame.columns:
            print(f"[skip] {event}: no {column} column in predictions")
            continue
        candidates = candidates_for_event(
            frame,
            event,
            quantile=args.candidate_quantile,
            merge_gap_ms=args.merge_gap_ms,
            pad_ms=args.pad_ms,
        )
        if candidates.empty:
            print(f"[skip] {event}: no candidates above the session quantile")
            continue
        candidates = apply_daily_cap(candidates, event, args.daily_cap_multiplier).reset_index(drop=True)

        budget = int(args.per_event)
        n_top = int(round(budget * args.top_fraction))
        n_boundary = int(round(budget * args.boundary_fraction))
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
        diverse = diverse_subset(pool, n_diverse) if len(pool) else pool

        for block, name in ((top, "active_top"), (boundary, "active_boundary"), (diverse, "active_diverse")):
            if len(block):
                queues.append(block.assign(selection_strategy=name, sampling_source="active"))
        control = random_control(
            frame, event, int(round(budget * args.random_fraction)), args.seed, args.pad_ms
        )
        if len(control):
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
        raise SystemExit("nothing to review; check the prediction files and the event codes")
    queue = pd.concat(queues, ignore_index=True)
    queue.insert(0, "candidate_id", [f"CND-{i:05d}" for i in range(1, len(queue) + 1)])
    for column in REVIEW_COLUMNS:
        queue[column] = ""
    queue["duration_s"] = (queue["t_end_rel_ms"] - queue["t_start_rel_ms"]) / 1000.0

    args.out.mkdir(parents=True, exist_ok=True)
    queue.to_csv(args.out / "review_queue.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "generated_from": [str(path) for path in args.predictions],
        "per_event_budget": args.per_event,
        "fractions": {
            "top": args.top_fraction,
            "boundary": args.boundary_fraction,
            "diverse": round(1.0 - args.top_fraction - args.boundary_fraction, 4),
            "random_control": args.random_fraction,
        },
        "events": summary,
        "rows": int(len(queue)),
        "rules": [
            "only rows with sampling_source=random may be used to estimate precision/recall",
            "review_decision must be filled by a human before any row enters the truth table",
            "keep this file; the next round's report must cite the round it came from",
        ],
    }
    with (args.out / "review_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[done] {args.out / 'review_queue.csv'} rows={len(queue)}")


if __name__ == "__main__":
    main()
