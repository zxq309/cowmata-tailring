"""Build leave-one-cow-out test folds with session-held-out validation."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cattle_imu.annotations import BODY_CODES, EVENT_CODES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    return parser.parse_args()


def session_key(device_mac: str, session_id: str) -> str:
    return f"{device_mac}|{session_id}"


def choose_validation_sessions(sessions: pd.DataFrame, fraction: float) -> set[str]:
    """Hold out complete sessions from every remaining cow, ordered in time."""

    chosen: set[str] = set()
    for _, group in sessions.groupby("cow_id", sort=True):
        ordered = group.sort_values(["session_id", "device_mac"], kind="stable")
        count = max(1, int(math.ceil(len(ordered) * fraction)))
        # Interleave validation sessions across the time span rather than using
        # adjacent windows from the same session on both sides of the split.
        positions = np.linspace(0, len(ordered) - 1, count, dtype=int)
        for row in ordered.iloc[np.unique(positions)].itertuples(index=False):
            chosen.add(session_key(str(row.device_mac), str(row.session_id)))
    return chosen


def feature_stats(cache_root: Path, cache_names: list[str]) -> tuple[list[float], list[float], int]:
    total = np.zeros(13, dtype=np.float64)
    total_square = np.zeros(13, dtype=np.float64)
    count = 0
    for name in sorted(set(cache_names)):
        values = np.load(cache_root / name / "features.npy", mmap_mode="r")
        # Accumulate in chunks to keep the peak temporary allocation bounded.
        for start in range(0, len(values), 100_000):
            chunk = np.asarray(values[start : start + 100_000], dtype=np.float64)
            total += chunk.sum(axis=0)
            total_square += np.square(chunk).sum(axis=0)
            count += len(chunk)
    mean = total / max(count, 1)
    variance = np.maximum(total_square / max(count, 1) - np.square(mean), 1e-8)
    std = np.sqrt(variance)
    return mean.tolist(), std.tolist(), count


def split_counts(samples: pd.DataFrame, keys: set[str]) -> dict[str, object]:
    frame = samples[samples["session_key"].isin(keys)]
    body = {
        BODY_CODES[index]: int((frame["body_target"] == index).sum())
        for index in range(len(BODY_CODES))
    }
    body_values = frame["body_target"].to_numpy(np.int64)
    events: dict[str, dict[str, int]] = {}
    for code in EVENT_CODES:
        positive = int(frame[f"event_{code}"].sum())
        mask = int(frame[f"mask_{code}"].sum())
        events[code] = {"positive": positive, "evaluated": mask, "negative": mask - positive}
    return {
        "samples": len(frame),
        "sessions": int(frame["session_key"].nunique()),
        "cows": sorted(frame["cow_id"].unique().tolist()),
        "body": body,
        "posture": {
            "UPRIGHT": int(np.sum(np.isin(body_values, [0, 2, 3]))),
            "LYING": int(np.sum(body_values == 1)),
        },
        "walking": {
            "positive": int(np.sum(body_values == 2)),
            "evaluated": int(np.sum(body_values >= 0)),
        },
        "events": events,
    }


def main() -> int:
    args = parse_args()
    samples = pd.read_csv(args.samples, encoding="utf-8-sig")
    sessions = pd.read_csv(args.sessions, encoding="utf-8-sig")
    sessions = sessions[sessions["status"] == "included"].copy()
    samples["session_key"] = [
        session_key(str(device), str(session))
        for device, session in zip(samples["device_mac"], samples["session_id"])
    ]
    sessions["session_key"] = [
        session_key(str(device), str(session))
        for device, session in zip(sessions["device_mac"], sessions["session_id"])
    ]
    all_keys = set(sessions["session_key"])
    cows = sorted(sessions["cow_id"].astype(str).unique())
    run_dir = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    folds: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for fold_index, test_cow in enumerate(cows, start=1):
        test_sessions = sessions[sessions["cow_id"].astype(str) == test_cow]
        remaining = sessions[sessions["cow_id"].astype(str) != test_cow]
        test_keys = set(test_sessions["session_key"])
        val_keys = choose_validation_sessions(remaining, args.validation_fraction)
        train_keys = all_keys - test_keys - val_keys
        if train_keys & val_keys or train_keys & test_keys or val_keys & test_keys:
            raise RuntimeError("split session overlap detected")
        if train_keys | val_keys | test_keys != all_keys:
            raise RuntimeError("split does not cover every included session")
        train_cows = set(sessions[sessions["session_key"].isin(train_keys)]["cow_id"].astype(str))
        val_cows = set(sessions[sessions["session_key"].isin(val_keys)]["cow_id"].astype(str))
        test_cows = set(test_sessions["cow_id"].astype(str))
        if train_cows & test_cows or val_cows & test_cows:
            raise RuntimeError("test cow leaked into train or validation")
        train_cache = sessions[sessions["session_key"].isin(train_keys)]["cache_key"].astype(str).tolist()
        mean, std, normalization_frames = feature_stats(args.cache_root, train_cache)
        counts = {
            "train": split_counts(samples, train_keys),
            "validation": split_counts(samples, val_keys),
            "test": split_counts(samples, test_keys),
        }
        fold = {
            "fold": fold_index,
            "test_cow": test_cow,
            "train_sessions": sorted(train_keys),
            "validation_sessions": sorted(val_keys),
            "test_sessions": sorted(test_keys),
            "normalization": {
                "source": "all resampled frames from training sessions only",
                "frames": normalization_frames,
                "mean": mean,
                "std": std,
            },
            "counts": counts,
        }
        folds.append(fold)
        for split_name, split_summary in counts.items():
            summary_rows.append(
                {
                    "fold": fold_index,
                    "test_cow": test_cow,
                    "split": split_name,
                    "samples": split_summary["samples"],
                    "sessions": split_summary["sessions"],
                    "cows": ";".join(split_summary["cows"]),
                    **{f"body_{key}": value for key, value in split_summary["body"].items()},
                    **{
                        f"event_{code}_positive": value["positive"]
                        for code, value in split_summary["events"].items()
                    },
                    **{
                        f"event_{code}_negative": value["negative"]
                        for code, value in split_summary["events"].items()
                    },
                }
            )
    manifest = {
        "schema_version": 2,
        "protocol": "strict_loco",
        "purpose": "cross-cow evaluation; keep separate from development_all candidate-mining runs",
        "samples_path": str(args.samples),
        "sessions_path": str(args.sessions),
        "cache_root": str(args.cache_root),
        "test_policy": "leave one normalized cow_id out; the test cow is absent from training, validation, normalization and threshold selection",
        "validation_policy": "complete sessions held out within the remaining cows; validation is session-disjoint but not necessarily cow-disjoint",
        "folds": folds,
    }
    with (run_dir / "loco_splits.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    pd.DataFrame(summary_rows).to_csv(run_dir / "fold_summary.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"run_dir": str(run_dir), "folds": len(folds), "test_cows": cows}, ensure_ascii=False, indent=2))
    print(pd.DataFrame(summary_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
