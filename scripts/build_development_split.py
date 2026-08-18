"""Build the all-cow development split used only for candidate mining.

Every cow contributes training sessions. Complete sessions are held out for
validation, but cow identity is deliberately shared between train and
validation. This protocol must never be reported as cross-cow generalization.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from cattle_imu.annotations import BODY_CODES, EVENT_CODES


def session_key(device: object, session: object) -> str:
    return f"{device}|{session}"


def choose_validation_sessions(sessions: pd.DataFrame, fraction: float) -> set[str]:
    chosen: set[str] = set()
    for cow, group in sessions.groupby("cow_id", sort=True):
        ordered = group.sort_values(["session_id", "device_mac"], kind="stable")
        if len(ordered) < 2:
            # A one-session cow is kept in training so its scarce evidence is
            # not wasted. It contributes no validation evidence.
            continue
        count = min(len(ordered) - 1, max(1, int(math.ceil(len(ordered) * fraction))))
        positions = np.linspace(0, len(ordered) - 1, count, dtype=int)
        chosen.update(ordered.iloc[np.unique(positions)]["session_key"].astype(str))
    return chosen


def feature_stats(cache_root: Path, cache_names: list[str]) -> tuple[list[float], list[float], int]:
    total = np.zeros(13, dtype=np.float64)
    square = np.zeros(13, dtype=np.float64)
    count = 0
    for name in sorted(set(cache_names)):
        values = np.load(cache_root / name / "features.npy", mmap_mode="r")
        for start in range(0, len(values), 100_000):
            chunk = np.asarray(values[start : start + 100_000], dtype=np.float64)
            total += chunk.sum(0)
            square += np.square(chunk).sum(0)
            count += len(chunk)
    mean = total / max(count, 1)
    std = np.sqrt(np.maximum(square / max(count, 1) - np.square(mean), 1e-8))
    return mean.tolist(), std.tolist(), count


def counts(samples: pd.DataFrame, keys: set[str]) -> dict[str, object]:
    frame = samples[samples["session_key"].isin(keys)]
    body = frame["body_target"].to_numpy(np.int64)
    return {
        "samples": int(len(frame)),
        "sessions": int(frame["session_key"].nunique()),
        "cows": sorted(frame["cow_id"].astype(str).unique().tolist()),
        "source_body": {BODY_CODES[index]: int(np.sum(body == index)) for index in range(len(BODY_CODES))},
        "posture": {
            "UPRIGHT": int(np.sum(np.isin(body, [0, 2, 3]))),
            "LYING": int(np.sum(body == 1)),
        },
        "walking": {"positive": int(np.sum(body == 2)), "evaluated": int(np.sum(body >= 0))},
        "events": {
            code: {
                "positive": int(frame[f"event_{code}"].sum()),
                "evaluated": int(frame[f"mask_{code}"].sum()),
            }
            for code in EVENT_CODES
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    samples = pd.read_csv(args.samples, encoding="utf-8-sig")
    sessions = pd.read_csv(args.sessions, encoding="utf-8-sig")
    sessions = sessions[sessions["status"] == "included"].copy()
    for frame in (samples, sessions):
        frame["session_key"] = [session_key(device, session) for device, session in zip(frame["device_mac"], frame["session_id"])]
    all_keys = set(sessions["session_key"].astype(str))
    validation_keys = choose_validation_sessions(sessions, args.validation_fraction)
    train_keys = all_keys - validation_keys
    train_cows = set(sessions[sessions["session_key"].isin(train_keys)]["cow_id"].astype(str))
    all_cows = set(sessions["cow_id"].astype(str))
    if train_cows != all_cows:
        raise RuntimeError(f"not every cow contributes training data: {sorted(all_cows - train_cows)}")
    cache_names = sessions[sessions["session_key"].isin(train_keys)]["cache_key"].astype(str).tolist()
    mean, std, frames = feature_stats(args.cache_root, cache_names)
    manifest = {
        "schema_version": 2,
        "protocol": "development_all_cows_session_holdout",
        "purpose": "candidate mining and manual label expansion only; not a cross-cow accuracy estimate",
        "samples_path": str(args.samples),
        "sessions_path": str(args.sessions),
        "cache_root": str(args.cache_root),
        "limitations": [
            "Cow identities occur in both training and validation.",
            "Reported validation metrics measure iteration progress only.",
            "Keep cow_id permanently and use LOCO or a frozen cow holdout for final claims.",
        ],
        "folds": [
            {
                "fold": 0,
                "test_cow": None,
                "train_sessions": sorted(train_keys),
                "validation_sessions": sorted(validation_keys),
                "test_sessions": [],
                "normalization": {
                    "source": "training sessions only",
                    "frames": frames,
                    "mean": mean,
                    "std": std,
                },
                "counts": {
                    "train": counts(samples, train_keys),
                    "validation": counts(samples, validation_keys),
                },
            }
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(args.output), "train_sessions": len(train_keys), "validation_sessions": len(validation_keys), "cows": sorted(all_cows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
