# -*- coding: utf-8 -*-
"""Train the hand-crafted-feature branch (gradient boosting) under LOCO.

Discipline enforced here, because it is where the previous tail-feature scripts
went wrong:

* **Grouping.** Splits are by *cow*.  A random split puts 2 Hz points from the
  same event on both sides and produces meaningless scores.
* **Threshold selection.** Thresholds come from the validation cows only and
  are then frozen for the test cows.
* **Pooled evaluation.** Per-fold event metrics on a fold with 13 test samples
  are noise.  Test predictions from all folds are concatenated and scored once,
  which keeps the leave-one-cow-out property while making the statistic stable.
  Per-fold detail is still written out for audit.
* **Rate plausibility.** Every event reports its predicted rate against a
  published physiological bound.

Usage
-----
    python scripts/train_feature_model.py \
        --feature-table runs/feature_table/feature_table.parquet \
        --splits datasets/cowmata_imu/loco_splits/loco_splits.json \
        --annotations datasets/cowmata_imu/annotations/annotations_adjudicated_minimal.csv \
        --out runs/feature_model --backend xgboost --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cattle_imu.annotations import EVENT_CODES  # noqa: E402
from cattle_imu.features import feature_columns  # noqa: E402
from cattle_imu.gbdt import BinaryBooster, BoosterConfig, available_backends  # noqa: E402
from cattle_imu.metrics import (  # noqa: E402
    PHYSIOLOGICAL_RATE_PER_HOUR,
    binary_point_metrics,
    choose_threshold,
    event_level_metrics,
    selection_score,
)

BODY_CODES = ("STANDING", "LYING", "WALKING", "FEEDING")
POINT_SECONDS = 0.5


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def session_key(frame: pd.DataFrame) -> pd.Series:
    return frame["device_mac"].astype(str) + "|" + frame["session_id"].astype(str)


def task_targets(frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return ``{task: (target, mask)}`` for every trainable head."""

    body = frame["body_target"].to_numpy(np.int64)
    labelled = (body >= 0).astype(np.float32)
    tasks: dict[str, tuple[np.ndarray, np.ndarray]] = {
        # posture: LYING (index 1) against everything upright
        "POSTURE_LYING": ((body == BODY_CODES.index("LYING")).astype(np.int8), labelled),
        "WALKING": ((body == BODY_CODES.index("WALKING")).astype(np.int8), labelled),
    }
    for code in EVENT_CODES:
        target_column = f"event_{code}"
        mask_column = f"mask_{code}"
        if target_column in frame.columns and mask_column in frame.columns:
            tasks[code] = (
                frame[target_column].to_numpy(np.int8),
                frame[mask_column].to_numpy(np.float32),
            )
    return tasks


def build_folds(table: pd.DataFrame, splits_path: Path | None, seed: int) -> list[dict[str, object]]:
    keys = session_key(table)
    if splits_path is not None and splits_path.exists():
        with splits_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        folds = []
        for fold in payload["folds"]:
            folds.append(
                {
                    "fold": int(fold["fold"]),
                    "test_cow": str(fold.get("test_cow", "")),
                    "train": set(fold.get("train_sessions", [])),
                    "validation": set(fold.get("validation_sessions", [])),
                    "test": set(fold.get("test_sessions", [])),
                }
            )
        return folds

    # Fallback: leave-one-cow-out generated on the fly, with one training cow
    # held out for threshold selection.
    cows = sorted(table["cow_id"].astype(str).unique())
    rng = np.random.default_rng(seed)
    folds = []
    for index, cow in enumerate(cows, start=1):
        others = [c for c in cows if c != cow]
        validation_cow = others[int(rng.integers(len(others)))] if others else cow
        by_cow = table.assign(_key=keys).groupby(table["cow_id"].astype(str))["_key"]
        sets = {name: set(group) for name, group in by_cow}
        folds.append(
            {
                "fold": index,
                "test_cow": cow,
                "train": set().union(*[sets[c] for c in others if c != validation_cow])
                if len(others) > 1
                else set(),
                "validation": sets.get(validation_cow, set()),
                "test": sets.get(cow, set()),
            }
        )
    return folds


def run_task(
    task: str,
    table: pd.DataFrame,
    features: list[str],
    folds: list[dict[str, object]],
    config: BoosterConfig,
    backend: str | None,
    annotations: pd.DataFrame | None,
    rate_constrained_threshold: bool = False,
) -> dict[str, object]:
    keys = session_key(table)
    targets, masks = task_targets(table)[task]
    matrix = table[features].to_numpy(np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    fold_reports: list[dict[str, object]] = []
    pooled_frames: list[pd.DataFrame] = []
    validation_frames: list[pd.DataFrame] = []
    importance: list[dict[str, object]] = []
    rate_bound = PHYSIOLOGICAL_RATE_PER_HOUR.get(task)

    for fold in folds:
        train_rows = keys.isin(fold["train"]).to_numpy() & (masks > 0)
        validation_rows = keys.isin(fold["validation"]).to_numpy() & (masks > 0)
        test_rows = keys.isin(fold["test"]).to_numpy() & (masks > 0)
        train_positive = int(targets[train_rows].sum())
        if train_rows.sum() < 50 or train_positive < 3:
            fold_reports.append(
                {
                    "fold": fold["fold"],
                    "test_cow": fold["test_cow"],
                    "status": "skipped_insufficient_training_positives",
                    "train_rows": int(train_rows.sum()),
                    "train_positive": train_positive,
                }
            )
            continue

        booster = BinaryBooster(config, backend=backend).fit(
            matrix[train_rows], targets[train_rows]
        )
        if not importance:
            importance = booster.feature_importance(features)[:30]

        threshold = 0.5
        if validation_rows.sum() > 0 and np.unique(targets[validation_rows]).size == 2:
            validation_probability = booster.predict_proba(matrix[validation_rows])
            max_rate = None
            if rate_constrained_threshold and rate_bound is not None:
                # Convert an events/hour ceiling into a point-positive ceiling.
                # OFF by default: the labelled subset is heavily enriched in
                # events compared with a real 24 h day, so applying a
                # wall-clock rate bound to it silently destroys recall.  The
                # bound belongs in reporting (rate_plausibility) and in
                # candidate mining, not in threshold selection.
                max_rate = min(1.0, rate_bound * 120.0 * POINT_SECONDS / 3600.0 * 2.0)
            threshold = choose_threshold(
                targets[validation_rows],
                validation_probability,
                np.ones(int(validation_rows.sum()), dtype=bool),
                max_positive_rate=max_rate,
            )

        report: dict[str, object] = {
            "fold": fold["fold"],
            "test_cow": fold["test_cow"],
            "status": "trained",
            "backend": booster.backend,
            "train_rows": int(train_rows.sum()),
            "train_positive": train_positive,
            "validation_rows": int(validation_rows.sum()),
            "threshold": float(threshold),
        }
        if validation_rows.sum() > 0:
            validation_probability = booster.predict_proba(matrix[validation_rows])
            report["validation_point"] = binary_point_metrics(
                targets[validation_rows], validation_probability,
                np.ones(int(validation_rows.sum()), dtype=bool), threshold,
            )
            vblock = table.loc[validation_rows, ["device_mac", "session_id", "cow_id", "center_time_ms"]].copy()
            vblock[f"target_{task}"] = targets[validation_rows]
            vblock[f"mask_{task}"] = 1
            vblock[f"prob_{task}"] = validation_probability
            vblock["fold"] = fold["fold"]
            vblock["threshold"] = float(threshold)
            validation_frames.append(vblock)
        if test_rows.sum() > 0:
            probability = booster.predict_proba(matrix[test_rows])
            report["test_rows"] = int(test_rows.sum())
            report["test_point"] = binary_point_metrics(
                targets[test_rows], probability, np.ones(int(test_rows.sum()), dtype=bool), threshold
            )
            block = table.loc[test_rows, ["device_mac", "session_id", "cow_id", "center_time_ms"]].copy()
            block[f"target_{task}"] = targets[test_rows]
            block[f"mask_{task}"] = 1
            block[f"prob_{task}"] = probability
            block["fold"] = fold["fold"]
            block["threshold"] = float(threshold)
            pooled_frames.append(block)
        else:
            report["test_rows"] = 0
        fold_reports.append(report)

    pooled: dict[str, object] = {"available": False}
    predictions = pd.DataFrame()
    if pooled_frames:
        predictions = pd.concat(pooled_frames, ignore_index=True)
        # Each fold keeps its own threshold, so binarise before pooling.
        binary = (
            predictions[f"prob_{task}"].to_numpy() >= predictions["threshold"].to_numpy()
        ).astype(np.uint8)
        pooled_frame = predictions.copy()
        # Feed the event matcher a probability column already comparable to 0.5.
        pooled_frame[f"prob_{task}"] = binary.astype(float)
        pooled = {
            "available": True,
            "rows": int(len(predictions)),
            "point": binary_point_metrics(
                predictions[f"target_{task}"].to_numpy(),
                predictions[f"prob_{task}"].to_numpy(),
                np.ones(len(predictions), dtype=bool),
                float(np.median(predictions["threshold"])),
            ),
        }
        if task in EVENT_CODES:
            pooled["event"] = event_level_metrics(
                pooled_frame, task, 0.5, annotations=annotations
            )
    validation_predictions = (
        pd.concat(validation_frames, ignore_index=True) if validation_frames else pd.DataFrame()
    )
    return {
        "task": task,
        "folds": fold_reports,
        "pooled": pooled,
        "feature_importance_top": importance,
        "predictions": predictions,
        "validation_predictions": validation_predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the feature branch under LOCO")
    parser.add_argument("--feature-table", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=None)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", type=str, default=None, choices=[None, "xgboost", "lightgbm", "sklearn"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--tasks", type=str, default="all")
    parser.add_argument(
        "--rate-constrained-threshold",
        action="store_true",
        help="constrain threshold search by the physiological event rate (off by default)",
    )
    args = parser.parse_args()

    table = load_table(args.feature_table)
    features = feature_columns(table)
    if not features:
        raise SystemExit("no numeric feature columns found; did build_feature_table.py run?")
    annotations = (
        pd.read_csv(args.annotations, encoding="utf-8-sig") if args.annotations else None
    )
    folds = build_folds(table, args.splits, args.seed)
    available = task_targets(table)
    requested = (
        list(available)
        if args.tasks == "all"
        else [t for t in args.tasks.split(",") if t in available]
    )
    config = BoosterConfig(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        device=args.device,
        random_state=args.seed,
    )
    print(f"[info] rows={len(table)} features={len(features)} folds={len(folds)}")
    print(f"[info] backends available: {available_backends()} -> using {args.backend or 'first'}")

    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    reports: dict[str, object] = {}
    event_ap: dict[str, float | None] = {}
    for task in requested:
        task_started = time.perf_counter()
        result = run_task(
            task, table, features, folds, config, args.backend, annotations,
            rate_constrained_threshold=args.rate_constrained_threshold,
        )
        predictions = result.pop("predictions")
        validation_predictions = result.pop("validation_predictions", pd.DataFrame())
        if len(predictions):
            predictions.to_csv(
                args.out / f"predictions_{task}.csv", index=False, encoding="utf-8-sig"
            )
        if len(validation_predictions):
            validation_predictions.to_csv(
                args.out / f"validation_predictions_{task}.csv", index=False, encoding="utf-8-sig"
            )
        reports[task] = result
        point = (result.get("pooled") or {}).get("point") or {}
        if task in EVENT_CODES:
            event_ap[task] = point.get("average_precision")
        print(
            f"[task] {task:14s} pooled_AP={point.get('average_precision')} "
            f"pooled_F1={point.get('f1')} ({time.perf_counter() - task_started:.1f}s)"
        )

    posture = ((reports.get("POSTURE_LYING") or {}).get("pooled") or {}).get("point") or {}
    walking = ((reports.get("WALKING") or {}).get("pooled") or {}).get("point") or {}
    summary = {
        "feature_table": str(args.feature_table),
        "rows": int(len(table)),
        "features": len(features),
        "folds": [{"fold": f["fold"], "test_cow": f["test_cow"]} for f in folds],
        "selection": selection_score(
            posture.get("f1"), walking.get("average_precision"), event_ap
        ),
        "tasks": reports,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    with (args.out / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=float)

    rows = []
    for task, result in reports.items():
        for fold in result["folds"]:
            row = {"task": task, **{k: v for k, v in fold.items() if not isinstance(v, dict)}}
            for prefix in ("validation_point", "test_point"):
                block = fold.get(prefix) or {}
                for key in ("precision", "recall", "f1", "average_precision", "positive", "evaluated"):
                    row[f"{prefix}_{key}"] = block.get(key)
            rows.append(row)
    pd.DataFrame(rows).to_csv(args.out / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"[done] {args.out / 'report.json'} in {summary['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
