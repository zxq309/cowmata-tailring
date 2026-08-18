"""Train the full-cow GBDT bundle used for annotation-assistance inference."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cattle_imu.annotations import BODY_CODES, EVENT_CODES
from cattle_imu.features import feature_columns
from cattle_imu.gbdt import BinaryBooster, BoosterConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-table",
        type=Path,
        default=PROJECT_ROOT / "runs" / "feature_table" / "feature_table.parquet",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="new output directory; defaults to a timestamped runs/full_gbdt directory",
    )
    parser.add_argument("--backend", choices=("xgboost", "histgb"), default="xgboost")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--n-estimators", type=int, default=400)
    args = parser.parse_args()

    table = pd.read_parquet(args.feature_table)
    features = feature_columns(table)
    matrix = table[features].to_numpy(np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    body = table["body_target"].to_numpy(np.int64)
    labelled = (body >= 0).astype(np.float32)
    tasks: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "POSTURE_LYING": ((body == BODY_CODES.index("LYING")).astype(np.int8), labelled),
        "WALKING": ((body == BODY_CODES.index("WALKING")).astype(np.int8), labelled),
    }
    for code in EVENT_CODES:
        tasks[code] = (
            table[f"event_{code}"].to_numpy(np.int8),
            table[f"mask_{code}"].to_numpy(np.float32),
        )

    output_dir = args.out or (
        PROJECT_ROOT / "runs" / "full_gbdt" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    models: dict[str, BinaryBooster] = {}
    summary: dict[str, dict[str, int | str]] = {}
    for task, (target, mask) in tasks.items():
        rows = mask > 0
        positives = int(target[rows].sum())
        if int(rows.sum()) < 50 or positives < 3:
            summary[task] = {"status": "skipped", "positives": positives}
            print(f"{task:14s} skipped: only {positives} positive rows")
            continue
        booster = BinaryBooster(
            BoosterConfig(device=args.device, n_estimators=args.n_estimators),
            backend=args.backend,
        )
        booster.fit(matrix[rows], target[rows])
        models[task] = booster
        summary[task] = {
            "status": "trained",
            "positives": positives,
            "negatives": int(rows.sum() - positives),
        }
        print(f"{task:14s} trained: {positives} positive rows")

    model_path = output_dir / "gbdt_full.joblib"
    joblib.dump({"models": models, "features": features}, model_path)
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "feature_table": str(args.feature_table),
                "backend": args.backend,
                "device": args.device,
                "models": summary,
                "output": str(model_path),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"saved {len(models)} task models -> {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
