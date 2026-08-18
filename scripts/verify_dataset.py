"""Structural preflight for packaged caches, labels and split manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cattle_imu.annotations import EVENT_CODES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--split", type=Path, action="append", default=[])
    parser.add_argument("--full-cache-scan", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    data = root / "datasets" / "cowmata_imu"
    annotations_path = data / "annotations/annotations_adjudicated_minimal.csv"
    samples_path = data / "supervised_cache/samples.csv"
    sessions_path = data / "supervised_cache/sessions.csv"
    cache_root = data / "supervised_cache/session_cache"
    annotations = pd.read_csv(annotations_path, encoding="utf-8-sig")
    samples = pd.read_csv(samples_path, encoding="utf-8-sig")
    sessions = pd.read_csv(sessions_path, encoding="utf-8-sig")
    problems: list[str] = []
    warnings: list[str] = []
    if annotations["event_id"].duplicated().any():
        problems.append("duplicate annotation event_id")
    if (annotations["t_end_rel_ms"] <= annotations["t_start_rel_ms"]).any():
        problems.append("non-positive annotation interval")
    required = {"sample_id", "cow_id", "cache_key", "center_index", "body_target"}
    required.update(f"event_{code}" for code in EVENT_CODES)
    required.update(f"mask_{code}" for code in EVENT_CODES)
    missing = required - set(samples.columns)
    if missing:
        problems.append(f"samples.csv missing columns: {sorted(missing)}")
    if samples["sample_id"].duplicated().any():
        problems.append("duplicate sample_id")
    session_items = sessions.to_dict("records")
    unknown_hours = sum(
        int(item["frames"])
        for item in session_items
        if str(item["cow_id"]) == "unknown"
    ) / 50 / 3600
    if unknown_hours:
        warnings.append(
            f"{unknown_hours:.2f} h supervised data has cow_id=unknown; exclude it from LOCO"
        )
    cache_checked = 0
    for item in session_items:
        path = cache_root / str(item["cache_key"]) / "features.npy"
        if not path.exists():
            problems.append(f"missing cache: {path}")
            continue
        if args.full_cache_scan or cache_checked < 5:
            values = np.load(path, mmap_mode="r")
            if values.shape != (int(item["frames"]), 13):
                problems.append(f"cache shape mismatch: {path}")
            if args.full_cache_scan and not np.isfinite(values).all():
                problems.append(f"non-finite cache value: {path}")
            cache_checked += 1
    split_paths = args.split or [
        data / "loco_splits/loco_splits.json",
        data / "development_split/development_all.json",
    ]
    split_reports: list[dict[str, object]] = []
    for path in split_paths:
        if not path.exists():
            warnings.append(f"split manifest not found: {path}")
            continue
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        for fold in manifest["folds"]:
            train = set(fold["train_sessions"])
            validation = set(fold["validation_sessions"])
            test = set(fold.get("test_sessions", []))
            if train & validation or train & test or validation & test:
                problems.append(f"split overlap in {path}, fold {fold['fold']}")
        split_reports.append({"path": str(path), "protocol": manifest.get("protocol", "strict_loco"), "folds": len(manifest["folds"])})
    eligible_column = "sensor_eligible" if "sensor_eligible" in annotations else "sensor_training_eligible"
    eligible = annotations[annotations[eligible_column].astype(bool)]
    event_summary = {
        code: {
            "events": int((eligible["code"] == code).sum()),
            "cows": int(eligible.loc[eligible["code"] == code, "cow_id"].nunique()),
        }
        for code in EVENT_CODES
    }
    report = {
        "status": "PASS" if not problems else "FAIL",
        "root": str(root),
        "annotations": len(annotations),
        "sensor_eligible_annotations": len(eligible),
        "samples": len(samples),
        "supervised_sessions": len(session_items),
        "supervised_hours": sum(int(item["frames"]) for item in session_items) / 50 / 3600,
        "event_summary": event_summary,
        "split_reports": split_reports,
        "warnings": warnings,
        "problems": problems,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
