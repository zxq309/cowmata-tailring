"""Create gap-safe session arrays and a compact 2 Hz sample index."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from cattle_imu.io import load_v2_json
from cattle_imu.preprocessing import (
    cache_key,
    resample_session,
    save_processed_session,
    supervised_sample_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument(
        "--review-coverage",
        type=Path,
        help="event-specific exhaustive video-review ranges; strongly recommended for new caches",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    annotations = pd.read_csv(args.annotations, encoding="utf-8-sig")
    review_coverage = (
        pd.read_csv(args.review_coverage, encoding="utf-8-sig")
        if args.review_coverage is not None else None
    )
    with args.calibration_manifest.open("r", encoding="utf-8") as handle:
        calibration = json.load(handle)
    calibrations = {
        (str(item["device_mac"]), str(item["session_id"])): item
        for item in calibration["sessions"]
    }
    run_dir = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_root = run_dir / "session_cache"
    cache_root.mkdir(parents=True, exist_ok=False)
    sample_frames: list[pd.DataFrame] = []
    session_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    groups = list(annotations.groupby(["device_mac", "session_id"], sort=True))
    for ordinal, ((device_mac, session_id), group) in enumerate(groups, start=1):
        item = calibrations.get((str(device_mac), str(session_id)))
        if item is None:
            failures.append({"device_mac": str(device_mac), "session_id": str(session_id), "error": "missing calibration"})
            continue
        if not item["eligible_for_training"]:
            session_rows.append(
                {
                    "device_mac": device_mac,
                    "session_id": session_id,
                    "status": "excluded_sensor_qc",
                    "issues": ";".join(item.get("issues", [])),
                    "samples": 0,
                }
            )
            continue
        try:
            session = load_v2_json(item["path"])
            processed = resample_session(
                session,
                acc_divisor=float(item["acc_divisor"]),
                acc_bias_counts=item["acc_bias_counts"],
                gyro_divisor=float(item["gyro_divisor"]),
                gyro_bias_counts=item["gyro_bias_counts"],
                mag_divisor=float(item["mag_divisor"]),
            )
            name = cache_key(str(device_mac), str(session_id), str(item["path"]))
            cow_values = sorted(value for value in group["cow_id"].dropna().astype(str).unique() if value)
            if len(cow_values) != 1:
                raise ValueError(f"expected one normalized cow_id, got {cow_values}")
            device_keys = sorted(group["device_key"].astype(str).unique())
            if len(device_keys) != 1:
                raise ValueError(f"expected one device_key, got {device_keys}")
            frame = supervised_sample_frame(
                processed=processed,
                annotations=group,
                cache_name=name,
                cow_id=cow_values[0],
                device_key=device_keys[0],
                device_mac=str(device_mac),
                session_id=str(session_id),
                review_coverage=(
                    review_coverage[
                        (review_coverage["device_mac"].astype(str) == str(device_mac))
                        & (review_coverage["session_id"].astype(str) == str(session_id))
                    ]
                    if review_coverage is not None else None
                ),
            )
            save_processed_session(
                processed,
                cache_root / name,
                {
                    "cache_key": name,
                    "raw_path": item["path"],
                    "cow_id": cow_values[0],
                    "device_key": device_keys[0],
                    "device_mac": str(device_mac),
                    "session_id": str(session_id),
                    "feature_channels": [
                        "acc_x_g", "acc_y_g", "acc_z_g",
                        "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
                        "mag_x_imu_gauss", "mag_y_imu_gauss", "mag_z_imu_gauss",
                        "acc_norm_g", "gyro_norm_dps", "mag_norm_gauss", "timing_quality_flag",
                    ],
                    "label_contract": "timestamp mother labels; causal context selected at training time",
                    "target_hz": 50,
                    "decision_hz": 2,
                    "calibration": item,
                },
            )
            sample_frames.append(frame)
            session_rows.append(
                {
                    "device_mac": device_mac,
                    "session_id": session_id,
                    "cow_id": cow_values[0],
                    "cache_key": name,
                    "status": "included",
                    "issues": "",
                    "raw_frames": int(session.raw_values.shape[0]),
                    "resampled_frames": int(processed.features.shape[0]),
                    "continuous_segments": len(processed.segments),
                    "samples": len(frame),
                }
            )
        except Exception as exc:
            failures.append(
                {"device_mac": str(device_mac), "session_id": str(session_id), "error": f"{type(exc).__name__}: {exc}"}
            )
        print(f"Cache {ordinal}/{len(groups)}", flush=True)

    samples = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    if not samples.empty:
        samples.insert(0, "sample_id", [f"SMP-{index:07d}" for index in range(1, len(samples) + 1)])
    samples.to_csv(run_dir / "samples.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(session_rows).to_csv(run_dir / "sessions.csv", index=False, encoding="utf-8-sig")
    with (run_dir / "failures.json").open("w", encoding="utf-8") as handle:
        json.dump(failures, handle, ensure_ascii=False, indent=2)
    body_counts = samples["body_target"].value_counts().sort_index().to_dict() if not samples.empty else {}
    event_positive_counts = {
        column.removeprefix("event_"): int(samples[column].sum())
        for column in samples.columns
        if column.startswith("event_")
    }
    summary = {
        "run_dir": str(run_dir),
        "annotation_sessions": len(groups),
        "included_sessions": sum(row["status"] == "included" for row in session_rows),
        "excluded_sensor_qc_sessions": sum(row["status"] == "excluded_sensor_qc" for row in session_rows),
        "samples": len(samples),
        "body_target_counts": {str(key): int(value) for key, value in body_counts.items()},
        "event_positive_window_counts": event_positive_counts,
        "failures": len(failures),
        "gap_policy": "Split at dt<=0 or dt>40 ms; resample each run independently to 50 Hz; causal contexts never cross runs.",
        "event_negative_policy": (
            "Event-specific zeros are supervised only in exhaustive review_coverage ranges."
            if review_coverage is not None
            else "Legacy proxy: zeros are supervised in explicit body-state intervals; add review_coverage before deployment false-alarm claims."
        ),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
