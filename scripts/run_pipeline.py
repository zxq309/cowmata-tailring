# -*- coding: utf-8 -*-
"""Cross-platform reproduction driver.

Fixes against the previous ``_run_pipeline.py``:

* every fold in ``loco_splits.json`` is run (the old ``range(1, 6)`` skipped
  fold 6, which is the second largest test cow);
* no hard-coded ``D:/ProgramData/Anaconda3/...`` interpreter - it uses
  ``sys.executable``;
* ``PYTHONPATH`` is joined with ``os.pathsep`` so it works on Linux;
* no ``ctypes.windll`` - the lock uses a portable pid check;
* the GPU power cap is opt-in instead of always applied.

Stages (each can be skipped)::

    python run_pipeline.py --root . --stages diagnose,features,feature_model
    python run_pipeline.py --root . --stages deep_loco --epochs 30
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ALL_STAGES = ("diagnose", "features", "feature_model", "deep_loco", "mine")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":  # pragma: no cover - windows only
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True, check=False
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run(command: list[str], log_path: Path, env: dict[str, str], timeout: int) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{now()}] $ {' '.join(str(c) for c in command)}", flush=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            [str(c) for c in command],
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=timeout,
            check=False,
        )
    elapsed = time.perf_counter() - started
    status = "ok" if process.returncode == 0 else f"FAILED rc={process.returncode}"
    print(f"[{now()}] {status} in {elapsed:.1f}s -> {log_path}", flush=True)
    return process.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Cattle IMU pipeline driver")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--stages", type=str, default=",".join(ALL_STAGES))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--feature-mode", type=str, default="accgyro8")
    parser.add_argument("--window-mode", type=str, default="offline", choices=("causal", "offline"))
    parser.add_argument("--backend", type=str, default="xgboost")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stage-timeout", type=int, default=6 * 3600)
    parser.add_argument("--gpu-power-limit", type=int, default=0, help="watts; 0 disables the cap")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    data = root / "datasets" / "cowmata_imu"
    scripts = root / "scripts"
    out = root / "runs"
    logs = out / "logs"
    out.mkdir(parents=True, exist_ok=True)

    lock = out / "pipeline.lock"
    if lock.exists():
        try:
            owner = int(lock.read_text(encoding="utf-8").strip())
        except ValueError:
            owner = -1
        if process_alive(owner):
            print(f"[{now()}] another pipeline instance (pid {owner}) is running; exiting")
            return 2
    lock.write_text(str(os.getpid()), encoding="utf-8")

    if args.gpu_power_limit > 0:
        subprocess.run(
            ["nvidia-smi", "-pl", str(args.gpu_power_limit)],
            check=False, capture_output=True, timeout=60,
        )

    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root), existing]))
    python = sys.executable

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        raise SystemExit(f"unknown stages: {unknown}; choose from {ALL_STAGES}")

    splits_path = data / "loco_splits" / "loco_splits.json"
    feature_dir = out / "feature_table"
    feature_model_dir = out / "feature_model"
    results: list[dict[str, object]] = []

    def record(name: str, returncode: int) -> bool:
        results.append({"stage": name, "returncode": returncode})
        if returncode != 0 and not args.continue_on_error:
            print(f"[{now()}] stopping: stage {name} failed")
            return False
        return True

    try:
        if "diagnose" in stages:
            rc = run(
                [python, scripts / "diagnose_dataset.py", "--root", root, "--out", out / "diagnostics"],
                logs / "diagnose.log", env, args.stage_timeout,
            )
            if not record("diagnose", rc):
                return 1

        if "features" in stages:
            rc = run(
                [
                    python, scripts / "build_feature_table.py",
                    "--samples", data / "supervised_cache" / "samples.csv",
                    "--session-cache", data / "supervised_cache" / "session_cache",
                    "--out", feature_dir,
                    f"--{'causal' if args.window_mode == 'causal' else 'offline'}",
                    "--workers", str(max(1, args.num_workers)),
                ],
                logs / "features.log", env, args.stage_timeout,
            )
            if not record("features", rc):
                return 1

        if "feature_model" in stages:
            table = feature_dir / "feature_table.parquet"
            if not table.exists():
                table = feature_dir / "feature_table.csv.gz"
            rc = run(
                [
                    python, scripts / "train_feature_model.py",
                    "--feature-table", table,
                    "--splits", splits_path,
                    "--annotations", data / "annotations" / "annotations_adjudicated_minimal.csv",
                    "--out", feature_model_dir,
                    "--backend", args.backend,
                    "--device", args.device,
                ],
                logs / "feature_model.log", env, args.stage_timeout,
            )
            if not record("feature_model", rc):
                return 1

        if "deep_loco" in stages:
            with splits_path.open("r", encoding="utf-8") as handle:
                folds = json.load(handle)["folds"]
            print(f"[{now()}] deep LOCO over {len(folds)} folds: "
                  f"{[f['fold'] for f in folds]}")
            for fold in folds:
                index = int(fold["fold"])
                rc = run(
                    [
                        python, scripts / "train_loco.py",
                        "--folds", splits_path,
                        "--fold", str(index),
                        "--output-root", out / "models" / "loco",
                        "--epochs", str(args.epochs),
                        "--batch-size", str(args.batch_size),
                        "--num-workers", str(args.num_workers),
                        "--feature-mode", args.feature_mode,
                        "--window-mode", args.window_mode,
                    ],
                    logs / f"loco_fold{index}.log", env, args.stage_timeout,
                )
                if not record(f"deep_loco_fold{index}", rc):
                    return 1

        if "mine" in stages:
            predictions = sorted(feature_model_dir.glob("predictions_*.csv"))
            if not predictions:
                print(f"[{now()}] no predictions found in {feature_model_dir}; skipping mining")
            else:
                events = ",".join(
                    path.stem.replace("predictions_", "")
                    for path in predictions
                    if path.stem.replace("predictions_", "") not in {"POSTURE_LYING", "WALKING"}
                )
                rc = run(
                    [
                        python, scripts / "mine_candidates.py",
                        "--predictions", *predictions,
                        "--events", events,
                        "--per-event", "40",
                        "--out", out / f"review_round_{datetime.now():%Y%m%d_%H%M%S}",
                    ],
                    logs / "mine.log", env, args.stage_timeout,
                )
                if not record("mine", rc):
                    return 1
    finally:
        lock.unlink(missing_ok=True)

    summary = {"finished_at": now(), "stages": results}
    with (out / "pipeline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(item["returncode"] == 0 for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
