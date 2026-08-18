"""Unified COWMATA command-line entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from cattle_imu.inference import COWMATA, DEFAULT_DATA_ROOT, DEFAULT_MODEL, PROJECT_ROOT


def _forward(module: str, arguments: Sequence[str]) -> int:
    return subprocess.call([sys.executable, "-m", module, *arguments], cwd=PROJECT_ROOT)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cowmata", description="COWMATA IMU pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="predict one cached 50 Hz IMU session")
    predict.add_argument("--cache-key", required=True)
    predict.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    predict.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    predict.add_argument("--deep-checkpoint", type=Path)
    predict.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "predict")
    predict.add_argument("--threshold", type=float, default=0.5)

    check_data = subparsers.add_parser("check-data", help="validate caches, labels and cow splits")
    check_data.add_argument("--full-cache-scan", action="store_true")

    diagnose = subparsers.add_parser("diagnose", help="write dataset diagnostics under runs")
    diagnose.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "diagnostics")

    pipeline = subparsers.add_parser("pipeline", help="run the reproducible training pipeline")
    pipeline.add_argument("arguments", nargs=argparse.REMAINDER)

    check_env = subparsers.add_parser("check-env", help="run a model forward/backward smoke test")
    check_env.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    check_env.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")

    args = parser.parse_args(argv)
    if args.command == "predict":
        model = COWMATA(
            args.model,
            data_root=args.data_root,
            deep_checkpoint=args.deep_checkpoint,
        )
        result = model.predict(args.cache_key, project=args.out, threshold=args.threshold)
        print(f"session: {result.cache_key}")
        print(f"2 Hz points: {len(result.dense)}")
        print(f"event candidates: {len(result.candidates)}")
        print(f"dense probabilities: {result.dense_path}")
        print(f"candidate intervals: {result.candidates_path}")
        return 0
    if args.command == "check-data":
        forwarded = ["--root", str(PROJECT_ROOT)]
        if args.full_cache_scan:
            forwarded.append("--full-cache-scan")
        return _forward("scripts.verify_dataset", forwarded)
    if args.command == "diagnose":
        return _forward(
            "scripts.diagnose_dataset",
            ["--root", str(PROJECT_ROOT), "--out", str(args.out)],
        )
    if args.command == "pipeline":
        forwarded = list(args.arguments)
        if forwarded[:1] == ["--"]:
            forwarded.pop(0)
        return _forward("scripts.run_pipeline", ["--root", str(PROJECT_ROOT), *forwarded])
    return _forward(
        "scripts.verify_environment",
        ["--device", args.device, "--precision", args.precision],
    )


if __name__ == "__main__":
    raise SystemExit(main())
