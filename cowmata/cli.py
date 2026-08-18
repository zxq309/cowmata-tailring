"""Unified COWMATA command-line entry point.

The 20260818 CLI forwarded every subcommand to ``python -m scripts.xxx`` through
``subprocess``.  That made the CLI unable to pass an object, impossible to unit
test, and it discarded the traceback of anything that failed inside the child.
Every subcommand here calls a library function directly; ``scripts/`` keeps thin
``main()`` shims so the documented module paths still work.

Torch is imported lazily, and only by the subcommands that need it, so
``cowmata check-data`` and ``cowmata plan-storage`` run on a machine with no
deep-learning stack at all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "cowmata_imu"
DEFAULT_MODEL = PROJECT_ROOT / "weights" / "deploy" / "gbdt_full.joblib"


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=float))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cowmata", description="COWMATA IMU pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    predict = sub.add_parser("predict", help="predict one cached 50 Hz IMU session")
    predict.add_argument("--cache-key", required=True)
    predict.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    predict.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    predict.add_argument("--deep-checkpoint", type=Path)
    predict.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "predict")
    predict.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="override every per-event threshold; omit to use the bundle's own",
    )
    predict.add_argument("--causal", action="store_true")

    check = sub.add_parser("check-data", help="validate caches, labels and cow splits")
    check.add_argument("--root", type=Path, default=PROJECT_ROOT)
    check.add_argument("--full-cache-scan", action="store_true")

    diagnose = sub.add_parser("diagnose", help="write dataset diagnostics under runs")
    diagnose.add_argument("--root", type=Path, default=PROJECT_ROOT)
    diagnose.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "diagnostics")

    plan = sub.add_parser("plan-storage", help="cache size for a planned collection")
    plan.add_argument("--cows", type=int, required=True)
    plan.add_argument("--days", type=float, required=True)

    splits = sub.add_parser("make-splits", help="build cow-grouped k-fold splits")
    splits.add_argument("--root", type=Path, default=PROJECT_ROOT)
    splits.add_argument("--folds", type=int, default=5)
    splits.add_argument("--validation-fraction", type=float, default=0.2)
    splits.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "splits")

    cache = sub.add_parser("build-cache", help="raw JSON -> schema 2 session cache + labels")
    cache.add_argument("--annotations", type=Path, required=True)
    cache.add_argument("--calibration-manifest", type=Path, required=True)
    cache.add_argument("--output-root", type=Path, required=True)
    cache.add_argument(
        "--review-coverage",
        type=Path,
        help="exhaustive video-review ranges; without it no false-alarm rate may be quoted",
    )
    cache.add_argument(
        "--tail-raised-policy", choices=("derive", "exclude", "legacy"), default="derive"
    )
    cache.add_argument("--tail-position", choices=("root", "mid", "tip", "unknown"), default="unknown")
    cache.add_argument("--limit-sessions", type=int, default=0)

    table = sub.add_parser("build-features", help="cache + labels -> hand-crafted feature table")
    table.add_argument("--samples", type=Path, required=True)
    table.add_argument("--session-cache", type=Path, required=True)
    table.add_argument("--out", type=Path, required=True)
    table.add_argument("--workers", type=int, default=1)
    table.add_argument("--reference-stride", type=int, default=10)
    table.add_argument("--feature-version", type=int, default=None)
    table.add_argument("--no-calibration", action="store_true")
    table.add_argument("--limit-sessions", type=int, default=0)
    window = table.add_mutually_exclusive_group()
    window.add_argument("--offline", action="store_true", help="zero-phase filters, centred windows")
    window.add_argument("--causal", action="store_true", help="one-sided filters, trailing windows")

    gbdt = sub.add_parser("train-gbdt", help="feature table -> GBDT bundle with thresholds")
    gbdt.add_argument("--feature-table", type=Path, required=True)
    gbdt.add_argument("--out", type=Path, default=None)
    gbdt.add_argument("--backend", choices=("xgboost", "lightgbm", "sklearn"), default="xgboost")
    gbdt.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    gbdt.add_argument("--n-estimators", type=int, default=400)
    gbdt.add_argument("--feature-version", type=int, default=None)
    gbdt.add_argument("--validation-fraction", type=float, default=0.25)

    mine = sub.add_parser("mine", help="build a human review queue from predictions")
    mine.add_argument("--predictions", type=Path, nargs="+", required=True)
    mine.add_argument("--events", type=str, required=True)
    mine.add_argument("--per-event", type=int, default=40)
    mine.add_argument("--random-fraction", type=float, default=0.3)
    mine.add_argument("--out", type=Path, required=True)

    train = sub.add_parser("train", help="train the multi-stage temporal model on one fold")
    train.add_argument("--labels", type=Path, required=True)
    train.add_argument("--cache-root", type=Path, required=True)
    train.add_argument("--splits", type=Path, required=True)
    train.add_argument("--fold", type=int, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--mode", choices=("offline", "causal"), default="offline")
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--chunk-steps", type=int, default=1200)
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    train.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    train.add_argument("--annotations", type=Path)

    env = sub.add_parser("check-env", help="report the device, precision and load policy")
    env.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    env.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "predict":
        from .inference import COWMATA

        model = COWMATA(args.model, data_root=args.data_root, deep_checkpoint=args.deep_checkpoint)
        result = model.predict(
            args.cache_key, project=args.out, threshold=args.threshold, causal=args.causal
        )
        _print(
            {
                "session": result.cache_key,
                "feature_version": result.feature_version,
                "points_2hz": int(len(result.dense)),
                "event_candidates": int(len(result.candidates)),
                "thresholds": result.thresholds,
                "dense_probabilities": str(result.dense_path),
                "candidate_intervals": str(result.candidates_path),
            }
        )
        return 0

    if args.command == "check-data":
        from .tools import verify

        report = verify(args.root, full_cache_scan=args.full_cache_scan)
        _print(report)
        return 0 if report["status"] == "PASS" else 1

    if args.command == "diagnose":
        from .tools import diagnose

        payload = diagnose(args.root, out=args.out)
        _print({"out": str(args.out), "cow_balance": payload["cow_balance"].get("largest_share")})
        return 0

    if args.command == "build-cache":
        from .pipelines import build_cache

        _print(
            build_cache(
                annotations=args.annotations,
                calibration_manifest=args.calibration_manifest,
                output_root=args.output_root,
                review_coverage=args.review_coverage,
                tail_raised_policy=args.tail_raised_policy,
                tail_position=args.tail_position,
                limit_sessions=args.limit_sessions,
            )
        )
        return 0

    if args.command == "build-features":
        from .features import FEATURE_VERSION
        from .pipelines import build_features

        manifest = build_features(
            samples=args.samples,
            session_cache=args.session_cache,
            out=args.out,
            causal=bool(args.causal) or not bool(args.offline),
            calibrate=not args.no_calibration,
            reference_stride=args.reference_stride,
            workers=args.workers,
            feature_version=args.feature_version or FEATURE_VERSION,
            limit_sessions=args.limit_sessions,
        )
        manifest.pop("sessions", None)
        _print(manifest)
        return 0

    if args.command == "train-gbdt":
        from .features import FEATURE_VERSION
        from .pipelines import train_gbdt

        version = args.feature_version
        if version is None:
            sidecar = Path(args.feature_table).parent / "feature_manifest.json"
            version = (
                int(json.loads(sidecar.read_text(encoding="utf-8")).get("feature_version", FEATURE_VERSION))
                if sidecar.exists()
                else FEATURE_VERSION
            )
        _print(
            train_gbdt(
                feature_table=args.feature_table,
                out=args.out,
                backend=args.backend,
                device=args.device,
                n_estimators=args.n_estimators,
                feature_version=version,
                validation_fraction=args.validation_fraction,
                project_root=PROJECT_ROOT,
            )
        )
        return 0

    if args.command == "plan-storage":
        from .tools import plan_storage

        _print(plan_storage(args.cows, args.days))
        return 0

    if args.command == "make-splits":
        import pandas as pd

        from .train import grouped_folds

        sessions = pd.read_csv(
            Path(args.root) / "datasets/cowmata_imu/supervised_cache/sessions.csv",
            encoding="utf-8-sig",
        )
        if "status" in sessions:
            sessions = sessions[sessions["status"] == "included"]
        folds = grouped_folds(
            sessions, n_folds=args.folds, validation_fraction=args.validation_fraction
        )
        args.out.mkdir(parents=True, exist_ok=True)
        target = args.out / "splits.json"
        target.write_text(
            json.dumps(
                {"schema_version": 3, "protocol": folds[0]["protocol"], "folds": folds},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _print(
            {
                "out": str(target),
                "protocol": folds[0]["protocol"],
                "folds": [
                    {"fold": f["fold"], "test_cows": len(f["test_cows"])} for f in folds
                ],
            }
        )
        return 0

    if args.command == "mine":
        import pandas as pd

        from .tools import mine_candidates

        frame = pd.concat(
            [pd.read_csv(path, encoding="utf-8-sig") for path in args.predictions],
            ignore_index=True,
        )
        events = [code.strip() for code in args.events.split(",") if code.strip()]
        queue, manifest = mine_candidates(
            frame, events, per_event=args.per_event, random_fraction=args.random_fraction
        )
        args.out.mkdir(parents=True, exist_ok=True)
        queue.to_csv(args.out / "review_queue.csv", index=False, encoding="utf-8-sig")
        (args.out / "review_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _print({"out": str(args.out / "review_queue.csv"), **manifest})
        return 0

    if args.command == "train":
        import pandas as pd

        from .dataset import normalisation_statistics
        from .train import TrainConfig, train_fold

        labels = pd.read_csv(args.labels, encoding="utf-8-sig")
        manifest = json.loads(Path(args.splits).read_text(encoding="utf-8"))
        fold = next(item for item in manifest["folds"] if int(item["fold"]) == args.fold)
        keys = (
            labels.assign(
                _key=labels["device_mac"].astype(str) + "|" + labels["session_id"].astype(str)
            )
            .query("_key in @fold['train_sessions']")["cache_key"]
            .astype(str)
            .unique()
            .tolist()
        )
        mean, std, frames = normalisation_statistics(args.cache_root, keys)
        annotations = (
            pd.read_csv(args.annotations, encoding="utf-8-sig") if args.annotations else None
        )
        config = TrainConfig(
            mode=args.mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            chunk_steps=args.chunk_steps,
            num_workers=args.num_workers,
            precision=args.precision,
            device=args.device,
        )
        report = train_fold(
            labels,
            args.cache_root,
            fold,
            {"mean": mean, "std": std, "frames": frames},
            args.out,
            config,
            annotations=annotations,
        )
        _print(
            {
                "run_dir": str(args.out),
                "fold": report["fold"],
                "best_epoch": report["best_epoch"],
                "selection_score": report["validation"]["selection_score"],
            }
        )
        return 0

    from .runtime import environment_report, resolve_precision

    report = environment_report()
    try:
        report["selected_precision"] = resolve_precision(
            args.precision, device=args.device
        ).name
    except Exception as error:  # noqa: BLE001
        report["selected_precision"] = f"unavailable: {type(error).__name__}: {error}"
    _print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
