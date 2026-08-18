"""Tune the causal state machine on validation and apply once to test."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from cattle_imu.state_machine import apply_state_machine, state_machine_metrics, tune_state_machine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-run", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    reports: list[dict[str, object]] = []
    for model_run in args.model_run:
        with (model_run / "report.json").open("r", encoding="utf-8") as handle:
            model_report = json.load(handle)
        validation = pd.read_csv(model_run / "validation_predictions.csv", encoding="utf-8-sig")
        test = pd.read_csv(model_run / "test_predictions.csv", encoding="utf-8-sig")
        config, search = tune_state_machine(validation)
        validation_sm = apply_state_machine(validation, config)
        test_sm = apply_state_machine(test, config)
        fold_dir = run_dir / f"fold_{model_report['fold']}_{model_report['test_cow']}"
        fold_dir.mkdir()
        validation_sm.to_csv(fold_dir / "validation_state_machine.csv", index=False, encoding="utf-8-sig")
        test_sm.to_csv(fold_dir / "test_state_machine.csv", index=False, encoding="utf-8-sig")
        search.to_csv(fold_dir / "validation_grid_search.csv", index=False, encoding="utf-8-sig")
        result = {
            "fold": model_report["fold"],
            "test_cow": model_report["test_cow"],
            "source_model_run": str(model_run),
            "config_source": "validation predictions only",
            "config": config.to_dict(),
            "validation": state_machine_metrics(validation_sm),
            "test": state_machine_metrics(test_sm),
        }
        with (fold_dir / "report.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        reports.append(result)
    summary = {
        "run_dir": str(run_dir),
        "folds": reports,
        "important_scope": "This state machine constrains upright versus lying. It does not invent missing labels or improve rare-event truth coverage.",
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
