"""Train an optional causal event specialist under development or LOCO splits.

Use this only after a shared model shows negative transfer for selected events.
The default training path remains ``train_loco.py`` with shared representation.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from cattle_imu.amp import autocast_context, make_grad_scaler, resolve_cuda_precision
from cattle_imu.annotations import EVENT_CODES
from cattle_imu.dataset import EventWindowDataset, FEATURE_MODES, session_subset
from cattle_imu.metrics import binary_point_metrics, choose_threshold, event_level_metrics
from cattle_imu.model import EventOnlyTCN, parameter_count


DEFAULT_EVENTS = ("STANDING_UP", "LYING_DOWN", "URINATION")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--events", nargs="+", choices=EVENT_CODES, default=list(DEFAULT_EVENTS))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--feature-mode", choices=tuple(FEATURE_MODES), default="accgyro8")
    parser.add_argument("--context-seconds", type=float, default=40.96)
    parser.add_argument("--sample-rate-hz", type=int, default=50)
    parser.add_argument("--max-rotation-degrees", type=float, default=20.0)
    parser.add_argument("--max-pos-weight", type=float, default=10.0)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_data_path(value: str, manifest_path: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    for base in (Path.cwd(), *manifest_path.resolve().parents):
        resolved = base / candidate
        if resolved.exists():
            return resolved
    return candidate


def eligible(frame: pd.DataFrame, codes: tuple[str, ...]) -> pd.DataFrame:
    masks = frame[[f"mask_{code}" for code in codes]].to_numpy(np.uint8)
    targets = frame[[f"event_{code}" for code in codes]].to_numpy(np.uint8)
    if np.any((targets > 0) & (masks == 0)):
        raise ValueError("positive event targets must have mask=1")
    return frame.loc[np.any(masks > 0, axis=1)].reset_index(drop=True)


def event_pos_weight(frame: pd.DataFrame, codes: tuple[str, ...], maximum: float) -> torch.Tensor:
    values: list[float] = []
    for code in codes:
        mask = frame[f"mask_{code}"].to_numpy(bool)
        target = frame[f"event_{code}"].to_numpy(np.uint8)
        positive = int(target[mask].sum())
        negative = int(mask.sum() - positive)
        if positive == 0 or negative == 0:
            raise ValueError(f"training needs positive and reviewed-negative rows for {code}")
        values.append(float(np.clip(negative / positive, 1.0, maximum)))
    return torch.tensor(values, dtype=torch.float32)


def run_epoch(
    model: EventOnlyTCN,
    loader: DataLoader,
    device: torch.device,
    policy: object,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    pos_weight: torch.Tensor,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for batch in loader:
        inputs = batch["inputs"].to(device, non_blocking=True)
        event_target = batch["event_target"].to(device, non_blocking=True)
        event_mask = batch["event_mask"].to(device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        context = torch.enable_grad() if training else torch.inference_mode()
        with context:
            with autocast_context(policy):
                logits = model(inputs)
                raw = nn.functional.binary_cross_entropy_with_logits(
                    logits, event_target, pos_weight=pos_weight, reduction="none"
                )
                loss = (raw * event_mask).sum() / event_mask.sum().clamp_min(1.0)
            if training:
                assert optimizer is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
        size = inputs.shape[0]
        total += float(loss.detach()) * size
        count += size
        targets.append(event_target.detach().cpu().numpy())
        probabilities.append(torch.sigmoid(logits.detach()).float().cpu().numpy())
        masks.append(event_mask.detach().cpu().numpy())
        indices.append(batch["row_index"].numpy())
    if count == 0:
        raise ValueError("data loader produced no batches")
    return {
        "loss": total / count,
        "event_target": np.concatenate(targets),
        "event_probability": np.concatenate(probabilities),
        "event_mask": np.concatenate(masks),
        "row_index": np.concatenate(indices),
    }


def metrics(
    result: dict[str, object], codes: tuple[str, ...], thresholds: dict[str, float] | None = None
) -> tuple[dict[str, object], dict[str, float]]:
    chosen = {} if thresholds is None else dict(thresholds)
    output: dict[str, object] = {}
    for index, code in enumerate(codes):
        if thresholds is None:
            chosen[code] = choose_threshold(
                result["event_target"][:, index], result["event_probability"][:, index], result["event_mask"][:, index]
            )
        output[code] = binary_point_metrics(
            result["event_target"][:, index], result["event_probability"][:, index],
            result["event_mask"][:, index], chosen[code],
        )
    scores = [value["f1"] for value in output.values() if value["f1"] is not None]
    return {"events": output, "selection_score": float(np.mean(scores)) if scores else -float("inf")}, chosen


def prediction_frame(samples: pd.DataFrame, result: dict[str, object], codes: tuple[str, ...]) -> pd.DataFrame:
    order = np.asarray(result["row_index"], dtype=np.int64)
    output = samples.iloc[order][
        ["sample_id", "cow_id", "device_key", "device_mac", "session_id", "center_time_ms"]
    ].reset_index(drop=True)
    for index, code in enumerate(codes):
        output[f"target_{code}"] = result["event_target"][:, index].astype(np.uint8)
        output[f"mask_{code}"] = result["event_mask"][:, index].astype(np.uint8)
        output[f"prob_{code}"] = result["event_probability"][:, index]
    return output


def main() -> int:
    args = parse_args()
    codes = tuple(args.events)
    if len(codes) != len(set(codes)):
        raise ValueError("event codes must be unique")
    set_seed(args.seed)
    policy = resolve_cuda_precision(args.precision)
    device = torch.device("cuda:0")
    with args.folds.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    fold = next(item for item in manifest["folds"] if int(item["fold"]) == args.fold)
    samples = pd.read_csv(resolve_data_path(manifest["samples_path"], args.folds), encoding="utf-8-sig")
    cache_root = resolve_data_path(manifest["cache_root"], args.folds)
    train = eligible(session_subset(samples, fold["train_sessions"]), codes)
    validation = eligible(session_subset(samples, fold["validation_sessions"]), codes)
    if train.empty or validation.empty:
        raise ValueError("training and validation require evaluated event rows")
    context_samples = round(args.context_seconds * args.sample_rate_hz)
    options = {
        "feature_mode": args.feature_mode,
        "context_samples": context_samples,
    }
    mean, std = fold["normalization"]["mean"], fold["normalization"]["std"]
    train_dataset = EventWindowDataset(
        train, cache_root, mean, std, codes, augment=True,
        max_rotation_degrees=args.max_rotation_degrees, seed=args.seed, **options,
    )
    validation_dataset = EventWindowDataset(validation, cache_root, mean, std, codes, **options)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(args.seed), **loader_options,
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    model = EventOnlyTCN(
        in_channels=len(train_dataset.feature_indices), event_codes=codes, sample_rate_hz=args.sample_rate_hz
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = make_grad_scaler(policy)
    pos_weight = event_pos_weight(train, codes, args.max_pos_weight).to(device)
    run_dir = args.output_root / f"event_fold_{args.fold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    best_score = -float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, object]] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(model, train_loader, device, policy, optimizer=optimizer, scaler=scaler, pos_weight=pos_weight)
        validation_result = run_epoch(model, validation_loader, device, policy, optimizer=None, scaler=scaler, pos_weight=pos_weight)
        validation_metrics, thresholds = metrics(validation_result, codes)
        score = float(validation_metrics["selection_score"])
        scheduler.step(score)
        row = {"epoch": epoch, "train_loss": train_result["loss"], "validation_loss": validation_result["loss"], "validation_score": score, "seconds": time.perf_counter() - started}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if score > best_score + 1e-4:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
                "model_class": "EventOnlyTCN", "event_codes": list(codes), "fold": fold,
                "epoch": epoch, "thresholds": thresholds, "feature_mode": args.feature_mode,
                "feature_indices": train_dataset.feature_indices.tolist(),
                "feature_statistics": {"mean": mean, "std": std}, "context_samples": context_samples,
                "sample_rate_hz": args.sample_rate_hz, "precision": policy.name,
                "parameter_count": parameter_count(model), "seed": args.seed,
            }, run_dir / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    validation_result = run_epoch(model, validation_loader, device, policy, optimizer=None, scaler=scaler, pos_weight=pos_weight)
    validation_metrics, _ = metrics(validation_result, codes, checkpoint["thresholds"])
    validation_predictions = prediction_frame(validation, validation_result, codes)
    validation_predictions.to_csv(run_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    test_metrics = None
    if fold.get("test_sessions"):
        test = eligible(session_subset(samples, fold["test_sessions"]), codes)
        if not test.empty:
            test_dataset = EventWindowDataset(test, cache_root, mean, std, codes, **options)
            test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
            test_result = run_epoch(model, test_loader, device, policy, optimizer=None, scaler=scaler, pos_weight=pos_weight)
            test_metrics, _ = metrics(test_result, codes, checkpoint["thresholds"])
            test_predictions = prediction_frame(test, test_result, codes)
            test_predictions.to_csv(run_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
            test_metrics["event_level"] = {
                code: event_level_metrics(test_predictions, code, checkpoint["thresholds"][code]) for code in codes
            }
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
    report = {
        "status": "complete", "protocol": manifest.get("protocol", "strict_loco"),
        "task_mode": "optional_event_specialist", "event_codes": list(codes),
        "best_epoch": best_epoch, "parameter_count": parameter_count(model),
        "precision": policy.name, "validation": validation_metrics, "test": test_metrics,
        "thresholds": checkpoint["thresholds"],
        "warning": "Compare against the shared model before keeping a specialist; this run is not automatically preferred.",
    }
    with (run_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
