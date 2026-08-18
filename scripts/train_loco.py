"""Train the shared causal model under a development-all or strict LOCO split."""

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
from cattle_imu.annotations import EVENT_CODES, POSTURE_CODES
from cattle_imu.dataset import FEATURE_MODES, WindowDataset, session_subset
from cattle_imu.load_control import DutyThrottle, duty_cycle_from_env
from cattle_imu.metrics import binary_point_metrics, body_metrics, choose_threshold, event_level_metrics
from cattle_imu.model import build_model, parameter_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--feature-mode", choices=tuple(FEATURE_MODES), default="accgyro8")
    parser.add_argument("--window-mode", choices=("causal", "offline"), default="causal")
    parser.add_argument("--context-seconds", type=float, default=40.96)
    parser.add_argument("--sample-rate-hz", type=int, default=50)
    parser.add_argument("--max-rotation-degrees", type=float, default=20.0)
    parser.add_argument("--max-pos-weight", type=float, default=10.0)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Input shapes are fixed (drop_last=True + constant context length), so
    # cuDNN can safely autotune the fastest convolution algorithm.  The old
    # deterministic=True forced slow conv kernels and was the largest per-op
    # slowdown; run-to-run float differences are negligible for this task.
    torch.backends.cudnn.benchmark = True


def deterministic_cap(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if maximum <= 0 or len(frame) <= maximum:
        return frame.reset_index(drop=True)
    return frame.sample(maximum, random_state=seed).sort_values("sample_id").reset_index(drop=True)


def resolve_data_path(value: str, manifest_path: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    for base in (Path.cwd(), *manifest_path.resolve().parents):
        resolved = base / candidate
        if resolved.exists():
            return resolved
    return candidate


def positive_weights(frame: pd.DataFrame, maximum: float) -> tuple[torch.Tensor, torch.Tensor]:
    body = frame["body_target"].to_numpy(np.int64)
    body_valid = body >= 0
    walking_positive = int(np.sum(body[body_valid] == 2))
    walking_negative = int(np.sum(body_valid) - walking_positive)
    walking = torch.tensor(float(np.clip(walking_negative / max(walking_positive, 1), 1.0, maximum)))
    events: list[float] = []
    for code in EVENT_CODES:
        mask = frame[f"mask_{code}"].to_numpy(bool)
        target = frame[f"event_{code}"].to_numpy(np.uint8)
        positive = int(target[mask].sum())
        negative = int(mask.sum() - positive)
        events.append(float(np.clip(negative / max(positive, 1), 1.0, maximum)))
    return walking, torch.tensor(events, dtype=torch.float32)


def masked_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    *,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    raw = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    return (raw * masks).sum() / masks.sum().clamp_min(1.0)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    policy: object,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    walking_pos_weight: torch.Tensor,
    event_pos_weight: torch.Tensor,
    throttle: DutyThrottle | None = None,
    epoch: int | None = None,
    progress_every: int = 250,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    totals = 0.0
    count = 0
    started = time.perf_counter()
    total_batches = len(loader)
    phase = "train" if training else "validation"
    collected: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "posture_target", "posture_probability", "locomotion_target",
            "locomotion_probability", "locomotion_mask", "event_target",
            "event_probability", "event_mask", "row_index"
        )
    }
    for batch_index, batch in enumerate(loader, start=1):
        inputs = batch["inputs"].to(device, non_blocking=True)
        posture = batch["posture_target"].to(device, non_blocking=True)
        locomotion = batch["locomotion_target"].to(device, non_blocking=True)
        locomotion_mask = batch["locomotion_mask"].to(device, non_blocking=True)
        events = batch["event_target"].to(device, non_blocking=True)
        event_mask = batch["event_mask"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        context = torch.enable_grad() if training else torch.inference_mode()
        with context:
            with autocast_context(policy):
                output = model(inputs)
                posture_valid = posture >= 0
                posture_loss = (
                    nn.functional.cross_entropy(output["posture_logits"][posture_valid], posture[posture_valid])
                    if torch.any(posture_valid)
                    else output["posture_logits"].sum() * 0.0
                )
                locomotion_loss = masked_bce(
                    output["locomotion_logits"].squeeze(1), locomotion, locomotion_mask,
                    pos_weight=walking_pos_weight,
                )
                event_loss = masked_bce(
                    output["event_logits"], events, event_mask, pos_weight=event_pos_weight
                )
                loss = posture_loss + locomotion_loss + event_loss
            if training:
                assert optimizer is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                if throttle is not None:
                    throttle.tick()
        batch_size = inputs.shape[0]
        totals += float(loss.detach()) * batch_size
        count += batch_size
        if not training:
            # Predictions are only consumed by evaluation_metrics(); collecting
            # them on every training batch forces a GPU->CPU sync + copy per
            # batch and stalls the pipeline.  Only materialise them during eval.
            collected["posture_target"].append(posture.detach().cpu().numpy())
            collected["posture_probability"].append(torch.softmax(output["posture_logits"].detach(), 1).float().cpu().numpy())
            collected["locomotion_target"].append(locomotion.detach().cpu().numpy())
            collected["locomotion_probability"].append(torch.sigmoid(output["locomotion_logits"].detach()).squeeze(1).float().cpu().numpy())
            collected["locomotion_mask"].append(locomotion_mask.detach().cpu().numpy())
            collected["event_target"].append(events.detach().cpu().numpy())
            collected["event_probability"].append(torch.sigmoid(output["event_logits"].detach()).float().cpu().numpy())
            collected["event_mask"].append(event_mask.detach().cpu().numpy())
            collected["row_index"].append(batch["row_index"].numpy())
        if progress_every > 0 and (batch_index % progress_every == 0 or batch_index == total_batches):
            elapsed = time.perf_counter() - started
            eta = elapsed / batch_index * (total_batches - batch_index)
            print(json.dumps({
                "epoch": epoch,
                "phase": phase,
                "batch": batch_index,
                "batches": total_batches,
                "loss": round(totals / count, 5),
                "elapsed_s": round(elapsed, 1),
                "eta_s": round(eta, 1),
            }, ensure_ascii=False), flush=True)
    if count == 0:
        raise ValueError("data loader produced no batches")
    result: dict[str, object] = {"loss": totals / count}
    if not training:
        result.update({key: np.concatenate(value) for key, value in collected.items()})
    return result


def evaluation_metrics(
    result: dict[str, object], thresholds: dict[str, float] | None = None
) -> tuple[dict[str, object], dict[str, float]]:
    posture = body_metrics(result["posture_target"], result["posture_probability"])
    selected = {} if thresholds is None else dict(thresholds)
    if thresholds is None:
        selected["WALKING"] = choose_threshold(
            result["locomotion_target"], result["locomotion_probability"], result["locomotion_mask"]
        )
    walking = binary_point_metrics(
        result["locomotion_target"], result["locomotion_probability"], result["locomotion_mask"], selected["WALKING"]
    )
    events: dict[str, object] = {}
    for index, code in enumerate(EVENT_CODES):
        if thresholds is None:
            selected[code] = choose_threshold(
                result["event_target"][:, index], result["event_probability"][:, index], result["event_mask"][:, index]
            )
        events[code] = binary_point_metrics(
            result["event_target"][:, index], result["event_probability"][:, index],
            result["event_mask"][:, index], selected[code],
        )
    scores = [posture["macro_f1"], walking["f1"]] + [value["f1"] for value in events.values()]
    reportable = [float(value) for value in scores if value is not None]
    return {
        "posture": posture,
        "walking": walking,
        "events": events,
        "selection_score": float(np.mean(reportable)) if reportable else -float("inf"),
    }, selected


def prediction_frame(samples: pd.DataFrame, result: dict[str, object]) -> pd.DataFrame:
    order = np.asarray(result["row_index"], dtype=np.int64)
    output = samples.iloc[order][
        ["sample_id", "cow_id", "device_key", "device_mac", "session_id", "center_time_ms"]
    ].reset_index(drop=True)
    output["posture_target"] = result["posture_target"]
    output["posture_pred"] = np.argmax(result["posture_probability"], axis=1)
    for index, code in enumerate(POSTURE_CODES):
        output[f"prob_posture_{code}"] = result["posture_probability"][:, index]
    output["target_WALKING"] = result["locomotion_target"].astype(np.uint8)
    output["mask_WALKING"] = result["locomotion_mask"].astype(np.uint8)
    output["prob_WALKING"] = result["locomotion_probability"]
    for index, code in enumerate(EVENT_CODES):
        output[f"target_{code}"] = result["event_target"][:, index].astype(np.uint8)
        output[f"mask_{code}"] = result["event_mask"][:, index].astype(np.uint8)
        output[f"prob_{code}"] = result["event_probability"][:, index]
    return output


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    policy = resolve_cuda_precision(args.precision)
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    with args.folds.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    fold = next(item for item in manifest["folds"] if int(item["fold"]) == args.fold)
    samples_path = resolve_data_path(manifest["samples_path"], args.folds)
    cache_root = resolve_data_path(manifest["cache_root"], args.folds)
    samples = pd.read_csv(samples_path, encoding="utf-8-sig")
    train = deterministic_cap(session_subset(samples, fold["train_sessions"]), args.max_train_samples, args.seed)
    validation = deterministic_cap(session_subset(samples, fold["validation_sessions"]), args.max_eval_samples, args.seed + 1)
    if train.empty or validation.empty:
        raise ValueError("training and validation splits must each contain samples")
    context_samples = int(round(args.context_seconds * args.sample_rate_hz))
    dataset_options = {
        "feature_mode": args.feature_mode,
        "context_samples": context_samples,
        # WindowDataset/windowing use causal/centered; build_model uses
        # causal/offline.  Map the model-level mode to the window-level mode.
        "window_mode": "causal" if args.window_mode == "causal" else "centered",
    }
    mean = fold["normalization"]["mean"]
    std = fold["normalization"]["std"]
    train_dataset = WindowDataset(
        train, cache_root, mean, std, augment=True,
        max_rotation_degrees=args.max_rotation_degrees, seed=args.seed, **dataset_options,
    )
    validation_dataset = WindowDataset(validation, cache_root, mean, std, **dataset_options)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, drop_last=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    model = build_model(
        mode=args.window_mode,
        in_channels=len(train_dataset.feature_indices),
        sample_rate_hz=args.sample_rate_hz,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = make_grad_scaler(policy)
    walking_weight, event_weights = positive_weights(train, args.max_pos_weight)
    walking_weight = walking_weight.to(device)
    event_weights = event_weights.to(device)
    run_dir = args.output_root / f"fold_{args.fold}_{fold.get('test_cow', 'development')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    history: list[dict[str, object]] = []
    best_score = -float("inf")
    best_epoch = 0
    stale = 0
    start_epoch = 1
    if args.resume is not None:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        scaler.load_state_dict(resume["scaler_state"])
        start_epoch = int(resume["epoch"]) + 1
        # Preserve the best-so-far state so a resumed run keeps the historical
        # peak instead of resetting it and re-labelling a worse epoch as "best".
        best_score = float(resume.get("best_score", -float("inf")))
        best_epoch = int(resume.get("best_epoch", 0))
        stale = int(resume.get("stale", 0))
        # Seed this run dir with the resumed checkpoint so best.pt always exists
        # for the final evaluation even if no later epoch beats the restored peak.
        torch.save(resume, run_dir / "best.pt")
    started = time.perf_counter()
    throttle = DutyThrottle(duty_cycle_from_env())
    for epoch in range(start_epoch, args.epochs + 1):
        train_result = run_epoch(
            model, train_loader, device, policy, optimizer=optimizer, scaler=scaler,
            walking_pos_weight=walking_weight, event_pos_weight=event_weights, throttle=throttle,
            epoch=epoch,
        )
        validation_result = run_epoch(
            model, validation_loader, device, policy, optimizer=None, scaler=scaler,
            walking_pos_weight=walking_weight, event_pos_weight=event_weights,
            epoch=epoch,
        )
        metrics, thresholds = evaluation_metrics(validation_result)
        score = float(metrics["selection_score"])
        scheduler.step(score)
        row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "validation_loss": validation_result["loss"],
            "validation_score": score,
            "validation_posture_macro_f1": metrics["posture"]["macro_f1"],
            "validation_walking_f1": metrics["walking"]["f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if score > best_score + 1e-4:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "model_class": "CausalMultiTaskTCN" if args.window_mode == "causal" else "OfflineMultiTaskTCN",
                    "model_kwargs": {
                        "in_channels": len(train_dataset.feature_indices),
                        "sample_rate_hz": args.sample_rate_hz,
                        "event_codes": list(EVENT_CODES),
                    },
                    "fold": fold,
                    "epoch": epoch,
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "stale": stale,
                    "thresholds": thresholds,
                    "validation_metrics": metrics,
                    "feature_mode": args.feature_mode,
                    "feature_indices": train_dataset.feature_indices.tolist(),
                    "feature_statistics": {"mean": mean, "std": std},
                    "context_samples": context_samples,
                    "precision": policy.name,
                    "parameter_count": parameter_count(model),
                    "seed": args.seed,
                },
                run_dir / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    validation_result = run_epoch(
        model, validation_loader, device, policy, optimizer=None, scaler=scaler,
        walking_pos_weight=walking_weight, event_pos_weight=event_weights,
    )
    validation_metrics, _ = evaluation_metrics(validation_result, checkpoint["thresholds"])
    validation_predictions = prediction_frame(validation, validation_result)
    validation_predictions.to_csv(run_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    test_metrics: dict[str, object] | None = None
    test_predictions: pd.DataFrame | None = None
    test_sessions = fold.get("test_sessions", [])
    if test_sessions:
        test = deterministic_cap(session_subset(samples, test_sessions), args.max_eval_samples, args.seed + 2)
        if test.empty:
            raise ValueError("declared test split contains no samples")
        test_dataset = WindowDataset(test, cache_root, mean, std, **dataset_options)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
        test_result = run_epoch(
            model, test_loader, device, policy, optimizer=None, scaler=scaler,
            walking_pos_weight=walking_weight, event_pos_weight=event_weights,
        )
        test_metrics, _ = evaluation_metrics(test_result, checkpoint["thresholds"])
        test_predictions = prediction_frame(test, test_result)
        test_predictions.to_csv(run_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
    report = {
        "status": "complete",
        "protocol": manifest.get("protocol", "strict_loco"),
        "purpose": manifest.get("purpose", "cross-cow evaluation"),
        "fold": args.fold,
        "test_cow": fold.get("test_cow"),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "parameter_count": parameter_count(model),
        "feature_mode": args.feature_mode,
        "window_mode": args.window_mode,
        "context_seconds": args.context_seconds,
        "precision": policy.name,
        "train_samples": len(train),
        "validation_samples": len(validation),
        "thresholds": checkpoint["thresholds"],
        "validation": validation_metrics,
        "test": test_metrics,
        "validation_event_level": {
            code: event_level_metrics(validation_predictions, code, checkpoint["thresholds"][code])
            for code in EVENT_CODES
        },
        "test_event_level": (
            {
                code: event_level_metrics(test_predictions, code, checkpoint["thresholds"][code])
                for code in EVENT_CODES
            }
            if test_predictions is not None else None
        ),
        "limitations": manifest.get("limitations", []),
        "environment": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "bf16": torch.cuda.is_bf16_supported(),
        },
    }
    with (run_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
