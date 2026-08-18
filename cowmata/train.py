"""Dense multi-stage training for the COWMATA temporal model.

This replaces ``scripts/train_loco.py`` and ``scripts/train_event_loco.py`` from
the 20260818 baseline.  Three things are different and each of them was a
defect, not a preference.

**One objective.**  The old trainer computed its own model-selection score by
averaging eight thresholded F1 values with equal weight, while
``metrics.selection_score`` - in the same repository - already implemented an
evidence-weighted average-precision objective and was never called from the
training path.  Two disagreeing definitions of "better" is worse than either.
This trainer calls :func:`cowmata.metrics.selection_score` and nothing else.

**Dense supervision.**  A batch is a contiguous chunk of stream, supervised at
every step where the mask is 1, instead of one 40 s window per label point.
See :mod:`cowmata.dataset` for the arithmetic.

**Grouped folds, not only LOCO.**  With six animals, leave-one-cow-out was the
only option and it produced folds of 13 and 58 samples.  With two hundred, a
five-fold grouped split with forty test animals per fold is both computable and
statistically meaningful; :func:`grouped_folds` builds it, and the manifest
records which protocol was used so a report can never quietly compare the two.

Checkpoints record the mode, the channel layout, the normalisation statistics
and the selected per-event thresholds, so :mod:`cowmata.inference` can rebuild
the exact scoring configuration rather than assuming 0.5.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import EVENT_CODES
from .metrics import (
    binary_point_metrics,
    choose_threshold,
    event_level_metrics,
    multiclass_metrics,
    selection_score,
)


# ==========================================================================
# splits
# ==========================================================================
def session_key(device_mac: object, session_id: object) -> str:
    return f"{device_mac}|{session_id}"


def grouped_folds(
    sessions: pd.DataFrame,
    *,
    n_folds: int = 5,
    validation_fraction: float = 0.2,
    seed: int = 20260819,
) -> list[dict[str, object]]:
    """Cow-grouped k-fold with a cow-disjoint validation split inside each fold.

    Leave-one-cow-out is the special case ``n_folds = n_cows``.  It is the right
    protocol at six animals and the wrong one at two hundred: two hundred
    retrainings buy nothing that forty-animal test folds do not already give,
    and per-fold numbers computed on one animal are noise that later gets
    averaged as if it were evidence.

    Validation cows are held out from the *training* cows, not merely by
    session.  The 20260818 manifest held out whole sessions of animals that were
    also in training, so early stopping and threshold selection saw the identity
    they were later asked to generalise across.
    """

    frame = sessions.copy()
    frame["session_key"] = [
        session_key(d, s) for d, s in zip(frame["device_mac"], frame["session_id"])
    ]
    cows = sorted(frame["cow_id"].astype(str).unique())
    if len(cows) < 2:
        raise ValueError("at least two cows are needed to build a grouped split")
    n_folds = int(min(max(2, n_folds), len(cows)))
    rng = np.random.default_rng(seed)
    order = np.asarray(cows)[rng.permutation(len(cows))]
    blocks = [list(block) for block in np.array_split(order, n_folds)]

    by_cow = {
        str(cow): set(group["session_key"])
        for cow, group in frame.groupby(frame["cow_id"].astype(str))
    }
    folds: list[dict[str, object]] = []
    for index, test_cows in enumerate(blocks, start=1):
        remaining = [cow for cow in cows if cow not in set(test_cows)]
        n_validation = max(1, int(round(len(remaining) * validation_fraction)))
        validation_cows = sorted(remaining[:n_validation])
        train_cows = sorted(set(remaining) - set(validation_cows))
        if not train_cows:
            raise ValueError("validation_fraction leaves no training cows")
        folds.append(
            {
                "fold": index,
                "protocol": "grouped_kfold" if n_folds < len(cows) else "strict_loco",
                "test_cows": sorted(str(c) for c in test_cows),
                "validation_cows": validation_cows,
                "train_cows": train_cows,
                "train_sessions": sorted(set().union(*[by_cow[c] for c in train_cows])),
                "validation_sessions": sorted(
                    set().union(*[by_cow[c] for c in validation_cows])
                ),
                "test_sessions": sorted(set().union(*[by_cow[str(c)] for c in test_cows])),
            }
        )
    return folds


# ==========================================================================
# configuration
# ==========================================================================
@dataclass
class TrainConfig:
    mode: str = "offline"  # "offline" | "causal"
    channel_mode: str = "raw9"
    chunk_steps: int = 1200  # 10 minutes at 2 Hz
    chunk_overlap: int = 120
    batch_size: int = 4
    epochs: int = 40
    patience: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    channels: int = 64
    stage_layers: int = 8
    refinement_stages: int = 3
    dropout: float = 0.1
    max_pos_weight: float = 20.0
    smoothing_weight: float = 0.15
    boundary_weight: float = 1.0
    yaw_degrees: float = 180.0
    tilt_degrees: float = 35.0
    num_workers: int = 4
    precision: str = "auto"
    device: str = "cuda"
    seed: int = 20260819
    grad_clip: float = 5.0

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    validation_loss: float
    selection_score: float
    seconds: float
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "validation_loss": self.validation_loss,
            "selection_score": self.selection_score,
            "seconds": self.seconds,
            **self.extra,
        }


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Chunk lengths are near-constant, so cuDNN autotuning is safe and is worth
    # far more than the run-to-run float noise it introduces.  The 20260818
    # setting (deterministic=True) forced slow convolution kernels and was the
    # single largest per-operation slowdown in the old loop.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


# ==========================================================================
# one epoch
# ==========================================================================
def run_epoch(
    model,
    loader,
    device,
    policy,
    *,
    optimizer=None,
    scaler=None,
    config: TrainConfig,
    event_pos_weight=None,
    locomotion_pos_weight=None,
    boundary_pos_weight=None,
    throttle=None,
    collect: bool = False,
) -> dict[str, object]:
    """One pass over ``loader``. Predictions are collected only when asked.

    Materialising predictions on every training batch forces a GPU->CPU
    synchronisation per batch and stalls the pipeline; they are needed only for
    evaluation, so ``collect`` defaults to False.
    """

    import torch

    from .models import multi_stage_loss
    from .runtime import autocast_context

    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    collected: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "posture_target",
            "posture_probability",
            "locomotion_target",
            "locomotion_probability",
            "locomotion_mask",
            "event_target",
            "event_probability",
            "event_mask",
            "row_start",
            "row_stop",
            "lengths",
        )
    }
    started = time.perf_counter()
    for batch in loader:
        inputs = batch["inputs"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        targets = {
            key: batch[key].to(device, non_blocking=True)
            for key in (
                "posture_target",
                "posture_mask",
                "locomotion_target",
                "locomotion_mask",
                "event_target",
                "event_mask",
                "boundary_target",
                "boundary_mask",
            )
        }
        if training:
            optimizer.zero_grad(set_to_none=True)
        context = torch.enable_grad() if training else torch.inference_mode()
        with context:
            with autocast_context(policy):
                outputs = model(inputs, mask=valid.unsqueeze(1))
                parts = multi_stage_loss(
                    outputs,
                    targets,
                    event_pos_weight=event_pos_weight,
                    locomotion_pos_weight=locomotion_pos_weight,
                    boundary_pos_weight=boundary_pos_weight,
                    smoothing_weight=config.smoothing_weight,
                    boundary_weight=config.boundary_weight,
                )
                loss = parts["total"]
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()
                if throttle is not None:
                    throttle.tick()
        size = int(inputs.shape[0])
        total += float(loss.detach()) * size
        count += size
        if collect:
            keep = valid.detach().cpu().numpy().astype(bool)
            posture = torch.softmax(outputs["posture_logits"][-1].detach(), dim=1)
            events = torch.sigmoid(outputs["event_logits"][-1].detach())
            walking = torch.sigmoid(outputs["locomotion_logits"][-1].detach()).squeeze(1)
            collected["posture_target"].append(
                batch["posture_target"].numpy()[keep].reshape(-1)
            )
            collected["posture_probability"].append(
                posture.float().cpu().numpy().transpose(0, 2, 1)[keep]
            )
            collected["locomotion_target"].append(
                batch["locomotion_target"].numpy()[keep].reshape(-1)
            )
            collected["locomotion_probability"].append(walking.float().cpu().numpy()[keep])
            collected["locomotion_mask"].append(batch["locomotion_mask"].numpy()[keep])
            collected["event_target"].append(
                batch["event_target"].numpy().transpose(0, 2, 1)[keep]
            )
            collected["event_probability"].append(
                events.float().cpu().numpy().transpose(0, 2, 1)[keep]
            )
            collected["event_mask"].append(batch["event_mask"].numpy().transpose(0, 2, 1)[keep])
            collected["row_start"].append(batch["row_start"].numpy())
            collected["row_stop"].append(batch["row_stop"].numpy())
            collected["lengths"].append(batch["lengths"].numpy())
    if count == 0:
        raise ValueError("data loader produced no batches")
    result: dict[str, object] = {"loss": total / count, "seconds": time.perf_counter() - started}
    if collect:
        result.update({key: np.concatenate(value) for key, value in collected.items() if value})
    return result


def evaluation_metrics(
    result: dict[str, object], thresholds: dict[str, float] | None = None
) -> tuple[dict[str, object], dict[str, float]]:
    """Point metrics plus the single project-wide selection objective."""

    posture = multiclass_metrics(result["posture_target"], result["posture_probability"])
    selected = {} if thresholds is None else dict(thresholds)
    if thresholds is None:
        selected["WALKING"] = choose_threshold(
            result["locomotion_target"],
            result["locomotion_probability"],
            result["locomotion_mask"],
        )
    walking = binary_point_metrics(
        result["locomotion_target"],
        result["locomotion_probability"],
        result["locomotion_mask"],
        selected["WALKING"],
    )
    events: dict[str, object] = {}
    event_ap: dict[str, float | None] = {}
    for index, code in enumerate(EVENT_CODES):
        if thresholds is None:
            selected[code] = choose_threshold(
                result["event_target"][:, index],
                result["event_probability"][:, index],
                result["event_mask"][:, index],
            )
        report = binary_point_metrics(
            result["event_target"][:, index],
            result["event_probability"][:, index],
            result["event_mask"][:, index],
            selected[code],
        )
        events[code] = report
        event_ap[code] = report.get("average_precision")
    objective = selection_score(
        posture.get("macro_f1"), walking.get("average_precision"), event_ap
    )
    score = objective["selection_score"]
    return (
        {
            "posture": posture,
            "walking": walking,
            "events": events,
            "objective": objective,
            "selection_score": float(score) if score is not None else -math.inf,
        },
        selected,
    )


def prediction_frame(
    labels: pd.DataFrame, result: dict[str, object], chunks: list[tuple[int, int]]
) -> pd.DataFrame:
    """Map dense per-step predictions back onto the label rows they came from.

    Chunks overlap, so a row can be predicted more than once; the *last*
    prediction wins, which is the one produced with the most left context.
    """

    row_starts = np.asarray(result["row_start"], dtype=np.int64)
    row_stops = np.asarray(result["row_stop"], dtype=np.int64)
    rows = np.concatenate([np.arange(a, b) for a, b in zip(row_starts, row_stops)])
    identity = [
        column
        for column in ("cow_id", "device_key", "device_mac", "session_id", "center_time_ms")
        if column in labels.columns
    ]
    out = labels.iloc[rows][identity].reset_index(drop=True)
    out["posture_target"] = result["posture_target"]
    out["posture_pred"] = np.argmax(result["posture_probability"], axis=1)
    out["prob_posture_UPRIGHT"] = result["posture_probability"][:, 0]
    out["prob_posture_LYING"] = result["posture_probability"][:, 1]
    out["target_WALKING"] = result["locomotion_target"].astype(np.uint8)
    out["mask_WALKING"] = result["locomotion_mask"].astype(np.uint8)
    out["prob_WALKING"] = result["locomotion_probability"]
    for index, code in enumerate(EVENT_CODES):
        out[f"target_{code}"] = result["event_target"][:, index].astype(np.uint8)
        out[f"mask_{code}"] = result["event_mask"][:, index].astype(np.uint8)
        out[f"prob_{code}"] = result["event_probability"][:, index]
    out["_row"] = rows
    return out.drop_duplicates("_row", keep="last").drop(columns=["_row"]).reset_index(drop=True)


# ==========================================================================
# the fold driver
# ==========================================================================
def train_fold(
    labels: pd.DataFrame,
    cache_root: str | Path,
    fold: dict[str, object],
    normalisation: dict[str, list[float]],
    output_dir: str | Path,
    config: TrainConfig | None = None,
    *,
    annotations: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Train and evaluate one fold. Returns the report and writes it to disk."""

    import torch
    from torch.utils.data import DataLoader

    from .dataset import (
        CHANNEL_MODES,
        DenseSegmentDataset,
        collate_chunks,
        event_pos_weight as compute_event_pos_weight,
        session_subset,
    )
    from .models import build_model, parameter_count
    from .runtime import (
        DutyThrottle,
        autocast_context,  # noqa: F401  (re-exported for run_epoch's import path)
        dataloader_options,
        environment_report,
        make_grad_scaler,
        resolve_precision,
    )

    cfg = config or TrainConfig()
    set_seed(cfg.seed)
    policy = resolve_precision(cfg.precision, device=cfg.device)
    device = torch.device("cuda:0" if policy.device == "cuda" else "cpu")

    train_labels = session_subset(labels, fold["train_sessions"])
    validation_labels = session_subset(labels, fold["validation_sessions"])
    if train_labels.empty or validation_labels.empty:
        raise ValueError("training and validation splits must each contain label rows")

    dataset_options = {
        "chunk_steps": cfg.chunk_steps,
        "chunk_overlap": cfg.chunk_overlap,
        "channel_mode": cfg.channel_mode,
    }
    mean, std = normalisation["mean"], normalisation["std"]
    train_dataset = DenseSegmentDataset(
        train_labels,
        cache_root,
        mean,
        std,
        augment=True,
        yaw_degrees=cfg.yaw_degrees,
        tilt_degrees=cfg.tilt_degrees,
        seed=cfg.seed,
        **dataset_options,
    )
    validation_dataset = DenseSegmentDataset(
        validation_labels, cache_root, mean, std, **dataset_options
    )

    loader_options = dataloader_options(cfg.num_workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=len(train_dataset) > cfg.batch_size,
        collate_fn=collate_chunks,
        generator=torch.Generator().manual_seed(cfg.seed),
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_chunks,
        **loader_options,
    )

    model_kwargs = {
        "in_channels": len(CHANNEL_MODES[cfg.channel_mode]),
        "event_codes": list(EVENT_CODES),
        "channels": cfg.channels,
        "stage_layers": cfg.stage_layers,
        "refinement_stages": cfg.refinement_stages,
        "dropout": cfg.dropout,
    }
    model = build_model(cfg.mode, **model_kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler = make_grad_scaler(policy)

    weights, ratios = compute_event_pos_weight(train_labels, maximum=cfg.max_pos_weight)
    event_weight = torch.tensor(weights, dtype=torch.float32, device=device).view(1, -1, 1)
    body = train_labels["body_target"].to_numpy(np.int64)
    walking_positive = int(np.sum(body == 2))
    walking_negative = int(np.sum(body >= 0) - walking_positive)
    locomotion_weight = torch.tensor(
        float(np.clip(walking_negative / max(walking_positive, 1), 1.0, cfg.max_pos_weight)),
        device=device,
    )
    boundary_weight = torch.tensor(5.0, device=device)

    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    throttle = DutyThrottle()
    history: list[EpochRecord] = []
    best_score = -math.inf
    best_epoch = 0
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        train_result = run_epoch(
            model,
            train_loader,
            device,
            policy,
            optimizer=optimizer,
            scaler=scaler,
            config=cfg,
            event_pos_weight=event_weight,
            locomotion_pos_weight=locomotion_weight,
            boundary_pos_weight=boundary_weight,
            throttle=throttle,
        )
        validation_result = run_epoch(
            model,
            validation_loader,
            device,
            policy,
            config=cfg,
            event_pos_weight=event_weight,
            locomotion_pos_weight=locomotion_weight,
            boundary_pos_weight=boundary_weight,
            collect=True,
        )
        metrics, thresholds = evaluation_metrics(validation_result)
        score = float(metrics["selection_score"])
        scheduler.step(score)
        record = EpochRecord(
            epoch=epoch,
            train_loss=float(train_result["loss"]),
            validation_loss=float(validation_result["loss"]),
            selection_score=score,
            seconds=time.perf_counter() - started,
            extra={
                "posture_macro_f1": metrics["posture"].get("macro_f1"),
                "walking_ap": metrics["walking"].get("average_precision"),
                "learning_rate": optimizer.param_groups[0]["lr"],
            },
        )
        history.append(record)
        print(json.dumps(record.to_dict(), ensure_ascii=False), flush=True)

        if score > best_score + 1e-4:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "model_class": "MultiTaskMSTCN",
                    "model_kwargs": model_kwargs,
                    "mode": cfg.mode,
                    "channel_mode": cfg.channel_mode,
                    "channels": list(CHANNEL_MODES[cfg.channel_mode]),
                    "feature_statistics": {"mean": list(mean), "std": list(std)},
                    "thresholds": thresholds,
                    "epoch": epoch,
                    "best_score": best_score,
                    "fold": {k: v for k, v in fold.items() if not isinstance(v, (list, set))},
                    "config": cfg.to_dict(),
                    "parameter_count": parameter_count(model),
                    "receptive_field_seconds": model.receptive_field_seconds,
                    "event_pos_weight": weights,
                    "event_imbalance_ratio": ratios,
                },
                run_dir / "best.pt",
            )
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    thresholds = checkpoint["thresholds"]

    validation_result = run_epoch(
        model,
        validation_loader,
        device,
        policy,
        config=cfg,
        event_pos_weight=event_weight,
        locomotion_pos_weight=locomotion_weight,
        boundary_pos_weight=boundary_weight,
        collect=True,
    )
    validation_metrics, _ = evaluation_metrics(validation_result, thresholds)
    validation_predictions = prediction_frame(
        validation_dataset.labels, validation_result, validation_dataset.chunks
    )
    validation_predictions.to_csv(
        run_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig"
    )

    test_metrics: dict[str, object] | None = None
    test_event_level: dict[str, object] | None = None
    if fold.get("test_sessions"):
        test_labels = session_subset(labels, fold["test_sessions"])
        if not test_labels.empty:
            test_dataset = DenseSegmentDataset(
                test_labels, cache_root, mean, std, **dataset_options
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                collate_fn=collate_chunks,
                **loader_options,
            )
            test_result = run_epoch(
                model,
                test_loader,
                device,
                policy,
                config=cfg,
                event_pos_weight=event_weight,
                locomotion_pos_weight=locomotion_weight,
                boundary_pos_weight=boundary_weight,
                collect=True,
            )
            test_metrics, _ = evaluation_metrics(test_result, thresholds)
            test_predictions = prediction_frame(
                test_dataset.labels, test_result, test_dataset.chunks
            )
            test_predictions.to_csv(
                run_dir / "test_predictions.csv", index=False, encoding="utf-8-sig"
            )
            test_event_level = {
                code: event_level_metrics(
                    test_predictions, code, float(thresholds[code]), annotations=annotations
                )
                for code in EVENT_CODES
            }

    pd.DataFrame([record.to_dict() for record in history]).to_csv(
        run_dir / "history.csv", index=False, encoding="utf-8-sig"
    )
    report = {
        "status": "complete",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": fold.get("protocol", "grouped_kfold"),
        "fold": fold.get("fold"),
        "test_cows": fold.get("test_cows"),
        "validation_cows": fold.get("validation_cows"),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "config": cfg.to_dict(),
        "parameter_count": checkpoint["parameter_count"],
        "receptive_field_seconds": checkpoint["receptive_field_seconds"],
        "thresholds": thresholds,
        "validation": validation_metrics,
        "test": test_metrics,
        "test_event_level": test_event_level,
        "environment": environment_report(),
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    return report
