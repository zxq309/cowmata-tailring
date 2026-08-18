"""Stable inference API for a cached 50 Hz COWMATA IMU session.

Two defects of the 20260818 entry point are fixed here, and they are the kind
that only show up in production.

**Per-event thresholds survive to deployment.**  ``inference.py`` used a single
``threshold=0.5`` for all events, while the LOCO runs spent considerable effort
choosing one threshold per event on validation data.  Those thresholds were
never written into the joblib bundle, so every one of them was discarded at the
deployment boundary.  A bundle now carries ``thresholds`` and
``feature_version``; :class:`COWMATA` reads them, and a caller-supplied
threshold overrides them only when passed explicitly.

**The feature version is honoured.**  20260819 adds amplitude self-calibration
and two rotation invariants, which changes the *values* of the shared gyroscope
columns.  Scoring the deployed 20260818 model with version-2 features would
shift its inputs without anyone retraining it.  A bundle with no recorded
version is treated as version 1, which is what the deployed model is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cache import SessionCache, open_cache
from .features import estimate_reference, feature_columns, segment_features
from .labels import EVENT_CODES, label_en
from .postprocess import assemble_intervals

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "cowmata_imu"
DEFAULT_MODEL = PROJECT_ROOT / "weights" / "deploy" / "gbdt_full.joblib"
DEFAULT_FEATURE_VERSION = 1


@dataclass(frozen=True)
class PredictionResult:
    """Dense probabilities and merged event candidates for one session."""

    cache_key: str
    cache_dir: Path
    dense: pd.DataFrame
    candidates: pd.DataFrame
    thresholds: dict[str, float]
    feature_version: int
    dense_path: Path | None = None
    candidates_path: Path | None = None


def extract_features(
    cache: SessionCache,
    *,
    causal: bool = False,
    feature_version: int = DEFAULT_FEATURE_VERSION,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Feature table, 2 Hz timestamps and frame indices for one cached session."""

    blocks = [cache.physical(s.start_index, s.stop_index) for s in cache.segments]
    reference = estimate_reference(blocks, causal=causal)

    frames: list[pd.DataFrame] = []
    times: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    for segment, block in zip(cache.segments, blocks):
        local = np.arange(0, segment.length, 25, dtype=np.int64)
        if local.size == 0:
            continue
        quality = cache.quality_flag(segment.start_index, segment.stop_index)
        frames.append(
            segment_features(
                block,
                local,
                causal=causal,
                reference=reference,
                quality=quality,
                feature_version=feature_version,
            )
        )
        times.append(segment.start_ms + local * 20)
        centers.append(local + segment.start_index)
    if not frames:
        raise ValueError(f"cache has no usable IMU segment: {cache.directory}")
    return (
        pd.concat(frames, ignore_index=True),
        np.concatenate(times),
        np.concatenate(centers),
    )


def _candidate_frame(
    dense: pd.DataFrame,
    times: np.ndarray,
    thresholds: dict[str, float],
    *,
    use_boundary: bool,
    postprocess: bool = True,
) -> pd.DataFrame:
    columns = [
        "event_code",
        "label",
        "t_start_rel_ms",
        "t_end_rel_ms",
        "duration_ms",
        "max_prob",
        "threshold",
    ]
    boundary = (
        dense["prob_boundary"].to_numpy(float)
        if use_boundary and "prob_boundary" in dense.columns
        else None
    )
    rows: list[dict[str, Any]] = []
    for code in EVENT_CODES:
        column = f"prob_{code}"
        if column not in dense.columns:
            continue
        scores = dense[column].to_numpy(float)
        threshold = float(thresholds.get(code, 0.5))
        for start, stop in assemble_intervals(
            times,
            scores,
            code,
            threshold=threshold,
            boundary=boundary,
            postprocess=postprocess,
        ):
            inside = (times >= start) & (times < stop)
            rows.append(
                {
                    "event_code": code,
                    "label": label_en(code),
                    "t_start_rel_ms": int(start),
                    "t_end_rel_ms": int(stop),
                    "duration_ms": int(stop - start),
                    "max_prob": round(float(scores[inside].max()) if inside.any() else 0.0, 4),
                    "threshold": threshold,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values("t_start_rel_ms")
        .reset_index(drop=True)
    )


def deep_predict(
    checkpoint_path: Path, cache: SessionCache, *, device: str | None = None
) -> dict[str, np.ndarray]:
    """Run the multi-stage temporal model densely over every segment.

    Each segment is processed in one forward pass at the decision rate, so there
    is no window overlap to reconcile and the result is exactly what the trainer
    saw.  Long segments are chunked with an overlap of one receptive field and
    only the interior of each chunk is kept, which is exact rather than
    approximately equivalent.
    """

    import torch

    from .dataset import CHANNEL_MODES
    from .models import build_model

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint.get("mode", "offline"), **checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    resolved = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(resolved).eval()

    mean = np.asarray(checkpoint["feature_statistics"]["mean"], dtype=np.float32)
    std = np.maximum(np.asarray(checkpoint["feature_statistics"]["std"], dtype=np.float32), 1e-6)
    channels = np.asarray(
        checkpoint.get("channels", CHANNEL_MODES[checkpoint.get("channel_mode", "raw9")]),
        dtype=np.int64,
    )
    stride = int(model.stem_stride)
    overlap_frames = int(model.receptive_field_frames)
    overlap_steps = max(1, overlap_frames // stride)
    max_steps = int(checkpoint.get("inference_chunk_steps", 7200))

    posture_parts: list[np.ndarray] = []
    walking_parts: list[np.ndarray] = []
    event_parts: list[np.ndarray] = []
    boundary_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for segment in cache.segments:
            block = cache.physical(segment.start_index, segment.stop_index)
            total_steps = segment.length // stride
            if total_steps == 0:
                continue
            step_start = 0
            while step_start < total_steps:
                step_stop = min(step_start + max_steps, total_steps)
                pad_left = min(overlap_steps, step_start)
                pad_right = min(overlap_steps, total_steps - step_stop)
                lo = (step_start - pad_left) * stride
                hi = (step_stop + pad_right) * stride
                window = ((block[lo:hi] - mean) / std)[:, channels].T.astype(np.float32)
                tensor = torch.from_numpy(np.ascontiguousarray(window)).unsqueeze(0).to(resolved)
                out = model.predict(tensor)
                keep = slice(pad_left, pad_left + (step_stop - step_start))
                posture_parts.append(out["posture"][0].float().cpu().numpy().T[keep])
                walking_parts.append(out["locomotion"][0].float().cpu().numpy()[keep])
                event_parts.append(out["events"][0].float().cpu().numpy().T[keep])
                boundary_parts.append(out["boundary"][0].float().cpu().numpy()[keep])
                step_start = step_stop
    return {
        "posture": np.concatenate(posture_parts),
        "walking": np.concatenate(walking_parts),
        "events": np.concatenate(event_parts),
        "boundary": np.concatenate(boundary_parts),
        "event_codes": list(model.event_codes),
    }


class COWMATA:
    """Load once and predict many cached sessions, following a model-first API."""

    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        *,
        data_root: str | Path = DEFAULT_DATA_ROOT,
        deep_checkpoint: str | Path | None = None,
    ) -> None:
        candidate = Path(model)
        self.model_path = candidate / "gbdt_full.joblib" if candidate.is_dir() else candidate
        self.data_root = Path(data_root)
        self.deep_checkpoint = Path(deep_checkpoint) if deep_checkpoint is not None else None
        self._bundle: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    def _find_cache(self, cache_key: str) -> Path:
        if Path(cache_key).name != cache_key or any(mark in cache_key for mark in ("/", "\\")):
            raise ValueError("cache_key must be a single cache directory name")
        cache_dir = self.data_root / "supervised_cache" / "session_cache" / cache_key
        if (cache_dir / "signal.i16.npy").exists() or (cache_dir / "features.npy").exists():
            return cache_dir
        raise FileNotFoundError(f"cache_key not found under {self.data_root}: {cache_key}")

    def _load_bundle(self) -> dict[str, Any]:
        if self._bundle is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"GBDT model not found: {self.model_path}")
            import joblib

            self._bundle = joblib.load(self.model_path)
        return self._bundle

    @property
    def feature_version(self) -> int:
        return int(self._load_bundle().get("feature_version", DEFAULT_FEATURE_VERSION))

    @property
    def bundle_thresholds(self) -> dict[str, float]:
        stored = self._load_bundle().get("thresholds") or {}
        return {str(code): float(value) for code, value in stored.items()}

    # ------------------------------------------------------------------
    def predict(
        self,
        source: str,
        *,
        project: str | Path | None = None,
        threshold: float | None = None,
        causal: bool = False,
        postprocess: bool = True,
    ) -> PredictionResult:
        """Predict one cache key and optionally save CSV files under ``project``.

        ``threshold`` is an override.  Left as ``None`` - the recommended
        setting - each event uses the threshold selected for it during training
        and stored in the bundle, falling back to 0.5 only for an event the
        bundle never saw.
        """

        if threshold is not None and not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        cache = open_cache(self._find_cache(source))
        bundle = self._load_bundle()
        version = int(bundle.get("feature_version", DEFAULT_FEATURE_VERSION))
        features, times, centers = extract_features(
            cache, causal=causal, feature_version=version
        )

        model_features = list(bundle["features"])
        missing = sorted(set(model_features) - set(features.columns))
        if missing:
            raise ValueError(
                f"the loaded model requires features this pipeline does not produce at "
                f"feature_version={version}: {missing[:5]}"
                + (" ..." if len(missing) > 5 else "")
            )
        matrix = np.nan_to_num(
            features[model_features].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )

        dense = pd.DataFrame({"center_index": centers, "center_time_ms": times})
        for task, booster in bundle["models"].items():
            dense[f"prob_{task}"] = booster.predict_proba(matrix)
        dense["prob_posture_LYING"] = dense.get("prob_POSTURE_LYING", 0.5)
        dense["prob_posture_UPRIGHT"] = 1.0 - dense["prob_posture_LYING"]
        dense["prob_WALKING"] = dense.get("prob_WALKING", 0.0)
        for code in EVENT_CODES:
            if f"prob_{code}" not in dense:
                dense[f"prob_{code}"] = 0.0

        use_boundary = False
        if self.deep_checkpoint is not None:
            if not self.deep_checkpoint.exists():
                raise FileNotFoundError(f"checkpoint not found: {self.deep_checkpoint}")
            deep = deep_predict(self.deep_checkpoint, cache)
            count = min(len(dense), deep["posture"].shape[0])
            dense = dense.iloc[:count].reset_index(drop=True)
            times = times[:count]
            dense["prob_posture_UPRIGHT"] = deep["posture"][:count, 0]
            dense["prob_posture_LYING"] = deep["posture"][:count, 1]
            dense["prob_WALKING"] = deep["walking"][:count]
            for index, code in enumerate(deep["event_codes"]):
                dense[f"prob_{code}"] = deep["events"][:count, index]
            dense["prob_boundary"] = deep["boundary"][:count]
            use_boundary = True

        thresholds = {code: 0.5 for code in EVENT_CODES}
        thresholds.update(self.bundle_thresholds)
        if threshold is not None:
            thresholds = {code: float(threshold) for code in EVENT_CODES}

        # Identity columns travel with the probabilities.  ``cowmata mine`` and
        # the day-level layer both group by cow and session; a dense file that
        # carries only ``center_time_ms`` forces the caller to rejoin it against
        # the cache metadata, and in 20260818 that join simply was not done, so
        # concatenating two sessions silently interleaved them.
        dense.insert(0, "cache_key", source)
        dense.insert(1, "cow_id", cache.cow_id)
        dense.insert(2, "device_mac", cache.device_mac)
        dense.insert(3, "session_id", cache.session_id)
        output_columns = [
            "cache_key",
            "cow_id",
            "device_mac",
            "session_id",
            "center_index",
            "center_time_ms",
            "prob_posture_UPRIGHT",
            "prob_posture_LYING",
            "prob_WALKING",
            *(f"prob_{code}" for code in EVENT_CODES),
        ]
        if use_boundary:
            output_columns.append("prob_boundary")
        dense = dense[output_columns]
        candidates = _candidate_frame(
            dense, times, thresholds, use_boundary=use_boundary, postprocess=postprocess
        )

        dense_path: Path | None = None
        candidates_path: Path | None = None
        if project is not None:
            output_dir = Path(project)
            output_dir.mkdir(parents=True, exist_ok=True)
            dense_path = output_dir / f"{source}_dense.csv"
            candidates_path = output_dir / f"{source}_candidates.csv"
            dense.to_csv(dense_path, index=False, encoding="utf-8-sig")
            candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
            (output_dir / f"{source}_run.json").write_text(
                json.dumps(
                    {
                        "cache_key": source,
                        "schema_version": cache.schema_version,
                        "feature_version": version,
                        "thresholds": thresholds,
                        "deep_checkpoint": str(self.deep_checkpoint)
                        if self.deep_checkpoint
                        else None,
                        "boundary_snapping": use_boundary,
                        "points": int(len(dense)),
                        "candidates": int(len(candidates)),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        return PredictionResult(
            cache_key=source,
            cache_dir=cache.directory,
            dense=dense,
            candidates=candidates,
            thresholds=thresholds,
            feature_version=version,
            dense_path=dense_path,
            candidates_path=candidates_path,
        )
