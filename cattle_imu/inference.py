"""Stable inference API for a cached 50 Hz COWMATA IMU session."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cattle_imu.annotations import EVENT_CODES
from cattle_imu.features import gravity_split, segment_features, session_reference
from cattle_imu.windowing import context_bounds


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "cowmata_imu"
DEFAULT_MODEL = PROJECT_ROOT / "weights" / "deploy" / "gbdt_full.joblib"

EVENT_LABELS = {
    "STANDING_UP": "standing up",
    "LYING_DOWN": "lying down",
    "URINATION": "urination",
    "DEFECATION": "defecation",
    "TAIL_RAISED": "tail raised",
    "TAIL_WAGGING": "tail wagging",
}


@dataclass(frozen=True)
class PredictionResult:
    """Dense probabilities and merged event candidates for one session."""

    cache_key: str
    cache_dir: Path
    dense: pd.DataFrame
    candidates: pd.DataFrame
    dense_path: Path | None = None
    candidates_path: Path | None = None


def load_segments(cache_dir: Path) -> list[tuple[int, int, int]]:
    """Read valid continuous segments without inventing samples across gaps."""

    meta = cache_dir / "metadata.json"
    if meta.exists():
        with meta.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        segments = [
            (
                int(item["start_index"]),
                int(item["stop_index"]),
                int(item.get("start_ms", int(item["start_index"]) * 20)),
            )
            for item in payload.get("segments", [])
            if int(item["stop_index"]) > int(item["start_index"])
        ]
        if segments:
            return segments
    values = np.load(cache_dir / "features.npy", mmap_mode="r")
    return [(0, len(values), 0)]


def extract_features(
    cache_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Extract the same 104-feature, 2 Hz table used by the deployed GBDT."""

    values = np.load(cache_dir / "features.npy", mmap_mode="r")
    segments = load_segments(cache_dir)
    static_parts: list[np.ndarray] = []
    motion_parts: list[np.ndarray] = []
    for start, stop, _ in segments:
        block = np.asarray(values[start:stop, 0:3], dtype=np.float64)
        if block.shape[0] < 32:
            continue
        static, dynamic = gravity_split(block, causal=False)
        static_parts.append(static[::10])
        motion_parts.append(np.linalg.norm(dynamic, axis=1)[::10])
    reference = None
    if static_parts:
        reference = session_reference(np.concatenate(static_parts), np.concatenate(motion_parts))

    frames: list[pd.DataFrame] = []
    times: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    for start, stop, start_ms in segments:
        block = np.asarray(values[start:stop], dtype=np.float32)
        local = np.arange(0, stop - start, 25, dtype=np.int64)
        if local.size == 0:
            continue
        frames.append(segment_features(block, local, causal=False, reference=reference))
        times.append(start_ms + local * 20)
        centers.append(local + start)
    if not frames:
        raise ValueError(f"cache has no usable IMU segment: {cache_dir}")
    return pd.concat(frames, ignore_index=True), np.concatenate(times), np.concatenate(centers)


def deep_predict(
    checkpoint_path: Path,
    cache_dir: Path,
    centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the optional offline TCN posture and walking heads."""

    import torch

    from cattle_imu.model import build_model

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(mode="offline", **checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    mean = np.asarray(checkpoint["feature_statistics"]["mean"], dtype=np.float32)
    std = np.maximum(
        np.asarray(checkpoint["feature_statistics"]["std"], dtype=np.float32), 1e-6
    )
    context = int(checkpoint["context_samples"])
    indices = np.asarray(checkpoint["feature_indices"], dtype=np.int64)
    values = np.load(cache_dir / "features.npy", mmap_mode="r")
    segments = load_segments(cache_dir)

    posture_parts: list[np.ndarray] = []
    walking_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for offset in range(0, len(centers), 64):
            batch = centers[offset : offset + 64]
            windows = np.zeros((len(batch), len(indices), context), dtype=np.float32)
            for row, center in enumerate(batch):
                segment = next(
                    (item for item in segments if item[0] <= center < item[1]),
                    (0, len(values), 0),
                )
                start, stop, _ = segment
                source_start, source_stop, destination = context_bounds(
                    int(center), int(start), int(stop), context, "centered"
                )
                normalized = (
                    np.asarray(values[source_start:source_stop], dtype=np.float32) - mean
                ) / std
                selected = normalized[:, indices].T
                windows[row, :, destination : destination + selected.shape[1]] = selected
            output = model(torch.from_numpy(windows).to(device))
            posture_parts.append(torch.softmax(output["posture_logits"], 1).cpu().numpy())
            walking_parts.append(
                torch.sigmoid(output["locomotion_logits"]).cpu().numpy().reshape(-1)
            )
    return np.concatenate(posture_parts), np.concatenate(walking_parts)


def _candidate_frame(
    result: pd.DataFrame,
    times: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    columns = [
        "event_code",
        "label",
        "t_start_rel_ms",
        "t_end_rel_ms",
        "max_prob",
    ]
    rows: list[dict[str, object]] = []
    for code in EVENT_CODES:
        column = f"prob_{code}"
        selected = result[column].to_numpy(float) >= threshold
        if not selected.any():
            continue
        indices = np.flatnonzero(selected)
        start = previous = int(indices[0])
        for current_value in indices[1:]:
            current = int(current_value)
            if times[current] - times[previous] > 5000:
                rows.append(
                    {
                        "event_code": code,
                        "label": EVENT_LABELS[code],
                        "t_start_rel_ms": int(times[start]),
                        "t_end_rel_ms": int(times[previous] + 500),
                        "max_prob": round(float(result[column].iloc[start : previous + 1].max()), 4),
                    }
                )
                start = current
            previous = current
        rows.append(
            {
                "event_code": code,
                "label": EVENT_LABELS[code],
                "t_start_rel_ms": int(times[start]),
                "t_end_rel_ms": int(times[previous] + 500),
                "max_prob": round(float(result[column].iloc[start : previous + 1].max()), 4),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("t_start_rel_ms").reset_index(drop=True)


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
        self._bundle: dict[str, object] | None = None

    def _find_cache(self, cache_key: str) -> Path:
        if Path(cache_key).name != cache_key or any(mark in cache_key for mark in ("/", "\\")):
            raise ValueError("cache_key must be a single cache directory name")
        for cache_root in (
            self.data_root / "supervised_cache" / "session_cache",
        ):
            cache_dir = cache_root / cache_key
            if (cache_dir / "features.npy").exists():
                return cache_dir
        raise FileNotFoundError(f"cache_key not found under {self.data_root}: {cache_key}")

    def _load_bundle(self) -> dict[str, object]:
        if self._bundle is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"GBDT model not found: {self.model_path}")
            import joblib

            self._bundle = joblib.load(self.model_path)
        return self._bundle

    def predict(
        self,
        source: str,
        *,
        project: str | Path | None = None,
        threshold: float = 0.5,
    ) -> PredictionResult:
        """Predict one cache key and optionally save CSV files under ``project``."""

        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        cache_dir = self._find_cache(source)
        features, times, centers = extract_features(cache_dir)
        bundle = self._load_bundle()
        model_features = list(bundle["features"])
        missing = sorted(set(model_features) - set(features.columns))
        if missing:
            raise ValueError(f"deployed model requires missing features: {missing}")
        matrix = features[model_features].to_numpy(np.float32)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        dense = pd.DataFrame({"center_index": centers, "center_time_ms": times})
        for task, booster in bundle["models"].items():
            dense[f"prob_{task}"] = booster.predict_proba(matrix)
        dense["prob_posture_LYING"] = dense.get("prob_POSTURE_LYING", 0.5)
        dense["prob_posture_UPRIGHT"] = 1.0 - dense["prob_posture_LYING"]
        dense["prob_WALKING"] = dense.get("prob_WALKING", 0.0)
        for code in EVENT_CODES:
            if f"prob_{code}" not in dense:
                dense[f"prob_{code}"] = 0.0

        if self.deep_checkpoint is not None:
            if not self.deep_checkpoint.exists():
                raise FileNotFoundError(f"TCN checkpoint not found: {self.deep_checkpoint}")
            posture, walking = deep_predict(self.deep_checkpoint, cache_dir, centers)
            dense["prob_posture_UPRIGHT"] = posture[:, 0]
            dense["prob_posture_LYING"] = posture[:, 1]
            dense["prob_WALKING"] = walking

        output_columns = [
            "center_index",
            "center_time_ms",
            "prob_posture_UPRIGHT",
            "prob_posture_LYING",
            "prob_WALKING",
            *(f"prob_{code}" for code in EVENT_CODES),
        ]
        dense = dense[output_columns]
        candidates = _candidate_frame(dense, times, threshold)
        dense_path: Path | None = None
        candidates_path: Path | None = None
        if project is not None:
            output_dir = Path(project)
            output_dir.mkdir(parents=True, exist_ok=True)
            dense_path = output_dir / f"{source}_dense.csv"
            candidates_path = output_dir / f"{source}_candidates.csv"
            dense.to_csv(dense_path, index=False, encoding="utf-8-sig")
            candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
        return PredictionResult(
            cache_key=source,
            cache_dir=cache_dir,
            dense=dense,
            candidates=candidates,
            dense_path=dense_path,
            candidates_path=candidates_path,
        )
