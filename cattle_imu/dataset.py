"""Memory-mapped causal context dataset with hierarchical behavior targets."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .annotations import EVENT_CODES
from .windowing import context_bounds


# Cache channel layout: 0-8 acc/gyro/mag, 9-11 the three magnitudes, 12 the
# timing-quality flag.
ACC_SLICE = slice(0, 3)
GYRO_SLICE = slice(3, 6)
MAG_SLICE = slice(6, 9)
MAGNITUDE_COLUMNS = (9, 10, 11)


def recompute_magnitudes(source: np.ndarray) -> None:
    """Rewrite channels 9/10/11 in place so they match channels 0-8.

    Augmentation scales and adds noise to the nine raw axes only.  Without this
    step the magnitude channels contradict the axes they are supposed to
    summarise and the model receives physically impossible input.
    """

    if source.shape[1] <= max(MAGNITUDE_COLUMNS):
        return
    source[:, 9] = np.linalg.norm(source[:, ACC_SLICE], axis=1)
    source[:, 10] = np.linalg.norm(source[:, GYRO_SLICE], axis=1)
    source[:, 11] = np.linalg.norm(source[:, MAG_SLICE], axis=1)


# The timing-quality flag at column 12 is metadata, never a model input.  Old
# mode names remain aliases so historical commands fail safe instead of silently
# changing dimensions to include the flag again.
FEATURE_MODES = {
    "raw12": np.arange(12, dtype=np.int64),
    "accgyro8": np.asarray([0, 1, 2, 3, 4, 5, 9, 10], dtype=np.int64),
    "magnitude2": np.asarray([9, 10], dtype=np.int64),
    "raw13": np.arange(12, dtype=np.int64),
    "accgyro9": np.asarray([0, 1, 2, 3, 4, 5, 9, 10], dtype=np.int64),
    "magnitude3": np.asarray([9, 10], dtype=np.int64),
}


def _random_rotation(rng: np.random.Generator, max_degrees: float) -> np.ndarray:
    angles = np.deg2rad(rng.uniform(-max_degrees, max_degrees, size=3))
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    rx = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz = np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx


class SessionArrayStore:
    def __init__(self, cache_root: str | Path, max_open: int = 128) -> None:
        self.cache_root = Path(cache_root)
        self.max_open = max_open
        self._arrays: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, key: str) -> np.ndarray:
        if key in self._arrays:
            value = self._arrays.pop(key)
            self._arrays[key] = value
            return value
        value = np.load(self.cache_root / key / "features.npy", mmap_mode="r")
        self._arrays[key] = value
        while len(self._arrays) > self.max_open:
            self._arrays.popitem(last=False)
        return value


def hierarchical_targets(body_target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map historical body labels to posture and walking without losing FEEDING rows."""

    body = np.asarray(body_target, dtype=np.int64)
    valid = body >= 0
    posture = np.full(body.shape, -1, dtype=np.int64)
    posture[valid] = (body[valid] == 1).astype(np.int64)  # LYING=1; all others upright.
    walking = (body == 2).astype(np.float32)
    return posture, walking, valid.astype(np.float32)


class WindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Return a trailing causal context ending at each timestamp mother label."""

    def __init__(
        self,
        samples: pd.DataFrame,
        cache_root: str | Path,
        mean: list[float] | np.ndarray,
        std: list[float] | np.ndarray,
        *,
        augment: bool = False,
        feature_mode: str = "accgyro8",
        max_rotation_degrees: float = 0.0,
        context_samples: int = 2048,
        seed: int = 20260814,
        window_mode: str = "causal",
        require_segment_bounds: bool = True,
    ) -> None:
        if context_samples <= 0:
            raise ValueError("context_samples must be positive")
        if window_mode not in {"causal", "centered"}:
            raise ValueError(f"unknown window_mode: {window_mode!r}")
        self.window_mode = str(window_mode)
        self.cache_keys = samples["cache_key"].astype(str).to_numpy()
        self.center_indices = samples["center_index"].to_numpy(np.int64)
        has_bounds = {"segment_start_index", "segment_stop_index"} <= set(samples.columns)
        if not has_bounds and require_segment_bounds:
            raise ValueError(
                "samples.csv lacks segment_start_index/segment_stop_index. Rebuild the "
                "supervised cache with the patched build_supervised_cache.py, or pass "
                "require_segment_bounds=False to reproduce the old (leaky) behaviour."
            )
        if has_bounds:
            self.segment_starts = samples["segment_start_index"].to_numpy(np.int64)
            self.segment_stops = samples["segment_stop_index"].to_numpy(np.int64)
        else:  # legacy cache: whole session treated as one segment
            self.segment_starts = np.zeros(len(samples), dtype=np.int64)
            self.segment_stops = np.full(len(samples), np.iinfo(np.int64).max, dtype=np.int64)
        body = samples["body_target"].to_numpy(np.int64)
        self.posture_targets, self.locomotion_targets, self.body_masks = hierarchical_targets(body)
        self.event_targets = samples[[f"event_{code}" for code in EVENT_CODES]].to_numpy(np.float32)
        self.event_masks = samples[[f"mask_{code}" for code in EVENT_CODES]].to_numpy(np.float32)
        self.sample_ids = samples["sample_id"].astype(str).to_numpy()
        self.center_times = samples["center_time_ms"].to_numpy(np.int64)
        self.store = SessionArrayStore(cache_root)
        if feature_mode not in FEATURE_MODES:
            raise ValueError(f"unknown feature_mode: {feature_mode}")
        self.feature_mode = feature_mode
        self.feature_indices = FEATURE_MODES[feature_mode]
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if self.mean.shape != (13,) or self.std.shape != (13,):
            raise ValueError("normalization mean/std must each contain 13 cache channels")
        self.std = np.maximum(self.std, np.float32(1e-6))
        self.augment = bool(augment)
        self.max_rotation_degrees = float(max_rotation_degrees)
        self.context_samples = int(context_samples)
        self.seed = int(seed)
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.center_indices)

    def _worker_rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng(self.seed ^ int(torch.initial_seed()))
        return self._rng

    def _context(self, raw: np.ndarray, index: int) -> tuple[np.ndarray, int]:
        center = int(self.center_indices[index])
        if center < 0 or center >= len(raw):
            raise IndexError(f"center_index {center} is outside session with {len(raw)} frames")
        segment_start = max(0, int(self.segment_starts[index]))
        segment_stop = min(len(raw), int(self.segment_stops[index]))
        start, stop, dest_start = context_bounds(
            center, segment_start, segment_stop, self.context_samples, self.window_mode
        )
        source = np.asarray(raw[start:stop], dtype=np.float32).copy()
        if self.augment:
            rng = self._worker_rng()
            if self.max_rotation_degrees > 0:
                rotation = _random_rotation(rng, self.max_rotation_degrees)
                source[:, 0:3] = source[:, 0:3] @ rotation.T
                source[:, 3:6] = source[:, 3:6] @ rotation.T
                source[:, 6:9] = source[:, 6:9] @ rotation.T
            source[:, :9] *= np.float32(rng.uniform(0.98, 1.02))
            noise = np.asarray([0.004] * 3 + [0.20] * 3 + [0.002] * 3, dtype=np.float32)
            source[:, :9] += rng.normal(0.0, noise, size=source[:, :9].shape).astype(np.float32)
            # Magnitudes are model inputs in every accgyro* mode, so they must
            # be rebuilt from the augmented axes.
            recompute_magnitudes(source)
        normalized = (source - self.mean) / self.std
        selected = normalized[:, self.feature_indices]
        output = np.zeros((self.context_samples, len(self.feature_indices)), dtype=np.float32)
        output[dest_start : dest_start + len(selected)] = selected
        return output, len(selected)

    # Backwards-compatible alias; the window is no longer necessarily causal.
    def _causal_context(self, raw: np.ndarray, center: int) -> tuple[np.ndarray, int]:
        matches = np.flatnonzero(self.center_indices == int(center))
        if matches.size == 0:
            raise IndexError(f"center_index {center} is not part of this dataset")
        return self._context(raw, int(matches[0]))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window, valid_length = self._context(self.store.get(self.cache_keys[index]), index)
        return {
            "inputs": torch.from_numpy(window.T.copy()),
            "posture_target": torch.tensor(self.posture_targets[index], dtype=torch.long),
            "posture_mask": torch.tensor(self.body_masks[index], dtype=torch.float32),
            "locomotion_target": torch.tensor(self.locomotion_targets[index], dtype=torch.float32),
            "locomotion_mask": torch.tensor(self.body_masks[index], dtype=torch.float32),
            "event_target": torch.from_numpy(self.event_targets[index].copy()),
            "event_mask": torch.from_numpy(self.event_masks[index].copy()),
            "valid_length": torch.tensor(valid_length, dtype=torch.long),
            "row_index": torch.tensor(index, dtype=torch.long),
        }


class EventWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Expose selected event targets for an optional specialist model."""

    def __init__(
        self,
        samples: pd.DataFrame,
        cache_root: str | Path,
        mean: list[float] | np.ndarray,
        std: list[float] | np.ndarray,
        event_codes: tuple[str, ...] | list[str],
        **window_options: object,
    ) -> None:
        codes = tuple(event_codes)
        if not codes or len(codes) != len(set(codes)):
            raise ValueError("event_codes must be non-empty and unique")
        unknown = sorted(set(codes) - set(EVENT_CODES))
        if unknown:
            raise ValueError(f"unknown event codes: {unknown}")
        self.event_codes = codes
        self.event_indices = torch.tensor([EVENT_CODES.index(code) for code in codes], dtype=torch.long)
        self.base = WindowDataset(samples, cache_root, mean, std, **window_options)
        self.feature_indices = self.base.feature_indices

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        batch = self.base[index]
        return {
            "inputs": batch["inputs"],
            "event_target": batch["event_target"].index_select(0, self.event_indices),
            "event_mask": batch["event_mask"].index_select(0, self.event_indices),
            "valid_length": batch["valid_length"],
            "row_index": batch["row_index"],
        }


def filter_by_valid_context(
    samples: pd.DataFrame,
    context_samples: int,
    *,
    min_valid_seconds: float = 8.0,
    sample_rate_hz: float = 50.0,
    window_mode: str = "causal",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Drop samples whose window is mostly zero padding.

    At a segment boundary a window can be almost entirely padding, yet the old
    loss weighted such a sample exactly like one with 40 s of real history.
    Returns the filtered frame and a summary for ``report.json``.
    """

    from .windowing import context_bounds_batch

    if {"segment_start_index", "segment_stop_index"} - set(samples.columns):
        return samples.reset_index(drop=True), {
            "applied": False,
            "reason": "samples lack segment bounds; rebuild the supervised cache",
        }
    start, stop, _ = context_bounds_batch(
        samples["center_index"].to_numpy(np.int64),
        samples["segment_start_index"].to_numpy(np.int64),
        samples["segment_stop_index"].to_numpy(np.int64),
        int(context_samples),
        window_mode,
    )
    valid_samples = stop - start
    minimum = int(round(float(min_valid_seconds) * float(sample_rate_hz)))
    keep = valid_samples >= minimum
    filtered = samples.loc[keep].reset_index(drop=True)
    return filtered, {
        "applied": True,
        "window_mode": window_mode,
        "min_valid_seconds": float(min_valid_seconds),
        "min_valid_samples": minimum,
        "rows_before": int(len(samples)),
        "rows_after": int(len(filtered)),
        "rows_dropped": int(len(samples) - len(filtered)),
        "median_valid_samples": float(np.median(valid_samples)) if len(samples) else None,
    }


def session_subset(samples: pd.DataFrame, keys: list[str] | set[str]) -> pd.DataFrame:
    key_set = set(keys)
    session_keys = samples["device_mac"].astype(str) + "|" + samples["session_id"].astype(str)
    return samples[session_keys.isin(key_set)].reset_index(drop=True)
