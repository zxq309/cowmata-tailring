"""Dense training data: whole stretches of stream, not one window per label.

The arithmetic that motivates this module
-----------------------------------------
The 20260818 loader returned one 2048-sample window per label point, at stride
25.  One epoch therefore encoded

    344,287 samples x 2,048 = 7.05e8 frames

while the supervised data itself is only

    131.6 h x 3,600 x 50 = 2.37e7 frames

i.e. every raw sample was encoded about 30 times over, and about 82 times over
per unit of covered signal.  Feeding a contiguous chunk once and supervising all
of its label points costs one pass over the data, and it is also the only way a
model can learn that a urination is one object rather than 60 unrelated points.

Three further things happen here and nowhere else.

**Segment safety.**  A chunk never spans two contiguous segments.  Cached
segments are stored end to end, so a chunk crossing a boundary would splice
recordings that may be hours apart.

**Mounting augmentation.**  ``max_rotation_degrees`` defaulted to 20 in the old
code.  The rings are fastened by hand and are *not* consistently oriented, so a
20 degree cone does not cover the actual nuisance distribution.  The default
here is a full +-180 degrees about the gravity axis plus a +-35 degree tilt,
which is the same reasoning UniMTS uses to obtain orientation-invariance.

**Boundary targets.**  The ASRF-style boundary head needs a target: 1 within a
tolerance of any true event start or stop.  It is built here from the event
target matrix so it can never disagree with the labels it is derived from.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .cache import SessionCache, open_cache
from .labels import EVENT_CODES, STATE_TO_POSTURE
from .preprocessing import TARGET_STRIDE_SAMPLES

#: Boundary target half-width in decision steps (2 Hz), i.e. +-1.5 s.
BOUNDARY_TOLERANCE_STEPS = 3

#: Channel layouts offered to the stem.  ``raw9`` is the honest default now that
#: the cache no longer stores redundant magnitudes; ``accgyro6`` reproduces the
#: 20260818 decision to drop the magnetometer in a metal-rich barn.
CHANNEL_MODES: dict[str, tuple[int, ...]] = {
    "raw9": (0, 1, 2, 3, 4, 5, 6, 7, 8),
    "accgyro6": (0, 1, 2, 3, 4, 5),
}


class CacheStore:
    """Bounded LRU of open memory-mapped session caches."""

    def __init__(self, cache_root: str | Path, max_open: int = 64) -> None:
        self.cache_root = Path(cache_root)
        self.max_open = int(max_open)
        self._open: OrderedDict[str, SessionCache] = OrderedDict()

    def get(self, cache_key: str) -> SessionCache:
        if cache_key in self._open:
            value = self._open.pop(cache_key)
            self._open[cache_key] = value
            return value
        value = open_cache(self.cache_root / cache_key)
        self._open[cache_key] = value
        while len(self._open) > self.max_open:
            self._open.popitem(last=False)
        return value


def hierarchical_targets(body_target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map annotated body states to posture and walking without losing FEEDING."""

    body = np.asarray(body_target, dtype=np.int64)
    valid = body >= 0
    posture = np.full(body.shape, -1, dtype=np.int64)
    if np.any(valid):
        lookup = np.asarray(
            [STATE_TO_POSTURE[index] for index in range(len(STATE_TO_POSTURE))], dtype=np.int64
        )
        posture[valid] = lookup[body[valid]]
    walking = (body == 2).astype(np.float32)
    return posture, walking, valid.astype(np.float32)


def boundary_targets(
    event_target: np.ndarray, event_mask: np.ndarray, *, tolerance: int = BOUNDARY_TOLERANCE_STEPS
) -> tuple[np.ndarray, np.ndarray]:
    """Build the ASRF boundary target from the event target matrix.

    A boundary step is one within ``tolerance`` steps of any event turning on or
    off.  The mask is the union of the per-event masks: a step where no event is
    supervised cannot assert the absence of a boundary either.
    """

    events = np.asarray(event_target)
    masks = np.asarray(event_mask)
    if events.ndim != 2:
        raise ValueError("event_target must be (steps, events)")
    steps = events.shape[0]
    boundary = np.zeros(steps, dtype=np.float32)
    if steps > 1:
        change = np.any(events[1:] != events[:-1], axis=1)
        for position in np.flatnonzero(change):
            lo = max(0, position - tolerance + 1)
            hi = min(steps, position + tolerance + 1)
            boundary[lo:hi] = 1.0
    mask = (masks.max(axis=1) > 0).astype(np.float32)
    return boundary, mask


def random_rotation(
    rng: np.random.Generator, *, yaw_degrees: float = 180.0, tilt_degrees: float = 35.0
) -> np.ndarray:
    """Random mounting rotation: free about gravity, bounded in tilt.

    The tail ring can be fastened at any angle about the tail axis, so the yaw
    component is sampled over the full circle.  The tilt is bounded because the
    ring does not usually end up upside down, and letting it do so would teach
    the model that lying and standing are the same thing.
    """

    yaw = np.deg2rad(rng.uniform(-yaw_degrees, yaw_degrees))
    tilt = np.deg2rad(rng.uniform(-tilt_degrees, tilt_degrees))
    roll = np.deg2rad(rng.uniform(-tilt_degrees, tilt_degrees))
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(tilt), np.sin(tilt)
    cx, sx = np.cos(roll), np.sin(roll)
    rz = np.asarray([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    ry = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    return (rz @ ry @ rx).astype(np.float32)


class DenseSegmentDataset(Dataset):
    """One item is a contiguous chunk of one session with all of its labels.

    ``chunk_steps`` is measured in 2 Hz decision steps; the raw chunk is
    ``chunk_steps * stride`` samples.  Chunks are laid out with ``chunk_overlap``
    steps of overlap so that a label near a chunk edge still receives context
    from both sides during training.
    """

    def __init__(
        self,
        labels: pd.DataFrame,
        cache_root: str | Path,
        mean: np.ndarray | list[float],
        std: np.ndarray | list[float],
        *,
        chunk_steps: int = 1200,
        chunk_overlap: int = 120,
        stride: int = TARGET_STRIDE_SAMPLES,
        channel_mode: str = "raw9",
        augment: bool = False,
        yaw_degrees: float = 180.0,
        tilt_degrees: float = 35.0,
        require_supervision: bool = True,
        seed: int = 20260819,
    ) -> None:
        if channel_mode not in CHANNEL_MODES:
            raise ValueError(f"unknown channel_mode: {channel_mode!r}")
        if chunk_steps <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_steps:
            raise ValueError("need 0 <= chunk_overlap < chunk_steps and chunk_steps > 0")
        self.channels = np.asarray(CHANNEL_MODES[channel_mode], dtype=np.int64)
        self.channel_mode = channel_mode
        self.chunk_steps = int(chunk_steps)
        self.chunk_overlap = int(chunk_overlap)
        self.stride = int(stride)
        self.augment = bool(augment)
        self.yaw_degrees = float(yaw_degrees)
        self.tilt_degrees = float(tilt_degrees)
        self.seed = int(seed)
        self.store = CacheStore(cache_root)
        self._rng: np.random.Generator | None = None

        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        self.std = np.maximum(np.asarray(std, dtype=np.float32).reshape(-1), np.float32(1e-6))
        if self.mean.size != 9 or self.std.size != 9:
            raise ValueError("normalisation statistics must cover the 9 physical channels")

        required = {"cache_key", "center_index", "segment_start_index", "segment_stop_index"}
        missing = required - set(labels.columns)
        if missing:
            raise ValueError(f"label frame is missing columns: {sorted(missing)}")

        frame = labels.sort_values(["cache_key", "center_index"], kind="stable").reset_index(
            drop=True
        )
        self.labels = frame
        body = frame["body_target"].to_numpy(np.int64)
        self.posture, self.locomotion, self.body_mask = hierarchical_targets(body)
        self.event_target = frame[[f"event_{c}" for c in EVENT_CODES]].to_numpy(np.float32)
        self.event_mask = frame[[f"mask_{c}" for c in EVENT_CODES]].to_numpy(np.float32)
        self.center_index = frame["center_index"].to_numpy(np.int64)
        self.cache_keys = frame["cache_key"].astype(str).to_numpy()
        self.segment_start = frame["segment_start_index"].to_numpy(np.int64)
        self.segment_stop = frame["segment_stop_index"].to_numpy(np.int64)

        self.chunks = self._build_chunks(require_supervision)
        if not self.chunks:
            raise ValueError("no usable chunk was produced; check the label frame")

    # ------------------------------------------------------------------
    def _build_chunks(self, require_supervision: bool) -> list[tuple[int, int]]:
        """Return ``(row_start, row_stop)`` pairs of the sorted label frame."""

        chunks: list[tuple[int, int]] = []
        keys = self.cache_keys
        segment_id = self.segment_start
        step = self.chunk_steps - self.chunk_overlap
        boundaries = np.flatnonzero((keys[1:] != keys[:-1]) | (segment_id[1:] != segment_id[:-1]))
        starts = np.concatenate(([0], boundaries + 1))
        stops = np.concatenate((boundaries + 1, [len(keys)]))
        supervised = (self.body_mask > 0) | (self.event_mask.max(axis=1) > 0)
        for run_start, run_stop in zip(starts, stops):
            position = int(run_start)
            while position < run_stop:
                stop = min(position + self.chunk_steps, int(run_stop))
                if stop - position >= 2 and (
                    not require_supervision or bool(supervised[position:stop].any())
                ):
                    chunks.append((position, stop))
                if stop >= run_stop:
                    break
                position += step
        return chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def _worker_rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng(self.seed ^ int(torch.initial_seed()))
        return self._rng

    # ------------------------------------------------------------------
    def _signal(self, row_start: int, row_stop: int) -> np.ndarray:
        cache = self.store.get(str(self.cache_keys[row_start]))
        first_center = int(self.center_index[row_start])
        last_center = int(self.center_index[row_stop - 1])
        segment_stop = int(self.segment_stop[row_start])
        start = first_center
        stop = min(last_center + self.stride, segment_stop, cache.n_frames)
        block = cache.physical(start, stop)
        expected = (row_stop - row_start) * self.stride
        if block.shape[0] < expected:
            pad = np.zeros((expected - block.shape[0], block.shape[1]), dtype=np.float32)
            block = np.concatenate((block, pad), axis=0)
        return block[:expected]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row_start, row_stop = self.chunks[index]
        signal = self._signal(row_start, row_stop).copy()

        if self.augment:
            rng = self._worker_rng()
            rotation = random_rotation(
                rng, yaw_degrees=self.yaw_degrees, tilt_degrees=self.tilt_degrees
            )
            signal[:, 0:3] = signal[:, 0:3] @ rotation.T
            signal[:, 3:6] = signal[:, 3:6] @ rotation.T
            signal[:, 6:9] = signal[:, 6:9] @ rotation.T
            signal *= np.float32(rng.uniform(0.97, 1.03))
            noise = np.asarray([0.004] * 3 + [0.20] * 3 + [0.002] * 3, dtype=np.float32)
            signal += rng.normal(0.0, noise, size=signal.shape).astype(np.float32)

        normalised = (signal - self.mean) / self.std
        inputs = normalised[:, self.channels].T.astype(np.float32)

        event_target = self.event_target[row_start:row_stop]
        event_mask = self.event_mask[row_start:row_stop]
        boundary, boundary_mask = boundary_targets(event_target, event_mask)
        return {
            "inputs": torch.from_numpy(np.ascontiguousarray(inputs)),
            "posture_target": torch.from_numpy(self.posture[row_start:row_stop].copy()),
            "posture_mask": torch.from_numpy(self.body_mask[row_start:row_stop].copy()),
            "locomotion_target": torch.from_numpy(self.locomotion[row_start:row_stop].copy()),
            "locomotion_mask": torch.from_numpy(self.body_mask[row_start:row_stop].copy()),
            "event_target": torch.from_numpy(event_target.T.copy()),
            "event_mask": torch.from_numpy(event_mask.T.copy()),
            "boundary_target": torch.from_numpy(boundary),
            "boundary_mask": torch.from_numpy(boundary_mask),
            "valid": torch.ones(row_stop - row_start, dtype=torch.float32),
            "row_start": torch.tensor(row_start, dtype=torch.long),
            "row_stop": torch.tensor(row_stop, dtype=torch.long),
        }


def collate_chunks(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Right-pad unequal-length chunks and carry a validity mask.

    Padding is masked out of every loss and zeroed inside the network after each
    layer, so a short chunk at the end of a segment cannot contaminate the chunk
    it shares a batch with.
    """

    lengths = [int(item["valid"].shape[0]) for item in batch]
    longest = max(lengths)
    stride_ratio = batch[0]["inputs"].shape[-1] // lengths[0]
    out: dict[str, torch.Tensor] = {}

    channels = batch[0]["inputs"].shape[0]
    inputs = torch.zeros(len(batch), channels, longest * stride_ratio, dtype=torch.float32)
    for index, item in enumerate(batch):
        width = item["inputs"].shape[-1]
        inputs[index, :, :width] = item["inputs"]
    out["inputs"] = inputs

    n_events = batch[0]["event_target"].shape[0]
    for key, dtype, fill in (
        ("posture_target", torch.long, -1),
        ("posture_mask", torch.float32, 0),
        ("locomotion_target", torch.float32, 0),
        ("locomotion_mask", torch.float32, 0),
        ("boundary_target", torch.float32, 0),
        ("boundary_mask", torch.float32, 0),
        ("valid", torch.float32, 0),
    ):
        padded = torch.full((len(batch), longest), fill, dtype=dtype)
        for index, item in enumerate(batch):
            padded[index, : lengths[index]] = item[key].to(dtype)
        out[key] = padded

    for key in ("event_target", "event_mask"):
        padded = torch.zeros(len(batch), n_events, longest, dtype=torch.float32)
        for index, item in enumerate(batch):
            padded[index, :, : lengths[index]] = item[key]
        out[key] = padded

    out["row_start"] = torch.stack([item["row_start"] for item in batch])
    out["row_stop"] = torch.stack([item["row_stop"] for item in batch])
    out["lengths"] = torch.tensor(lengths, dtype=torch.long)
    return out


def normalisation_statistics(
    cache_root: str | Path, cache_keys: list[str], *, max_frames_per_session: int = 200_000
) -> tuple[list[float], list[float], int]:
    """Channel mean and standard deviation over the training sessions only.

    Reads at most ``max_frames_per_session`` frames from each session; at the
    planned data scale a full pass over every training cache just to compute two
    vectors of nine numbers would be hours of pure I/O for four significant
    digits.
    """

    total = np.zeros(9, dtype=np.float64)
    square = np.zeros(9, dtype=np.float64)
    count = 0
    store = CacheStore(cache_root, max_open=4)
    for key in sorted(set(cache_keys)):
        cache = store.get(str(key))
        limit = min(cache.n_frames, int(max_frames_per_session))
        for start in range(0, limit, 100_000):
            chunk = cache.physical(start, min(start + 100_000, limit)).astype(np.float64)
            total += chunk.sum(axis=0)
            square += np.square(chunk).sum(axis=0)
            count += chunk.shape[0]
    mean = total / max(count, 1)
    std = np.sqrt(np.maximum(square / max(count, 1) - np.square(mean), 1e-8))
    return mean.tolist(), std.tolist(), int(count)


def session_subset(frame: pd.DataFrame, keys: list[str] | set[str]) -> pd.DataFrame:
    key_set = set(keys)
    session_keys = frame["device_mac"].astype(str) + "|" + frame["session_id"].astype(str)
    return frame[session_keys.isin(key_set)].reset_index(drop=True)


def event_pos_weight(
    frame: pd.DataFrame, *, maximum: float = 20.0
) -> tuple[list[float], list[float]]:
    """Per-event positive weights and their unclipped imbalance ratios."""

    weights: list[float] = []
    ratios: list[float] = []
    for code in EVENT_CODES:
        mask = frame[f"mask_{code}"].to_numpy(bool)
        target = frame[f"event_{code}"].to_numpy(np.uint8)
        positive = int(target[mask].sum())
        negative = int(mask.sum() - positive)
        ratio = float(negative / positive) if positive else float("inf")
        ratios.append(ratio)
        weights.append(float(np.clip(ratio, 1.0, maximum)) if positive else 1.0)
    return weights, ratios
