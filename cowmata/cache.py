"""Session cache: on-disk format for resampled 50 Hz sessions.

Why the format changed in 20260819
----------------------------------
The 20260818 cache stored ``features.npy`` as ``(N, 13) float32``:

===========================  =====================  ============
channel                      content                bytes/frame
===========================  =====================  ============
0-8                          acc / gyro / mag       36
9-11                         the three magnitudes   12
12                           timing-quality flag    4
===========================  =====================  ============

That is 52 bytes per 50 Hz frame.  At the 20260818 scale (131.6 h) it cost
1.23 GB and nobody cared.  At the planned scale it does not survive::

    200 cows x 7 days = 1,400 cow-days = 33,600 h = 6.05e9 frames
    6.05e9 x 52 B ~= 315 GB

Three changes, all strictly lossless, bring that to ~109 GB:

1. **Store int16 counts, not float32 physical values.**  The device emits
   ``<i2`` (see :data:`cowmata.io.V2_FRAME_DTYPE`); widening to float32 was
   pure inflation.  Calibration divisors and biases live in ``meta.json`` and
   are applied on read, which also means a calibration correction no longer
   requires rebuilding terabytes of cache.
2. **Drop channels 9-11.**  They are ``norm(acc)``, ``norm(gyro)``,
   ``norm(mag)``, recomputable from 0-8 in microseconds.
3. **Store the quality flag as sparse intervals.**  It is zero almost
   everywhere; one float32 per frame for a mostly-zero flag is 4 bytes/frame of
   nothing.

Timestamps are no longer stored either.  Within a segment the grid is exactly
``start_ms + 20 * local_index`` by construction (see
:func:`cowmata.preprocessing.resample_session`), so an 8 byte/frame int64 array
was storing a closed-form expression.

Backward compatibility
----------------------
:func:`open_cache` auto-detects the layout.  A 20260818 directory containing
``features.npy`` is read as schema 1 and served through the identical API, so
the packaged demo session and any existing local cache keep working untouched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .io import rotate_mag_raw_to_imu

SCHEMA_VERSION = 2
TARGET_HZ = 50.0
TARGET_DT_MS = 20

SIGNAL_FILE_V2 = "signal.i16.npy"
SIGNAL_FILE_V1 = "features.npy"
META_FILE = "meta.json"
META_FILE_V1 = "metadata.json"

#: Bytes per 50 Hz frame, used by the storage planner in the CLI.
BYTES_PER_FRAME_V1 = 13 * 4
BYTES_PER_FRAME_V2 = 9 * 2

#: Coarse mounting position recorded by the field operator.  One tap at
#: fastening time; it turns "does tail position matter?" from a guess into a
#: measurable covariate.
TAIL_POSITIONS = ("root", "mid", "tip", "unknown")


class CacheFormatError(RuntimeError):
    pass


@dataclass(frozen=True)
class Segment:
    """One contiguous run of 50 Hz samples. ``stop_index`` is exclusive."""

    segment_id: int
    start_index: int
    stop_index: int
    start_ms: int
    stop_ms: int

    @property
    def length(self) -> int:
        return int(self.stop_index - self.start_index)

    def to_dict(self) -> dict[str, int]:
        return {
            "segment_id": int(self.segment_id),
            "start_index": int(self.start_index),
            "stop_index": int(self.stop_index),
            "start_ms": int(self.start_ms),
            "stop_ms": int(self.stop_ms),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Segment":
        start = int(payload["start_index"])
        stop = int(payload["stop_index"])
        start_ms = int(payload.get("start_ms", start * TARGET_DT_MS))
        stop_ms = int(
            payload.get("stop_ms", start_ms + max(stop - start - 1, 0) * TARGET_DT_MS)
        )
        return cls(int(payload.get("segment_id", 0)), start, stop, start_ms, stop_ms)


@dataclass(frozen=True)
class Calibration:
    """Counts -> physical units. Stored per session, applied on read."""

    acc_divisor: float
    gyro_divisor: float = 32.0
    mag_divisor: float = 1000.0
    acc_bias_counts: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gyro_bias_counts: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "acc_divisor": float(self.acc_divisor),
            "gyro_divisor": float(self.gyro_divisor),
            "mag_divisor": float(self.mag_divisor),
            "acc_bias_counts": [float(v) for v in self.acc_bias_counts],
            "gyro_bias_counts": [float(v) for v in self.gyro_bias_counts],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Calibration":
        return cls(
            acc_divisor=float(payload["acc_divisor"]),
            gyro_divisor=float(payload.get("gyro_divisor", 32.0)),
            mag_divisor=float(payload.get("mag_divisor", 1000.0)),
            acc_bias_counts=tuple(float(v) for v in payload.get("acc_bias_counts", (0, 0, 0))),
            gyro_bias_counts=tuple(float(v) for v in payload.get("gyro_bias_counts", (0, 0, 0))),
        )


def intervals_to_mask(intervals: Iterable[Iterable[int]], start: int, stop: int) -> np.ndarray:
    """Expand sparse ``[lo, hi)`` index intervals into a dense flag for a slice."""

    length = max(0, int(stop) - int(start))
    flag = np.zeros(length, dtype=np.float32)
    if length == 0:
        return flag
    for item in intervals or ():
        lo, hi = int(item[0]), int(item[1])
        lo = max(lo, int(start))
        hi = min(hi, int(stop))
        if hi > lo:
            flag[lo - int(start) : hi - int(start)] = 1.0
    return flag


def mask_to_intervals(flag: np.ndarray) -> list[list[int]]:
    """Compress a dense 0/1 flag into ``[lo, hi)`` intervals."""

    values = np.asarray(flag).astype(bool)
    if values.size == 0 or not values.any():
        return []
    padded = np.concatenate(([False], values, [False]))
    change = np.flatnonzero(padded[1:] != padded[:-1])
    return [[int(change[i]), int(change[i + 1])] for i in range(0, len(change), 2)]


class SessionCache:
    """Read access to one cached session, schema 1 or 2, identical API."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        meta_path = self.directory / META_FILE
        legacy_meta_path = self.directory / META_FILE_V1
        if meta_path.exists():
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        elif legacy_meta_path.exists():
            payload = json.loads(legacy_meta_path.read_text(encoding="utf-8"))
        else:
            payload = {}
        self.meta: dict[str, Any] = payload

        if (self.directory / SIGNAL_FILE_V2).exists():
            self.schema_version = 2
            self._array = np.load(self.directory / SIGNAL_FILE_V2, mmap_mode="r")
            if self._array.ndim != 2 or self._array.shape[1] != 9:
                raise CacheFormatError(f"schema 2 signal must be (N, 9): {self.directory}")
            self.calibration: Calibration | None = Calibration.from_dict(payload["calibration"])
            self._quality_intervals = payload.get("quality_intervals", [])
        elif (self.directory / SIGNAL_FILE_V1).exists():
            self.schema_version = 1
            self._array = np.load(self.directory / SIGNAL_FILE_V1, mmap_mode="r")
            if self._array.ndim != 2 or self._array.shape[1] < 9:
                raise CacheFormatError(f"schema 1 signal must be (N, >=9): {self.directory}")
            self.calibration = None
            self._quality_intervals = None
        else:
            raise FileNotFoundError(f"no cache signal found under {self.directory}")

        self.n_frames = int(self._array.shape[0])
        raw_segments = payload.get("segments") or []
        if raw_segments:
            segments = [Segment.from_dict(item) for item in raw_segments]
            self.segments = [s for s in segments if s.stop_index > s.start_index]
        else:
            self.segments = [
                Segment(0, 0, self.n_frames, 0, max(0, self.n_frames - 1) * TARGET_DT_MS)
            ]
        # A schema-1 cache may declare a stop_index past the array (the
        # 20260818 demo does: 2999 frames, stop_index 2999).  Clip so a reader
        # never slices past the end of the memory map.
        self.segments = [
            Segment(
                s.segment_id,
                min(s.start_index, self.n_frames),
                min(s.stop_index, self.n_frames),
                s.start_ms,
                s.stop_ms,
            )
            for s in self.segments
        ]
        self.segments = [s for s in self.segments if s.stop_index > s.start_index]

    def close(self) -> None:
        """Release the memory-mapped signal file.

        Windows cannot unlink a file while a memory map keeps it open, so
        deleting a cache directory while its :class:`SessionCache` is alive
        fails with ``PermissionError``.  Call this, or use the instance as a
        context manager, before removing a cache directory on disk.
        """

        mmap = getattr(self._array, "_mmap", None)
        if mmap is not None:
            mmap.close()

    def __enter__(self) -> "SessionCache":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    @property
    def cache_key(self) -> str:
        return str(self.meta.get("cache_key", self.directory.name))

    @property
    def cow_id(self) -> str:
        return str(self.meta.get("cow_id", ""))

    @property
    def device_mac(self) -> str:
        return str(self.meta.get("device_mac", ""))

    @property
    def session_id(self) -> str:
        return str(self.meta.get("session_id", ""))

    @property
    def tail_position(self) -> str:
        """``root`` / ``mid`` / ``tip`` / ``unknown``.

        New metadata field.  Mounting position changes the lever arm and so the
        gyroscope amplitude of the same behaviour by a large factor; recording a
        coarse three-way label costs the field operator one tap and lets the
        effect be measured instead of guessed.
        """

        value = str(self.meta.get("tail_position", "unknown")).strip().lower()
        return value if value in TAIL_POSITIONS else "unknown"

    def segment_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        starts = np.asarray([s.start_index for s in self.segments], dtype=np.int64)
        stops = np.asarray([s.stop_index for s in self.segments], dtype=np.int64)
        return starts, stops

    def segment_of(self, index: int) -> Segment:
        starts, _ = self.segment_bounds()
        position = int(np.searchsorted(starts, int(index), side="right")) - 1
        if position < 0 or int(index) >= self.segments[position].stop_index:
            raise IndexError(f"index {index} falls outside every segment")
        return self.segments[position]

    # ------------------------------------------------------------------
    def physical(self, start: int = 0, stop: int | None = None) -> np.ndarray:
        """Return ``(n, 9)`` float32 in g / deg s-1 / gauss."""

        stop = self.n_frames if stop is None else int(stop)
        start = int(start)
        block = np.asarray(self._array[start:stop])
        if self.schema_version == 1:
            return block[:, 0:9].astype(np.float32, copy=True)
        counts = block.astype(np.float32)
        cal = self.calibration
        assert cal is not None
        acc = (counts[:, 0:3] - np.asarray(cal.acc_bias_counts, np.float32)) / cal.acc_divisor
        gyro = (counts[:, 3:6] - np.asarray(cal.gyro_bias_counts, np.float32)) / cal.gyro_divisor
        mag = rotate_mag_raw_to_imu(counts[:, 6:9]) / cal.mag_divisor
        return np.concatenate((acc, gyro, mag), axis=1).astype(np.float32, copy=False)

    def quality_flag(self, start: int = 0, stop: int | None = None) -> np.ndarray:
        stop = self.n_frames if stop is None else int(stop)
        if self.schema_version == 1:
            block = np.asarray(self._array[int(start) : stop])
            if block.shape[1] > 12:
                return block[:, 12].astype(np.float32, copy=True)
            return np.zeros(block.shape[0], dtype=np.float32)
        return intervals_to_mask(self._quality_intervals, int(start), stop)

    def times_ms(self, indices: np.ndarray) -> np.ndarray:
        """Absolute-in-session milliseconds for arbitrary frame indices."""

        idx = np.asarray(indices, dtype=np.int64)
        out = np.full(idx.shape, -1, dtype=np.int64)
        for segment in self.segments:
            inside = (idx >= segment.start_index) & (idx < segment.stop_index)
            if np.any(inside):
                out[inside] = (
                    segment.start_ms + (idx[inside] - segment.start_index) * TARGET_DT_MS
                )
        if np.any(out < 0):
            raise ValueError("frame index outside every segment; cannot assign a timestamp")
        return out

    def bytes_on_disk(self) -> int:
        total = 0
        for item in self.directory.iterdir():
            if item.is_file():
                total += item.stat().st_size
        return total

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"SessionCache({self.cache_key!r}, schema={self.schema_version}, "
            f"frames={self.n_frames}, segments={len(self.segments)})"
        )


def open_cache(directory: str | Path) -> SessionCache:
    return SessionCache(directory)


def write_cache_v2(
    directory: str | Path,
    *,
    counts: np.ndarray,
    segments: list[Segment] | list[dict[str, Any]],
    calibration: Calibration,
    quality_flag: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write one session in schema 2.

    ``counts`` must be ``(N, 9) int16`` raw device counts already resampled onto
    the 50 Hz grid.  Nothing is quantised here; the caller is responsible for
    keeping the resampling lossless with respect to the device output.
    """

    target = Path(directory)
    if target.exists() and not overwrite:
        raise FileExistsError(f"cache directory already exists: {target}")
    target.mkdir(parents=True, exist_ok=True)

    array = np.asarray(counts)
    if array.ndim != 2 or array.shape[1] != 9:
        raise ValueError(f"counts must be (N, 9); got {array.shape}")
    if array.dtype != np.int16:
        if not np.isfinite(array).all():
            raise ValueError("counts contain non-finite values")
        if array.min() < np.iinfo(np.int16).min or array.max() > np.iinfo(np.int16).max:
            raise ValueError("counts do not fit in int16; check the resampling stage")
        array = np.rint(array).astype(np.int16)
    np.save(target / SIGNAL_FILE_V2, array, allow_pickle=False)

    normalised: list[dict[str, int]] = []
    for item in segments:
        normalised.append(item.to_dict() if isinstance(item, Segment) else dict(item))

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sample_rate_hz": int(TARGET_HZ),
        "channels": [
            "acc_x_counts", "acc_y_counts", "acc_z_counts",
            "gyro_x_counts", "gyro_y_counts", "gyro_z_counts",
            "mag_raw_x_counts", "mag_raw_y_counts", "mag_raw_z_counts",
        ],
        "frames": int(array.shape[0]),
        "calibration": calibration.to_dict(),
        "segments": normalised,
        "quality_intervals": mask_to_intervals(quality_flag) if quality_flag is not None else [],
        "timestamp_rule": "start_ms + 20 * (index - start_index), per segment",
    }
    payload.update(metadata or {})
    (target / META_FILE).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target


def estimate_storage_bytes(cows: int, days_per_cow: float, schema: int = 2) -> dict[str, float]:
    """Capacity planner used by ``cowmata plan-storage``."""

    frames = float(cows) * float(days_per_cow) * 24.0 * 3600.0 * TARGET_HZ
    per_frame = BYTES_PER_FRAME_V2 if schema == 2 else BYTES_PER_FRAME_V1
    total = frames * per_frame
    return {
        "cows": float(cows),
        "days_per_cow": float(days_per_cow),
        "frames": frames,
        "schema": int(schema),
        "bytes_per_frame": float(per_frame),
        "bytes": total,
        "gigabytes": total / (1024.0**3),
    }
