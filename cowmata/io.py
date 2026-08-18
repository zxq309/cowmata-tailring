"""Lossless readers for the V2 nine-axis tail-ring payload.

Raw JSON files are never edited.  The magnetometer mounting correction is
applied only to arrays returned by this module.

Decoding behaviour is byte-for-byte identical to the 20260818 baseline; only
the module location changed.  The frame phase search, garbage-prefix trimming
and gap splitting are all load-bearing and were left alone deliberately.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


V2_FRAME_DTYPE = np.dtype(
    [("elapsed_ms", "<u4"), ("values", "<i2", (9,))], align=False
)
V2_FRAME_BYTES = 22
RAW_CHANNELS = (
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "mag_raw_x",
    "mag_raw_y",
    "mag_raw_z",
)
MODEL_CHANNELS = (
    "acc_x_g",
    "acc_y_g",
    "acc_z_g",
    "gyro_x_dps",
    "gyro_y_dps",
    "gyro_z_dps",
    "mag_x_imu_gauss",
    "mag_y_imu_gauss",
    "mag_z_imu_gauss",
)

# User-confirmed mounting relation:
# raw magnetometer +X points toward IMU -Y;
# raw magnetometer +Y points toward IMU +X.
# Therefore v_imu = R @ v_raw, with det(R) == +1.
MAG_RAW_TO_IMU = np.asarray(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)


@dataclass(frozen=True)
class DecodeDiagnostics:
    payload_bytes: int
    frame_offset_bytes: int
    dropped_prefix_bytes: int
    dropped_trailing_bytes: int
    phase_score: float
    frames: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "payload_bytes": int(self.payload_bytes),
            "frame_offset_bytes": int(self.frame_offset_bytes),
            "dropped_prefix_bytes": int(self.dropped_prefix_bytes),
            "dropped_trailing_bytes": int(self.dropped_trailing_bytes),
            "phase_score": float(self.phase_score),
            "frames": int(self.frames),
        }


@dataclass(frozen=True)
class ImuSession:
    path: Path
    device: str
    create_time_ms: int | None
    elapsed_ms: np.ndarray
    raw_values: np.ndarray
    diagnostics: DecodeDiagnostics
    json_metadata: dict[str, Any]

    def physical_values(
        self,
        *,
        acc_divisor: float,
        acc_bias_counts: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
        gyro_divisor: float = 32.0,
        gyro_bias_counts: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
        mag_divisor: float = 1000.0,
    ) -> np.ndarray:
        """Return 9 channels in the common IMU coordinate frame.

        Output units are g, degree/s, and gauss.  The acceleration divisor is
        explicit because some devices do not match the 4096 counts/g metadata.
        """

        if acc_divisor <= 0 or gyro_divisor <= 0 or mag_divisor <= 0:
            raise ValueError("all sensitivity divisors must be positive")
        values = self.raw_values.astype(np.float32, copy=False)
        acc_bias = np.asarray(acc_bias_counts, dtype=np.float32)
        gyro_bias = np.asarray(gyro_bias_counts, dtype=np.float32)
        if acc_bias.shape != (3,) or gyro_bias.shape != (3,):
            raise ValueError("accelerometer and gyroscope bias must each have shape (3,)")
        acc = (values[:, 0:3] - acc_bias) / float(acc_divisor)
        gyro = (values[:, 3:6] - gyro_bias) / float(gyro_divisor)
        mag = rotate_mag_raw_to_imu(values[:, 6:9]) / float(mag_divisor)
        return np.concatenate((acc, gyro, mag), axis=1).astype(np.float32, copy=False)


def rotate_mag_raw_to_imu(raw_mag: np.ndarray) -> np.ndarray:
    """Rotate (..., 3) raw magnetometer samples into the IMU coordinate frame."""

    raw_mag = np.asarray(raw_mag)
    if raw_mag.shape[-1] != 3:
        raise ValueError(f"expected last magnetometer dimension 3, got {raw_mag.shape}")
    return raw_mag @ MAG_RAW_TO_IMU.T


def _phase_score(payload: bytes, offset: int, max_frames: int = 8000) -> float:
    available = len(payload) - offset
    count = available // V2_FRAME_BYTES
    if count < 10:
        return float("-inf")
    sample_count = min(count, max_frames)
    view = memoryview(payload)[offset : offset + sample_count * V2_FRAME_BYTES]
    frames = np.frombuffer(view, dtype=V2_FRAME_DTYPE, count=sample_count)
    time_ms = frames["elapsed_ms"].astype(np.int64)
    dt = np.diff(time_ms)
    positive = float(np.mean(dt > 0))
    nominal = float(np.mean((dt >= 10) & (dt <= 40)))
    plausible = float(np.mean((dt >= 1) & (dt <= 1000)))
    positive_dt = dt[dt > 0]
    if positive_dt.size:
        median_dt = float(np.median(positive_dt))
        cadence = float(np.exp(-abs(median_dt - 20.0) / 20.0))
    else:
        cadence = 0.0
    return 0.55 * nominal + 0.25 * positive + 0.10 * plausible + 0.10 * cadence


def _first_plausible_frame(time_ms: np.ndarray, run: int = 8) -> int:
    """Trim complete garbage frames that can precede an aligned V2 payload."""

    if time_ms.size <= run:
        return 0
    dt = np.diff(time_ms.astype(np.int64))
    good = ((dt >= 5) & (dt <= 100)).astype(np.int16)
    hits = np.convolve(good, np.ones(run, dtype=np.int16), mode="valid")
    matched = np.flatnonzero(hits == run)
    return int(matched[0]) if matched.size else 0


def decode_v2_bytes(payload: bytes, *, min_phase_score: float = 0.80) -> tuple[np.ndarray, np.ndarray, DecodeDiagnostics]:
    """Decode ``<I9h`` frames, recovering a polluted byte prefix when possible."""

    if V2_FRAME_DTYPE.itemsize != V2_FRAME_BYTES:
        raise RuntimeError("internal V2 frame layout is not 22 bytes")
    if len(payload) < V2_FRAME_BYTES * 10:
        raise ValueError("V2 payload is too short")

    scores = np.asarray([_phase_score(payload, offset) for offset in range(V2_FRAME_BYTES)])
    best_offset = int(np.argmax(scores))
    best_score = float(scores[best_offset])
    if not np.isfinite(best_score) or best_score < min_phase_score:
        raise ValueError(f"could not recover a credible V2 frame phase (score={best_score:.3f})")

    frame_count = (len(payload) - best_offset) // V2_FRAME_BYTES
    trailing = (len(payload) - best_offset) % V2_FRAME_BYTES
    view = memoryview(payload)[best_offset : best_offset + frame_count * V2_FRAME_BYTES]
    frames = np.frombuffer(view, dtype=V2_FRAME_DTYPE, count=frame_count)
    time_ms = frames["elapsed_ms"].astype(np.int64)
    leading_frames = _first_plausible_frame(time_ms)
    if leading_frames:
        frames = frames[leading_frames:]
        time_ms = time_ms[leading_frames:]

    raw_values = np.asarray(frames["values"], dtype=np.int16).copy()
    diagnostics = DecodeDiagnostics(
        payload_bytes=len(payload),
        frame_offset_bytes=best_offset,
        dropped_prefix_bytes=best_offset + leading_frames * V2_FRAME_BYTES,
        dropped_trailing_bytes=trailing,
        phase_score=best_score,
        frames=int(frames.size),
    )
    return time_ms.copy(), raw_values, diagnostics


def load_v2_json(path: str | Path) -> ImuSession:
    """Load one V2 JSON recording without modifying it."""

    source = Path(path)
    with source.open("r", encoding="utf-8-sig") as handle:
        document = json.load(handle)
    encoded = document.get("imu")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError(f"missing non-empty 'imu' Base64 field: {source}")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except ValueError:
        payload = base64.b64decode("".join(encoded.split()), validate=True)
    elapsed_ms, raw_values, diagnostics = decode_v2_bytes(payload)
    metadata = {key: value for key, value in document.items() if key != "imu"}
    create_time = document.get("create_time")
    return ImuSession(
        path=source,
        device=str(document.get("device", "")),
        create_time_ms=int(create_time) if isinstance(create_time, (int, float)) else None,
        elapsed_ms=elapsed_ms,
        raw_values=raw_values,
        diagnostics=diagnostics,
        json_metadata=metadata,
    )


def contiguous_slices(elapsed_ms: np.ndarray, *, gap_threshold_ms: float = 100.0) -> list[slice]:
    """Return runs that never cross a reset, duplicate, or long recording gap."""

    time_ms = np.asarray(elapsed_ms, dtype=np.int64)
    if time_ms.ndim != 1:
        raise ValueError("elapsed_ms must be one-dimensional")
    if time_ms.size == 0:
        return []
    dt = np.diff(time_ms)
    boundaries = np.flatnonzero((dt <= 0) | (dt > gap_threshold_ms)) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [time_ms.size]))
    return [slice(int(start), int(stop)) for start, stop in zip(starts, stops) if stop > start]
