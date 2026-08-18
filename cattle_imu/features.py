"""Hand-crafted feature bank for tail-ring IMU behaviour recognition.

Design notes
------------
* **Gravity / motion separation.**  Tilt angles are computed from the
  *static* (low-pass, 0.3 Hz) acceleration component and motion intensity from
  the *dynamic* (high-pass) component, following the standard livestock
  accelerometer pipeline (ODBA / VeDBA / pitch / roll).  Computing tilt from
  raw acceleration - as the earlier ``_learn_tail_features.py`` did - lets
  motion artefacts leak into the angle.
* **Session self-calibration.**  The tail ring is mounted at an arbitrary
  angle on every animal, so the raw device axes are not comparable across
  cows.  ``session_reference`` estimates the dominant gravity direction from
  low-motion samples and every angular feature is expressed *relative* to it.
  This is a software-only, post-hoc step: it needs no hardware action and
  works on data that has already been collected.
* **Segment safety.**  Every filter and every rolling window is applied inside
  one contiguous segment.  Nothing ever bridges a data gap.
* **Causality switch.**  ``causal=True`` uses one-sided filters and trailing
  windows (online model).  ``causal=False`` uses zero-phase filters and
  centred windows (offline retrospective model).  Never mix the two between
  training and evaluation.

All rolling statistics are O(N) via cumulative sums; a 24 h session is
processed in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:  # scipy is already a hard dependency of the project
    from scipy.signal import butter, sosfilt, sosfiltfilt
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - degraded fallback
    _HAVE_SCIPY = False


SAMPLE_RATE_HZ = 50.0
GRAVITY_CUTOFF_HZ = 0.3
# Rolling scales in seconds. 0.5 s ~ one label step, 30 s ~ a defecation event.
SCALE_SECONDS = (1.0, 5.0, 15.0, 30.0)
# Transition half-spans in seconds, used for stand-up / lie-down features.
TRANSITION_SECONDS = (2.0, 5.0, 10.0)
TILT_PLATEAU_DEG = (5.0, 10.0, 20.0, 40.0)
LONG_BASELINE_SECONDS = 1800.0  # 30 min drift-tracking baseline


# --------------------------------------------------------------------------
# rolling primitives
# --------------------------------------------------------------------------
def window_stats(values: np.ndarray, lo_offset: int, hi_offset: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation over ``[i+lo_offset, i+hi_offset)`` clipped to the array.

    ``lo_offset`` is negative for history.  Both bounds are sample offsets, the
    upper one exclusive.  Edge windows are shortened, never wrapped or padded,
    so a value is never influenced by another segment.
    """

    x = np.asarray(values, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        return np.empty(0), np.empty(0)
    cs = np.concatenate(([0.0], np.cumsum(x)))
    cs2 = np.concatenate(([0.0], np.cumsum(x * x)))
    idx = np.arange(n, dtype=np.int64)
    lo = np.clip(idx + int(lo_offset), 0, n - 1)
    hi = np.clip(idx + int(hi_offset), 1, n)
    hi = np.maximum(hi, lo + 1)
    lo = np.minimum(lo, hi - 1)
    count = (hi - lo).astype(np.float64)
    total = cs[hi] - cs[lo]
    total2 = cs2[hi] - cs2[lo]
    mean = total / count
    var = np.maximum(total2 / count - mean * mean, 0.0)
    return mean, np.sqrt(var)


def window_max(values: np.ndarray, lo_offset: int, hi_offset: int) -> np.ndarray:
    """Rolling maximum. Uses pandas' O(N) implementation on a shifted view."""

    x = pd.Series(np.asarray(values, dtype=np.float64).ravel())
    span = int(hi_offset) - int(lo_offset)
    if span <= 0:
        raise ValueError("hi_offset must exceed lo_offset")
    rolled = x.rolling(window=span, min_periods=1).max().to_numpy()
    # ``rolling`` is trailing: position i covers [i-span+1, i].  Shift so it
    # covers [i+lo, i+hi).
    shift = int(hi_offset) - 1
    out = np.empty_like(rolled)
    if shift > 0:
        out[:-shift or None] = rolled[shift:]
        out[len(out) - shift:] = rolled[-1]
    elif shift < 0:
        out[-shift:] = rolled[:shift]
        out[:-shift] = rolled[0]
    else:
        out = rolled
    return out


def _bandpass(values: np.ndarray, low: float, high: float, *, causal: bool) -> np.ndarray:
    n = values.shape[0]
    if not _HAVE_SCIPY or n < 64:
        return np.asarray(values, dtype=np.float64)
    nyq = SAMPLE_RATE_HZ / 2.0
    low_n = max(low / nyq, 1e-4)
    high_n = min(high / nyq, 0.99)
    if high_n <= low_n:
        return np.asarray(values, dtype=np.float64)
    sos = butter(4, [low_n, high_n], btype="bandpass", output="sos")
    data = np.asarray(values, dtype=np.float64)
    if causal:
        return sosfilt(sos, data, axis=0)
    padlen = 3 * (sos.shape[0] * 2)
    if n <= padlen + 1:
        return sosfilt(sos, data, axis=0)
    return sosfiltfilt(sos, data, axis=0)


def gravity_split(acc: np.ndarray, *, causal: bool) -> tuple[np.ndarray, np.ndarray]:
    """Split acceleration into static (gravity) and dynamic (motion) parts."""

    data = np.asarray(acc, dtype=np.float64)
    n = data.shape[0]
    if not _HAVE_SCIPY or n < 64:
        static = np.repeat(data.mean(axis=0, keepdims=True), n, axis=0)
        return static, data - static
    nyq = SAMPLE_RATE_HZ / 2.0
    sos = butter(4, GRAVITY_CUTOFF_HZ / nyq, btype="lowpass", output="sos")
    if causal:
        static = sosfilt(sos, data, axis=0)
    else:
        padlen = 3 * (sos.shape[0] * 2)
        static = sosfiltfilt(sos, data, axis=0) if n > padlen + 1 else sosfilt(sos, data, axis=0)
    return static, data - static


# --------------------------------------------------------------------------
# session calibration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SessionReference:
    """Estimated mounting orientation of one ring on one animal."""

    gravity_unit: np.ndarray  # (3,) dominant gravity direction, device frame
    rotation: np.ndarray  # (3, 3) device -> body frame
    quiet_fraction: float

    def to_dict(self) -> dict[str, object]:
        return {
            "gravity_unit": [float(v) for v in self.gravity_unit],
            "rotation": [[float(v) for v in row] for row in self.rotation],
            "quiet_fraction": float(self.quiet_fraction),
        }


def session_reference(static_acc: np.ndarray, vedba: np.ndarray, *, quiet_quantile: float = 0.3) -> SessionReference:
    """Estimate the mounting orientation from low-motion samples.

    Only samples in the quietest ``quiet_quantile`` of VeDBA are used, so the
    reference tracks the resting orientation of the tail rather than an average
    that is dragged around by walking bouts.
    """

    static = np.asarray(static_acc, dtype=np.float64)
    motion = np.asarray(vedba, dtype=np.float64).ravel()
    if static.shape[0] == 0:
        unit = np.asarray([0.0, 0.0, 1.0])
        return SessionReference(unit, np.eye(3), 0.0)
    threshold = np.quantile(motion, quiet_quantile) if motion.size else np.inf
    quiet = motion <= threshold
    if quiet.sum() < 10:
        quiet = np.ones(static.shape[0], dtype=bool)
    reference = np.median(static[quiet], axis=0)
    norm = float(np.linalg.norm(reference))
    unit = reference / norm if norm > 1e-9 else np.asarray([0.0, 0.0, 1.0])
    return SessionReference(unit, _rotation_to_z(unit), float(quiet.mean()))


def _rotation_to_z(unit: np.ndarray) -> np.ndarray:
    """Rotation matrix mapping ``unit`` onto +Z (Rodrigues, shortest arc)."""

    source = np.asarray(unit, dtype=np.float64)
    target = np.asarray([0.0, 0.0, 1.0])
    v = np.cross(source, target)
    c = float(np.dot(source, target))
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.asarray([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


# --------------------------------------------------------------------------
# derived series
# --------------------------------------------------------------------------
def derived_series(
    segment_array: np.ndarray,
    *,
    causal: bool,
    reference: SessionReference | None = None,
) -> dict[str, np.ndarray]:
    """Compute every base time-series for one contiguous segment."""

    arr = np.asarray(segment_array, dtype=np.float64)
    acc = arr[:, 0:3]
    gyro = arr[:, 3:6]
    quality = arr[:, 12] if arr.shape[1] > 12 else np.zeros(arr.shape[0])

    static, dynamic = gravity_split(acc, causal=causal)
    static_norm = np.linalg.norm(static, axis=1)
    safe_norm = np.maximum(static_norm, 1e-9)
    unit = static / safe_norm[:, None]

    vedba = np.linalg.norm(dynamic, axis=1)
    odba = np.abs(dynamic).sum(axis=1)

    if reference is None:
        reference = session_reference(static, vedba)
    body_unit = unit @ reference.rotation.T
    tilt_cos = np.clip(body_unit[:, 2], -1.0, 1.0)
    tilt_deg = np.degrees(np.arccos(tilt_cos))

    pitch = np.degrees(np.arctan2(static[:, 0], np.hypot(static[:, 1], static[:, 2])))
    roll = np.degrees(np.arctan2(static[:, 1], np.hypot(static[:, 0], static[:, 2])))

    gyro_norm = np.linalg.norm(gyro, axis=1)
    band_low = _bandpass(gyro, 0.5, 2.0, causal=causal)
    band_mid = _bandpass(gyro, 2.0, 5.0, causal=causal)
    band_high = _bandpass(gyro, 5.0, 15.0, causal=causal)

    return {
        "tilt_deg": tilt_deg,
        "body_ux": body_unit[:, 0],
        "body_uy": body_unit[:, 1],
        "body_uz": body_unit[:, 2],
        "pitch_deg": pitch,
        "roll_deg": roll,
        "static_norm": static_norm,
        "vedba": vedba,
        "odba": odba,
        "gyro_norm": gyro_norm,
        "gyro_x": gyro[:, 0],
        "gyro_y": gyro[:, 1],
        "gyro_z": gyro[:, 2],
        "dyn_x": dynamic[:, 0],
        "dyn_y": dynamic[:, 1],
        "dyn_z": dynamic[:, 2],
        "band_low_energy": np.square(band_low).sum(axis=1),
        "band_mid_energy": np.square(band_mid).sum(axis=1),
        "band_high_energy": np.square(band_high).sum(axis=1),
        "quality_flag": quality,
    }


def _offsets(seconds: float, *, causal: bool) -> tuple[int, int]:
    span = max(1, int(round(seconds * SAMPLE_RATE_HZ)))
    if causal:
        return -span + 1, 1
    half = span // 2
    return -half, half + 1


def segment_features(
    segment_array: np.ndarray,
    local_centers: np.ndarray,
    *,
    causal: bool,
    reference: SessionReference | None = None,
) -> pd.DataFrame:
    """Feature table for the given centre positions *within one segment*."""

    series = derived_series(segment_array, causal=causal, reference=reference)
    centers = np.asarray(local_centers, dtype=np.int64)
    n = segment_array.shape[0]
    if centers.size == 0:
        return pd.DataFrame()
    if centers.min() < 0 or centers.max() >= n:
        raise IndexError("local centre outside segment")

    columns: dict[str, np.ndarray] = {}

    def add(name: str, values: np.ndarray) -> None:
        columns[name] = np.asarray(values, dtype=np.float32)[centers]

    tilt = series["tilt_deg"]
    for seconds in SCALE_SECONDS:
        lo, hi = _offsets(seconds, causal=causal)
        tag = f"{seconds:g}s".replace(".", "p")
        mean, std = window_stats(tilt, lo, hi)
        add(f"tilt_mean_{tag}", mean)
        add(f"tilt_std_{tag}", std)
        add(f"tilt_max_{tag}", window_max(tilt, lo, hi))
        for degrees in TILT_PLATEAU_DEG:
            frac, _ = window_stats((tilt > degrees).astype(np.float64), lo, hi)
            add(f"tilt_frac_gt{int(degrees)}_{tag}", frac)
        for key in ("vedba", "odba", "gyro_norm"):
            mean, std = window_stats(series[key], lo, hi)
            add(f"{key}_mean_{tag}", mean)
            add(f"{key}_std_{tag}", std)
        add("vedba_max_" + tag, window_max(series["vedba"], lo, hi))
        for key in ("band_low_energy", "band_mid_energy", "band_high_energy"):
            mean, _ = window_stats(series[key], lo, hi)
            add(f"{key}_mean_{tag}", mean)
        if seconds <= 5.0:
            for key in ("gyro_x", "gyro_y", "gyro_z", "dyn_x", "dyn_y", "dyn_z"):
                _, std = window_stats(series[key], lo, hi)
                add(f"{key}_std_{tag}", std)
            for key in ("body_ux", "body_uy", "body_uz", "pitch_deg", "roll_deg"):
                mean, _ = window_stats(series[key], lo, hi)
                add(f"{key}_mean_{tag}", mean)

    # Posture-transition geometry: how far the gravity direction moved between
    # a window before and a window after (or, when causal, two past windows).
    for seconds in TRANSITION_SECONDS:
        span = max(1, int(round(seconds * SAMPLE_RATE_HZ)))
        tag = f"{seconds:g}s".replace(".", "p")
        if causal:
            before = (-2 * span, -span)
            after = (-span, 1)
        else:
            before = (-2 * span, -span)
            after = (span, 2 * span)
        before_vec = np.stack(
            [window_stats(series[k], *before)[0] for k in ("body_ux", "body_uy", "body_uz")],
            axis=1,
        )
        after_vec = np.stack(
            [window_stats(series[k], *after)[0] for k in ("body_ux", "body_uy", "body_uz")],
            axis=1,
        )
        before_vec /= np.maximum(np.linalg.norm(before_vec, axis=1, keepdims=True), 1e-9)
        after_vec /= np.maximum(np.linalg.norm(after_vec, axis=1, keepdims=True), 1e-9)
        cosine = np.clip((before_vec * after_vec).sum(axis=1), -1.0, 1.0)
        add(f"transition_angle_{tag}", np.degrees(np.arccos(cosine)))
        add(
            f"transition_tilt_delta_{tag}",
            window_stats(tilt, *after)[0] - window_stats(tilt, *before)[0],
        )
        add(
            f"transition_vedba_ratio_{tag}",
            (window_stats(series["vedba"], *after)[0] + 1e-6)
            / (window_stats(series["vedba"], *before)[0] + 1e-6),
        )

    # Long-horizon drift baseline: tilt relative to the resting tilt of the
    # last 30 minutes.  Always causal so it cannot leak future posture.
    baseline = _long_baseline(tilt)
    add("tilt_rel_baseline", tilt - baseline)
    add("tilt_baseline", baseline)
    mean, _ = window_stats(series["quality_flag"], *_offsets(5.0, causal=causal))
    add("quality_mean_5s", mean)
    columns["segment_position"] = (centers / max(n - 1, 1)).astype(np.float32)
    columns["segment_length"] = np.full(centers.size, float(n), dtype=np.float32)
    return pd.DataFrame(columns)


def _long_baseline(tilt: np.ndarray, seconds: float = LONG_BASELINE_SECONDS) -> np.ndarray:
    """Trailing median of tilt, evaluated on a 1 Hz decimation for speed."""

    n = tilt.size
    step = int(SAMPLE_RATE_HZ)
    coarse = pd.Series(tilt[::step])
    window = max(1, int(seconds))
    median = coarse.rolling(window=window, min_periods=1).median().to_numpy()
    return np.repeat(median, step)[:n] if median.size * step >= n else np.interp(
        np.arange(n), np.arange(median.size) * step, median
    )


FEATURE_PREFIXES_DROPPED_WHEN_CAUSAL = ()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Names of the numeric feature columns produced by :func:`segment_features`."""

    reserved = {
        "sample_id",
        "cache_key",
        "cow_id",
        "device_key",
        "device_mac",
        "session_id",
        "center_index",
        "center_time_ms",
        "segment_id",
        "segment_start_index",
        "segment_stop_index",
        "body_target",
        "posture_target",
        "locomotion_target",
        "sampling_source",
    }
    return [
        column
        for column in frame.columns
        if column not in reserved
        and not column.startswith("event_")
        and not column.startswith("mask_")
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
