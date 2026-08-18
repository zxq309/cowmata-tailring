"""Segment-safe window bounds.

This module contains the *only* place where a model context window is turned
into array slice bounds.  It is deliberately free of torch so the contract can
be unit tested without a GPU or a deep-learning install.

Two modes exist:

``causal``
    The window ends at the label centre and extends backwards.  Nothing after
    the centre is ever read.  Used for the online / streaming model.

``centered``
    The window is centred on the label.  Used for the offline (retrospective)
    model that mines annotation candidates on a server.  It is *not* causal by
    design; see ``README_V2.md`` section 3.

In both modes the window is clipped to the enclosing contiguous segment, so a
context can never bridge a data gap (the P0-1 defect of the previous version).
"""

from __future__ import annotations

import numpy as np

__all__ = ["context_bounds", "context_bounds_batch", "WindowSpec"]


class WindowSpec:
    """Immutable description of the window geometry."""

    __slots__ = ("context_samples", "mode")

    def __init__(self, context_samples: int, mode: str = "causal") -> None:
        if int(context_samples) <= 0:
            raise ValueError("context_samples must be positive")
        if mode not in {"causal", "centered"}:
            raise ValueError(f"unknown window mode: {mode!r}")
        self.context_samples = int(context_samples)
        self.mode = str(mode)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"WindowSpec(context_samples={self.context_samples}, mode={self.mode!r})"


def context_bounds(
    center: int,
    segment_start: int,
    segment_stop: int,
    context_samples: int,
    mode: str = "causal",
) -> tuple[int, int, int]:
    """Return ``(start, stop, dest_start)`` for one label centre.

    ``[start, stop)`` indexes the cached session array; ``dest_start`` is where
    that slice must be written inside a zero-filled buffer of length
    ``context_samples``.  ``stop - start`` is the number of real samples, i.e.
    the ``valid_length`` used for sample filtering and loss weighting.

    ``segment_stop`` is exclusive, matching ``ProcessedSession.segments``.
    """

    center = int(center)
    segment_start = int(segment_start)
    segment_stop = int(segment_stop)
    context = int(context_samples)
    if context <= 0:
        raise ValueError("context_samples must be positive")
    if segment_stop <= segment_start:
        raise ValueError("empty segment")
    if not (segment_start <= center < segment_stop):
        raise IndexError(
            f"center {center} outside segment [{segment_start}, {segment_stop})"
        )

    if mode == "causal":
        stop = center + 1
        start = max(segment_start, stop - context)
        dest_start = context - (stop - start)
    elif mode == "centered":
        half = context // 2
        ideal_start = center - half
        start = max(segment_start, ideal_start)
        stop = min(segment_stop, ideal_start + context)
        dest_start = start - ideal_start
    else:
        raise ValueError(f"unknown window mode: {mode!r}")
    return int(start), int(stop), int(dest_start)


def context_bounds_batch(
    centers: np.ndarray,
    segment_starts: np.ndarray,
    segment_stops: np.ndarray,
    context_samples: int,
    mode: str = "causal",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised :func:`context_bounds` used when filtering a sample table."""

    centers = np.asarray(centers, dtype=np.int64)
    segment_starts = np.asarray(segment_starts, dtype=np.int64)
    segment_stops = np.asarray(segment_stops, dtype=np.int64)
    context = int(context_samples)
    if context <= 0:
        raise ValueError("context_samples must be positive")
    if mode == "causal":
        stop = centers + 1
        start = np.maximum(segment_starts, stop - context)
        dest = context - (stop - start)
    elif mode == "centered":
        half = context // 2
        ideal = centers - half
        start = np.maximum(segment_starts, ideal)
        stop = np.minimum(segment_stops, ideal + context)
        dest = start - ideal
    else:
        raise ValueError(f"unknown window mode: {mode!r}")
    return start, stop, dest
