"""Explicit GPU duty-cycle control for training on WDDM machines.

Sustained 100% CUDA load on a display-attached WDDM GPU can starve kernel
DPCs and trigger bugcheck 0x133 (DPC_WATCHDOG_VIOLATION) on this host.
Training scripts tick the throttle after every training batch; the sleep keeps
the GPU-active fraction near ``CATTLE_IMU_GPU_DUTY`` without changing the math
of the training loop.

Synchronisation is amortised over a small window of batches so the normal
GPU/CPU pipeline overlap is preserved; a per-batch ``cuda.synchronize`` would
serialise the pipeline and cost several times more wall-clock time.

Default duty is 1.0 (no throttling), so behavior is unchanged unless the
environment variable is set.
"""

from __future__ import annotations

import os
import time

import torch

DUTY_ENV = "CATTLE_IMU_GPU_DUTY"
WINDOW = 16  # batches per duty-cycle calibration window


def duty_cycle_from_env() -> float:
    raw = os.environ.get(DUTY_ENV, "1.0").strip()
    try:
        value = float(raw)
    except ValueError:
        value = 1.0
    if not 0.0 < value <= 1.0:
        value = 1.0
    return value


class DutyThrottle:
    """Sleep once per window of training batches to cap the GPU-active fraction."""

    def __init__(self, duty: float, window: int = WINDOW) -> None:
        self.duty = float(duty)
        self.window = int(window)
        self._started = time.perf_counter()
        self._count = 0

    def tick(self) -> None:
        if self.duty >= 1.0:
            return
        self._count += 1
        if self._count < self.window:
            return
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._started
        sleep = elapsed * (1.0 / self.duty - 1.0)
        if sleep > 0.001:
            time.sleep(min(sleep, 10.0))
        self._started = time.perf_counter()
        self._count = 0
