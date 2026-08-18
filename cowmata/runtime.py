"""Device, precision and load-control policy for the training host.

This merges the 20260818 ``amp.py`` (51 lines) and ``load_control.py`` (60
lines).  They were separate files that were always imported together by every
training script and never used independently.

Two host-specific facts are encoded here and are worth stating explicitly,
because both were wrong in an earlier revision of the plan:

* **The training host is an RTX 3090 (Ampere).**  Ampere supports BF16, so the
  ``auto`` policy resolves to bf16, which needs no gradient scaler and does not
  suffer the overflow behaviour of fp16.  Nothing in this project should be
  tuned for Turing any more.
* **The host runs Windows.**  Sustained 100% CUDA load on a display-attached
  WDDM GPU can starve kernel DPCs and trigger bugcheck 0x133
  (DPC_WATCHDOG_VIOLATION).  The duty-cycle throttle is therefore *kept*, not
  deleted, and matters more at 24 GB than it did before because the dense
  trainer keeps the GPU busy for longer stretches.  It is off by default
  (``duty = 1.0``) and enabled through ``COWMATA_GPU_DUTY``.

The synchronisation is amortised over a window of batches so the normal GPU/CPU
pipeline overlap is preserved; a per-batch ``cuda.synchronize`` would serialise
the pipeline and cost several times more wall-clock time than it saves.

Torch is imported lazily.  ``cowmata check-env`` exists precisely to be run on a
machine whose deep-learning stack is not set up yet, so a module-level
``import torch`` made the one command that diagnoses a missing torch the one
command that could not run without it.
"""

from __future__ import annotations

import os
import platform
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

DUTY_ENV = "COWMATA_GPU_DUTY"
LEGACY_DUTY_ENV = "CATTLE_IMU_GPU_DUTY"
WINDOW = 16  # batches per duty-cycle calibration window


def torch_available() -> bool:
    """True when torch can be imported, without raising if it cannot."""

    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def _torch() -> Any:
    """Import torch, or fail with a message that names the actual problem."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "this operation needs PyTorch, which is not installed in the current "
            "environment; run `cowmata check-env` for a full report, and note that "
            "the cache, feature, GBDT and metric paths all work without it"
        ) from error
    return torch


def _cuda_available() -> bool:
    if not torch_available():
        return False
    return bool(_torch().cuda.is_available())


@dataclass(frozen=True)
class PrecisionPolicy:
    """Resolved precision policy.

    ``auto`` uses BF16 on supported accelerators and otherwise FP16 with
    gradient scaling.  FP32 remains available for debugging numerical issues.
    """

    name: str
    dtype: Any | None  # torch.dtype, typed loosely so torch stays optional
    use_scaler: bool
    device: str = "cuda"


def resolve_precision(requested: str = "auto", *, device: str = "cuda") -> PrecisionPolicy:
    torch = _torch()
    choice = str(requested).lower()
    if device == "cpu":
        if choice not in {"auto", "fp32"}:
            raise ValueError("CPU runs support only auto or fp32 precision")
        return PrecisionPolicy("fp32", None, False, "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for training. Install a CUDA-enabled PyTorch build on "
            "the training computer and run `cowmata check-env` first."
        )
    if choice == "auto":
        choice = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if choice == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was requested but this CUDA device does not support it")
        return PrecisionPolicy("bf16", torch.bfloat16, False, "cuda")
    if choice == "fp16":
        return PrecisionPolicy("fp16", torch.float16, True, "cuda")
    if choice == "fp32":
        return PrecisionPolicy("fp32", None, False, "cuda")
    raise ValueError(f"unknown precision policy: {requested}")


#: Historical name kept so old scripts keep importing successfully.
resolve_cuda_precision = resolve_precision


def autocast_context(policy: PrecisionPolicy):
    if policy.dtype is None:
        return nullcontext()
    return _torch().autocast(device_type=policy.device, dtype=policy.dtype)


def make_grad_scaler(policy: PrecisionPolicy) -> Any:
    torch = _torch()
    return torch.amp.GradScaler(policy.device, enabled=policy.use_scaler)


def duty_cycle_from_env() -> float:
    raw = os.environ.get(DUTY_ENV, os.environ.get(LEGACY_DUTY_ENV, "1.0")).strip()
    try:
        value = float(raw)
    except ValueError:
        value = 1.0
    if not 0.0 < value <= 1.0:
        value = 1.0
    return value


class DutyThrottle:
    """Sleep once per window of training batches to cap the GPU-active fraction."""

    def __init__(self, duty: float | None = None, window: int = WINDOW) -> None:
        self.duty = float(duty_cycle_from_env() if duty is None else duty)
        self.window = int(window)
        self._started = time.perf_counter()
        self._count = 0

    @property
    def active(self) -> bool:
        return self.duty < 1.0

    def tick(self) -> None:
        if self.duty >= 1.0:
            return
        self._count += 1
        if self._count < self.window:
            return
        if _cuda_available():
            _torch().cuda.synchronize()
        elapsed = time.perf_counter() - self._started
        sleep = elapsed * (1.0 / self.duty - 1.0)
        if sleep > 0.001:
            time.sleep(min(sleep, 10.0))
        self._started = time.perf_counter()
        self._count = 0


def dataloader_options(num_workers: int, *, pin_memory: bool = True) -> dict[str, object]:
    """Loader options that behave on Windows as well as Linux.

    On Windows a worker process is created with ``spawn``, not ``fork``, so it
    re-imports the whole package.  ``persistent_workers`` keeps that cost to
    once per run instead of once per epoch, which is why ``num_workers=0`` is
    the wrong fix for slow start-up on that host.
    """

    workers = max(0, int(num_workers))
    options: dict[str, object] = {
        "num_workers": workers,
        "pin_memory": bool(pin_memory) and _cuda_available(),
    }
    if workers > 0:
        options["persistent_workers"] = True
        options["prefetch_factor"] = 2
    return options


def environment_report() -> dict[str, object]:
    """Describe the host.  Never raises, including when torch is absent."""

    report: dict[str, object] = {
        "platform": platform.system(),
        "python": platform.python_version(),
        "gpu_duty": duty_cycle_from_env(),
        "torch_installed": torch_available(),
    }
    if not torch_available():
        report.update(
            {
                "torch": None,
                "cuda_available": False,
                "usable_without_torch": [
                    "build-cache", "build-features", "train-gbdt",
                    "predict (GBDT branch)", "check-data", "diagnose",
                    "plan-storage", "make-splits", "mine",
                ],
                "blocked_without_torch": ["train", "predict --deep-checkpoint"],
            }
        )
        return report

    torch = _torch()
    report.update(
        {
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime": torch.version.cuda,
        }
    )
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        report.update(
            {
                "gpu": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "gpu_memory_gib": round(properties.total_memory / (1024**3), 3),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
    if report["platform"] == "Windows":
        report["notes"] = [
            "WDDM: keep COWMATA_GPU_DUTY below 1.0 if the host bugchecks under sustained load",
            "mamba_ssm and several Triton kernels are Linux-only; WSL2 is the escape hatch",
        ]
    return report
