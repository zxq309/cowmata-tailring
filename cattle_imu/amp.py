"""CUDA and automatic mixed-precision helpers shared by training scripts."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CudaPrecision:
    """Resolved CUDA precision policy.

    ``auto`` uses BF16 on supported accelerators and otherwise uses FP16 with
    gradient scaling.  FP32 remains available for debugging numerical issues.
    """

    name: str
    dtype: torch.dtype | None
    use_scaler: bool


def resolve_cuda_precision(requested: str = "auto") -> CudaPrecision:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for training. Install a CUDA-enabled PyTorch build "
            "on the training computer and run scripts/verify_environment.py first."
        )
    choice = str(requested).lower()
    if choice == "auto":
        choice = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if choice == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was requested but this CUDA device does not support it")
        return CudaPrecision("bf16", torch.bfloat16, False)
    if choice == "fp16":
        return CudaPrecision("fp16", torch.float16, True)
    if choice == "fp32":
        return CudaPrecision("fp32", None, False)
    raise ValueError(f"unknown precision policy: {requested}")


def autocast_context(policy: CudaPrecision):
    if policy.dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=policy.dtype)


def make_grad_scaler(policy: CudaPrecision) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=policy.use_scaler)
