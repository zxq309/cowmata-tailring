"""Fail-fast model smoke test for the CPU or CUDA environment."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import yaml

from cattle_imu.amp import CudaPrecision, autocast_context, make_grad_scaler, resolve_cuda_precision
from cattle_imu.model import CausalMultiTaskTCN


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda":
        policy = resolve_cuda_precision(args.precision)
        device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(device)
    else:
        if args.precision not in ("auto", "fp32"):
            raise ValueError("CPU smoke tests support only auto or fp32 precision")
        policy = CudaPrecision("fp32", None, False)
        device = torch.device("cpu")
        properties = None
    torch.manual_seed(20260814)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(20260814)
    model = CausalMultiTaskTCN(in_channels=8, width=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = make_grad_scaler(policy) if args.device == "cuda" else None
    sample = torch.randn((2, 8, 2048), device=device)  # Covers the largest 1,500-sample context.
    target = torch.randn((2, 6), device=device)
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(policy):
        prediction = model(sample)["event_logits"]
        loss = torch.nn.functional.mse_loss(prediction.float(), target)
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    if args.device == "cuda":
        torch.cuda.synchronize()
    result = {
        "status": "PASS",
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "device": args.device,
        "cuda_available": torch.cuda.is_available(),
        "gpu": properties.name if properties is not None else None,
        "compute_capability": (
            f"{properties.major}.{properties.minor}" if properties is not None else None
        ),
        "gpu_memory_gib": (
            round(properties.total_memory / (1024**3), 3) if properties is not None else None
        ),
        "bf16_supported": torch.cuda.is_bf16_supported() if properties is not None else False,
        "selected_precision": policy.name,
        "smoke_test_seconds": round(time.perf_counter() - start, 4),
        "loss": float(loss.detach().cpu()),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "pyyaml": yaml.__version__,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
