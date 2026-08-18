# -*- coding: utf-8 -*-
"""Model contracts that require torch and pytorch-tcn. Run on the training host.

    python tests/test_torch_contracts.py

These are the tests that catch the failure mode "training metrics look fine but
production is wrong": training uses ``forward``/``forward_last`` while
continuous inference uses ``forward_dense``.  If those two ever disagree - for
instance because a pytorch-tcn upgrade changes the internal weight semantics
that ``forward_last`` relies on - every downstream number silently becomes
meaningless.  Pin ``pytorch-tcn`` to an exact version and keep this test green.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from cattle_imu.model import (  # noqa: E402
    CausalMultiTaskTCN,
    OfflineMultiTaskTCN,
    build_model,
    parameter_count,
)

CHANNELS = 8
LENGTH = 2048
EVENTS = ("STANDING_UP", "URINATION")


def _model(mode: str) -> torch.nn.Module:
    torch.manual_seed(0)
    model = build_model(
        mode,
        in_channels=CHANNELS,
        event_codes=EVENTS,
        width=16,
        dropout=0.0,
        event_context_seconds={"STANDING_UP": 4.0, "URINATION": 8.0},
    )
    model.eval()
    return model


def test_causal_forward_matches_dense_last_position() -> None:
    model = _model("causal")
    inputs = torch.randn(2, CHANNELS, LENGTH)
    with torch.no_grad():
        fast = model(inputs)
        dense = model.forward_dense(inputs)
    for key in ("posture_logits", "locomotion_logits", "event_logits"):
        torch.testing.assert_close(fast[key], dense[key][..., -1], rtol=1e-4, atol=1e-5)
    print("PASS test_causal_forward_matches_dense_last_position")


def test_offline_forward_matches_dense_center_position() -> None:
    model = _model("offline")
    inputs = torch.randn(2, CHANNELS, LENGTH)
    with torch.no_grad():
        fast = model(inputs)
        dense = model.forward_dense(inputs)
    center = LENGTH // 2
    for key in ("posture_logits", "locomotion_logits", "event_logits"):
        torch.testing.assert_close(fast[key], dense[key][..., center], rtol=1e-4, atol=1e-5)
    print("PASS test_offline_forward_matches_dense_center_position")


def test_causal_model_cannot_see_the_future() -> None:
    model = _model("causal")
    inputs = torch.randn(1, CHANNELS, LENGTH)
    edited = inputs.clone()
    edited[..., LENGTH // 2 :] = torch.randn(1, CHANNELS, LENGTH - LENGTH // 2)
    with torch.no_grad():
        base = model.forward_dense(inputs)["event_logits"]
        after = model.forward_dense(edited)["event_logits"]
    cut = LENGTH // 2
    torch.testing.assert_close(base[..., :cut], after[..., :cut], rtol=1e-4, atol=1e-5)
    print("PASS test_causal_model_cannot_see_the_future")


def test_offline_model_is_expected_to_see_the_future() -> None:
    """Documents the difference: the offline model is deliberately non-causal."""

    model = _model("offline")
    inputs = torch.randn(1, CHANNELS, LENGTH)
    edited = inputs.clone()
    edited[..., LENGTH // 2 :] = torch.randn(1, CHANNELS, LENGTH - LENGTH // 2)
    with torch.no_grad():
        base = model.forward_dense(inputs)["event_logits"]
        after = model.forward_dense(edited)["event_logits"]
    difference = (base[..., : LENGTH // 2] - after[..., : LENGTH // 2]).abs().max().item()
    assert difference > 0, "the offline model should use future context"
    print(f"PASS test_offline_model_is_expected_to_see_the_future (delta={difference:.4f})")


def test_streaming_matches_whole_sequence() -> None:
    model = _model("causal")
    inputs = torch.randn(1, CHANNELS, 512)
    with torch.no_grad():
        whole = model.forward_dense(inputs, inference=True)["posture_logits"]
        model.reset_stream()
        chunks = [
            model.forward_dense(inputs[..., start : start + 64], inference=True)["posture_logits"]
            for start in range(0, 512, 64)
        ]
        streamed = torch.cat(chunks, dim=-1)
    torch.testing.assert_close(whole, streamed, rtol=1e-4, atol=1e-5)
    print("PASS test_streaming_matches_whole_sequence")


def main() -> int:
    print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    causal = _model("causal")
    offline = _model("offline")
    print(f"parameters: causal={parameter_count(causal)}, offline={parameter_count(offline)}")
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"FAIL {test.__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
