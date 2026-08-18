# -*- coding: utf-8 -*-
"""Model contracts that need torch. Run these on the training host.

    python tests/test_torch_contracts.py
    pytest tests/test_torch_contracts.py

These catch the failure mode "training metrics look fine but production is
wrong".  The three that matter most:

* **causality** - the causal model must not change an output when a *later*
  sample changes.  Get this wrong and the online alerting model is validated
  with information it will never have in the barn.
* **chunk equivalence** - dense inference chunks a long segment with overlap.
  If the overlap is smaller than the receptive field the chunked result differs
  from the whole-segment result, silently and only on long sessions.
* **padding isolation** - a short chunk padded into a batch must not influence
  its neighbour through the dilated convolutions.

Every test skips itself cleanly when torch is not installed, so this file is
safe to keep in CI on a machine without a deep-learning stack.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch

    HAVE_TORCH = True
except Exception:  # pragma: no cover - environment dependent
    HAVE_TORCH = False

CHANNELS = 9
STRIDE = 25
EVENTS = ("STANDING_UP", "URINATION", "MOUNTED_BY")


def _skip(name: str) -> bool:
    if not HAVE_TORCH:
        print(f"  [skip] {name}: torch not installed")
        return True
    return False


def _model(mode: str, steps_layers: int = 4, stages: int = 1):
    from cowmata.models import build_model

    torch.manual_seed(0)
    model = build_model(
        mode,
        in_channels=CHANNELS,
        event_codes=list(EVENTS),
        channels=16,
        stage_layers=steps_layers,
        refinement_stages=stages,
        dropout=0.0,
    )
    model.eval()
    return model


def test_output_rate_is_the_decision_rate() -> None:
    if _skip("test_output_rate_is_the_decision_rate"):
        return
    model = _model("offline")
    steps = 200
    inputs = torch.randn(2, CHANNELS, steps * STRIDE)
    out = model(inputs)
    assert out["event_logits"][-1].shape == (2, len(EVENTS), steps)
    assert out["posture_logits"][-1].shape == (2, 2, steps)
    assert out["boundary_logits"][-1].shape == (2, 1, steps)
    print("PASS test_output_rate_is_the_decision_rate")


def test_every_stage_is_supervised_and_shapes_agree() -> None:
    if _skip("test_every_stage_is_supervised_and_shapes_agree"):
        return
    model = _model("offline", stages=3)
    out = model(torch.randn(1, CHANNELS, 100 * STRIDE))
    assert len(out["event_logits"]) == 4, "one generation stage plus three refinements"
    assert all(item.shape == out["event_logits"][0].shape for item in out["event_logits"])
    print("PASS test_every_stage_is_supervised_and_shapes_agree")


def test_causal_model_cannot_see_the_future() -> None:
    if _skip("test_causal_model_cannot_see_the_future"):
        return
    model = _model("causal")
    steps = 200
    inputs = torch.randn(1, CHANNELS, steps * STRIDE)
    edited = inputs.clone()
    cut = steps // 2
    edited[..., cut * STRIDE :] = torch.randn(1, CHANNELS, (steps - cut) * STRIDE)
    with torch.no_grad():
        base = model(inputs)["event_logits"][-1]
        after = model(edited)["event_logits"][-1]
    torch.testing.assert_close(base[..., :cut], after[..., :cut], rtol=1e-4, atol=1e-5)
    print("PASS test_causal_model_cannot_see_the_future")


def test_offline_model_is_deliberately_not_causal() -> None:
    if _skip("test_offline_model_is_deliberately_not_causal"):
        return
    model = _model("offline")
    steps = 200
    inputs = torch.randn(1, CHANNELS, steps * STRIDE)
    edited = inputs.clone()
    cut = steps // 2
    edited[..., cut * STRIDE :] = torch.randn(1, CHANNELS, (steps - cut) * STRIDE)
    with torch.no_grad():
        base = model(inputs)["event_logits"][-1]
        after = model(edited)["event_logits"][-1]
    delta = (base[..., :cut] - after[..., :cut]).abs().max().item()
    assert delta > 0, "the offline model is supposed to use future context"
    print(f"PASS test_offline_model_is_deliberately_not_causal (delta={delta:.4f})")


def test_chunked_inference_matches_the_whole_segment() -> None:
    if _skip("test_chunked_inference_matches_the_whole_segment"):
        return
    model = _model("offline", steps_layers=3, stages=1)
    steps = 400
    overlap = max(1, model.receptive_field_steps)
    inputs = torch.randn(1, CHANNELS, steps * STRIDE)
    with torch.no_grad():
        whole = model.predict(inputs)["events"]
        pieces = []
        position = 0
        block = 120
        while position < steps:
            stop = min(position + block, steps)
            left = min(overlap, position)
            right = min(overlap, steps - stop)
            window = inputs[..., (position - left) * STRIDE : (stop + right) * STRIDE]
            out = model.predict(window)["events"]
            pieces.append(out[..., left : left + (stop - position)])
            position = stop
        chunked = torch.cat(pieces, dim=-1)
    torch.testing.assert_close(whole, chunked, rtol=1e-4, atol=1e-5)
    print("PASS test_chunked_inference_matches_the_whole_segment")


def test_padding_does_not_leak_between_batch_members() -> None:
    if _skip("test_padding_does_not_leak_between_batch_members"):
        return
    model = _model("offline")
    steps = 120
    short = 60
    inputs = torch.zeros(2, CHANNELS, steps * STRIDE)
    inputs[0, :, : short * STRIDE] = torch.randn(CHANNELS, short * STRIDE)
    inputs[1] = torch.randn(CHANNELS, steps * STRIDE)
    mask = torch.zeros(2, 1, steps)
    mask[0, :, :short] = 1.0
    mask[1] = 1.0
    with torch.no_grad():
        batched = model(inputs, mask=mask)["event_logits"][-1]
        alone = model(inputs[1:2], mask=mask[1:2])["event_logits"][-1]
    torch.testing.assert_close(batched[1], alone[0], rtol=1e-4, atol=1e-5)
    print("PASS test_padding_does_not_leak_between_batch_members")


def test_multi_stage_loss_is_finite_and_differentiable() -> None:
    if _skip("test_multi_stage_loss_is_finite_and_differentiable"):
        return
    from cowmata.models import multi_stage_loss

    model = _model("offline")
    model.train()
    steps = 80
    inputs = torch.randn(2, CHANNELS, steps * STRIDE)
    targets = {
        "posture_target": torch.randint(0, 2, (2, steps)),
        "posture_mask": torch.ones(2, steps),
        "locomotion_target": torch.randint(0, 2, (2, steps)).float(),
        "locomotion_mask": torch.ones(2, steps),
        "event_target": torch.randint(0, 2, (2, len(EVENTS), steps)).float(),
        "event_mask": torch.ones(2, len(EVENTS), steps),
        "boundary_target": torch.randint(0, 2, (2, steps)).float(),
        "boundary_mask": torch.ones(2, steps),
    }
    parts = multi_stage_loss(model(inputs), targets)
    assert torch.isfinite(parts["total"])
    parts["total"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    print("PASS test_multi_stage_loss_is_finite_and_differentiable")


def test_dense_dataset_never_crosses_a_segment() -> None:
    if _skip("test_dense_dataset_never_crosses_a_segment"):
        return
    import pandas as pd

    from cowmata.cache import Calibration, Segment, write_cache_v2
    from cowmata.dataset import DenseSegmentDataset
    from cowmata.labels import EVENT_CODES

    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        frames = 200 * STRIDE
        counts = np.zeros((frames, 9), dtype=np.int16)
        counts[:, 0] = np.arange(frames, dtype=np.int16) % 1000
        segments = [
            Segment(0, 0, 100 * STRIDE, 0, (100 * STRIDE - 1) * 20),
            Segment(1, 100 * STRIDE, frames, 7_200_000, 7_200_000 + (100 * STRIDE - 1) * 20),
        ]
        write_cache_v2(
            root / "s",
            counts=counts,
            segments=segments,
            calibration=Calibration(4096.0),
            metadata={"cache_key": "s", "cow_id": "c"},
        )
        rows = []
        for segment in segments:
            for local in range(0, segment.length, STRIDE):
                row = {
                    "cache_key": "s",
                    "cow_id": "c",
                    "device_mac": "A",
                    "session_id": "S",
                    "center_index": segment.start_index + local,
                    "center_time_ms": segment.start_ms + local * 20,
                    "body_target": 0,
                    "segment_start_index": segment.start_index,
                    "segment_stop_index": segment.stop_index,
                }
                for code in EVENT_CODES:
                    row[f"event_{code}"] = 0
                    row[f"mask_{code}"] = 1
                rows.append(row)
        dataset = DenseSegmentDataset(
            pd.DataFrame(rows),
            root,
            np.zeros(9),
            np.ones(9),
            chunk_steps=80,
            chunk_overlap=10,
        )
        for row_start, row_stop in dataset.chunks:
            starts = dataset.segment_start[row_start:row_stop]
            assert len(set(starts.tolist())) == 1, "a chunk spanned two segments"
        print(f"PASS test_dense_dataset_never_crosses_a_segment ({len(dataset)} chunks)")


def main() -> int:
    if not HAVE_TORCH:
        print("torch is not installed; every model contract was skipped.")
        print("Run this file on the training host before trusting any deep-model number.")
        return 0
    print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    model = _model("offline")
    print(
        f"receptive field: {model.receptive_field_steps} steps "
        f"= {model.receptive_field_seconds:.1f} s"
    )
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"FAIL {test.__name__}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
