"""Stream cached continuous IMU through a trained causal model.

Outputs dense 2 Hz probabilities and a compact candidate-review queue whose
relative timestamps can be opened directly in the synchronized video.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cattle_imu.amp import autocast_context, resolve_cuda_precision
from cattle_imu.annotations import EVENT_CODES, POSTURE_CODES
from cattle_imu.model import CausalMultiTaskTCN


EVENT_POSTPROCESS = {
    "STANDING_UP": {"min_ms": 500, "merge_gap_ms": 1500},
    "LYING_DOWN": {"min_ms": 500, "merge_gap_ms": 1500},
    "URINATION": {"min_ms": 1500, "merge_gap_ms": 3000},
    "DEFECATION": {"min_ms": 1500, "merge_gap_ms": 3000},
    "TAIL_RAISED": {"min_ms": 1000, "merge_gap_ms": 2000},
    "TAIL_WAGGING": {"min_ms": 500, "merge_gap_ms": 1000},
}


def frame_times_ms(indices: np.ndarray, segments: list[dict[str, object]]) -> np.ndarray:
    output = np.full(indices.shape, -1, dtype=np.int64)
    for segment in segments:
        start_index = int(segment["start_index"])
        stop_index = int(segment["stop_index"])
        selected = (indices >= start_index) & (indices < stop_index)
        if not np.any(selected):
            continue
        start_ms = float(segment["start_ms"])
        stop_ms = float(segment["stop_ms"])
        scale = (stop_ms - start_ms) / max(stop_index - start_index, 1)
        output[selected] = np.rint(start_ms + (indices[selected] - start_index) * scale).astype(np.int64)
    if np.any(output < 0):
        raise ValueError("catalog segments do not cover every requested output frame")
    return output


def candidate_intervals(
    frame: pd.DataFrame,
    code: str,
    threshold: float,
    metadata: dict[str, object],
    checkpoint_name: str,
) -> list[dict[str, object]]:
    config = EVENT_POSTPROCESS[code]
    times = frame["time_rel_ms"].to_numpy(np.int64)
    scores = frame[f"prob_{code}"].to_numpy(float)
    selected = np.flatnonzero(scores >= threshold)
    if selected.size == 0:
        return []
    groups: list[np.ndarray] = []
    start = 0
    for position in range(1, len(selected)):
        if times[selected[position]] - times[selected[position - 1]] > config["merge_gap_ms"]:
            groups.append(selected[start:position])
            start = position
    groups.append(selected[start:])
    rows: list[dict[str, object]] = []
    for values in groups:
        start_ms = int(times[values[0]])
        end_ms = int(times[values[-1]] + 500)
        if end_ms - start_ms < config["min_ms"]:
            continue
        candidate_scores = scores[values]
        rows.append(
            {
                "candidate_id": f"{metadata['device_mac']}|{metadata['session_id']}|{code}|{start_ms}",
                "device_key": metadata["device_key"],
                "device_mac": metadata["device_mac"],
                "cow_id": metadata["cow_id"],
                "session_id": metadata["session_id"],
                "event_code": code,
                "t_start_rel_ms": start_ms,
                "t_end_rel_ms": end_ms,
                "duration_ms": end_ms - start_ms,
                "max_probability": float(candidate_scores.max()),
                "mean_probability": float(candidate_scores.mean()),
                "threshold": float(threshold),
                "source_model": checkpoint_name,
                "review_status": "PENDING",
                "reviewed_code": "",
                "reviewed_start_rel_ms": "",
                "reviewed_end_rel_ms": "",
                "reviewer": "",
                "review_notes": "",
            }
        )
    return rows


@torch.inference_mode()
def predict_session(
    model: CausalMultiTaskTCN,
    features_path: Path,
    metadata: dict[str, object],
    feature_indices: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    policy: object,
    block_samples: int,
    output_stride: int,
) -> pd.DataFrame:
    values = np.load(features_path, mmap_mode="r")
    model.reset_stream()
    rows: list[pd.DataFrame] = []
    for start in range(0, len(values), block_samples):
        stop = min(start + block_samples, len(values))
        raw = np.asarray(values[start:stop], dtype=np.float32)
        normalized = ((raw - mean) / std)[:, feature_indices]
        inputs = torch.from_numpy(normalized.T.copy()).unsqueeze(0).to(device)
        with autocast_context(policy):
            output = model.forward_dense(inputs, inference=True)
        local = np.arange(0, stop - start, output_stride, dtype=np.int64)
        local_tensor = torch.as_tensor(local, device=device, dtype=torch.long)
        absolute = start + local
        data: dict[str, object] = {"frame_index": absolute}
        posture = torch.softmax(output["posture_logits"], dim=1)[0].index_select(1, local_tensor).float().cpu().numpy().T
        walking = torch.sigmoid(output["locomotion_logits"])[0, 0].index_select(0, local_tensor).float().cpu().numpy()
        events = torch.sigmoid(output["event_logits"])[0].index_select(1, local_tensor).float().cpu().numpy().T
        for index, code in enumerate(POSTURE_CODES):
            data[f"prob_posture_{code}"] = posture[:, index]
        data["prob_WALKING"] = walking
        for index, code in enumerate(model.event_codes):
            data[f"prob_{code}"] = events[:, index]
        rows.append(pd.DataFrame(data))
    result = pd.concat(rows, ignore_index=True)
    result.insert(0, "time_rel_ms", frame_times_ms(result["frame_index"].to_numpy(np.int64), metadata["segments"]))
    result.insert(0, "session_id", str(metadata["session_id"]))
    result.insert(0, "cow_id", str(metadata["cow_id"]))
    result.insert(0, "device_mac", str(metadata["device_mac"]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--session", action="append", help="device_mac|session_id; repeat as needed")
    parser.add_argument("--block-samples", type=int, default=500)
    parser.add_argument("--output-stride", type=int, default=25)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    args = parser.parse_args()
    policy = resolve_cuda_precision(args.precision)
    device = torch.device("cuda:0")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("model_class") != "CausalMultiTaskTCN":
        raise ValueError("predict_continuous.py requires a CausalMultiTaskTCN checkpoint")
    model = CausalMultiTaskTCN(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with args.catalog.open("r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    requested = set(args.session or [])
    sessions = [
        item for item in catalog["sessions"]
        if not requested or f"{item['device_mac']}|{item['session_id']}" in requested
    ]
    if requested and len(sessions) != len(requested):
        found = {f"{item['device_mac']}|{item['session_id']}" for item in sessions}
        raise ValueError(f"catalog sessions not found: {sorted(requested - found)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    indices = np.asarray(checkpoint["feature_indices"], dtype=np.int64)
    statistics = checkpoint["feature_statistics"]
    mean = np.asarray(statistics["mean"], dtype=np.float32)
    std = np.maximum(np.asarray(statistics["std"], dtype=np.float32), 1e-6)
    thresholds = {code: float(checkpoint.get("thresholds", {}).get(code, 0.5)) for code in EVENT_CODES}
    candidates: list[dict[str, object]] = []
    for item in sessions:
        cache_path = args.root / str(item["cache_path"]).replace("\\", "/") / "features.npy"
        dense = predict_session(
            model, cache_path, item, indices, mean, std, device, policy,
            args.block_samples, args.output_stride,
        )
        safe_name = f"{item['device_mac']}_{str(item['session_id']).replace(':', '_').replace(' ', '_')}"
        dense.to_csv(args.output_root / f"{safe_name}_dense.csv", index=False, encoding="utf-8-sig")
        for code in model.event_codes:
            candidates.extend(candidate_intervals(dense, code, thresholds[code], item, args.checkpoint.name))
    queue = pd.DataFrame(candidates)
    queue.to_csv(args.output_root / "candidate_review_queue.csv", index=False, encoding="utf-8-sig")
    summary = {
        "status": "complete",
        "sessions": len(sessions),
        "candidates": len(queue),
        "output_root": str(args.output_root),
        "precision": policy.name,
        "important": "Only manually confirmed rows should enter gold labels; preserve rejected rows as hard negatives.",
    }
    with (args.output_root / "prediction_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
