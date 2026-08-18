"""Traceable annotation ingestion and code-level conflict adjudication."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


# These are source annotation codes, not mutually exclusive model outputs.
# FEEDING remains readable so historical labels are never discarded, but it is
# mapped to UPRIGHT and is no longer trained or emitted as a prediction target.
BODY_ANNOTATION_CODES = ("STANDING", "LYING", "WALKING", "FEEDING")
BODY_CODES = BODY_ANNOTATION_CODES  # Backward-compatible cache schema name.
POSTURE_CODES = ("UPRIGHT", "LYING")
LOCOMOTION_CODE = "WALKING"
EVENT_CODES = (
    "STANDING_UP",
    "LYING_DOWN",
    "URINATION",
    "DEFECATION",
    "TAIL_RAISED",
    "TAIL_WAGGING",
)
ALL_CODES = BODY_CODES + EVENT_CODES


def normalize_cow_id(value: object, device_key: str) -> tuple[str, str]:
    """Normalize punctuation, falling back to the traceable device suffix."""

    text = "" if pd.isna(value) else str(value).strip()
    text = re.sub(r"[_\s]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    if text:
        return text, "csv"
    pieces = str(device_key).split("-", 1)
    fallback = pieces[1].strip() if len(pieces) == 2 else ""
    return fallback, "device_directory" if fallback else "missing"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_annotation_sources(roots: Iterable[str | Path]) -> pd.DataFrame:
    """Load flat event exports and retain provenance for every row."""

    frames: list[pd.DataFrame] = []
    for root_value in roots:
        root = Path(root_value)
        for path in sorted(root.rglob("*.events.csv")):
            frame = pd.read_csv(path, encoding="utf-8-sig")
            meta_path = path.with_name(path.name.replace(".events.csv", ".events_meta.json"))
            metadata: dict[str, object] = {}
            if meta_path.exists():
                with meta_path.open("r", encoding="utf-8-sig") as handle:
                    metadata = json.load(handle)
            device_key = path.parent.name
            device_mac = device_key.split("-", 1)[0]
            source_hash = _sha256(path)
            frame["source_path"] = str(path)
            frame["source_sha256"] = source_hash
            frame["source_root"] = str(root)
            frame["source_group"] = path.relative_to(root).parts[0]
            frame["device_key"] = device_key
            frame["device_mac"] = device_mac
            frame["meta_annotator"] = str(metadata.get("annotator", ""))
            frame["meta_cow_id"] = str(metadata.get("cow_id", ""))
            frame["exported_bj"] = str(metadata.get("exported_bj", ""))
            frame["video_name"] = str(metadata.get("video_name", ""))
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("no *.events.csv files found under annotation roots")
    result = pd.concat(frames, ignore_index=True)
    for name in ("annotator", "cow_id", "code", "layer", "session_id"):
        result[name] = result[name].fillna("").astype(str).str.strip()
    normalized = [normalize_cow_id(value, device) for value, device in zip(result["cow_id"], result["device_key"])]
    result["cow_id_original"] = result["cow_id"]
    result["cow_id"] = [item[0] for item in normalized]
    result["cow_id_source"] = [item[1] for item in normalized]
    result["annotator"] = result["annotator"].replace("", "unknown")
    result["t_start_rel_ms"] = pd.to_numeric(result["t_start_rel_ms"], errors="raise").astype(float)
    result["t_end_rel_ms"] = pd.to_numeric(result["t_end_rel_ms"], errors="raise").astype(float)
    result["duration_ms"] = result["t_end_rel_ms"] - result["t_start_rel_ms"]
    if (result["duration_ms"] <= 0).any():
        bad = result.loc[result["duration_ms"] <= 0, ["source_path", "index", "duration_ms"]]
        raise ValueError(f"non-positive annotation interval(s): {bad.to_dict('records')}")
    unknown_codes = sorted(set(result["code"]) - set(ALL_CODES))
    if unknown_codes:
        raise ValueError(f"unsupported annotation codes: {unknown_codes}")
    return result


def adjudicate_annotations(
    source: pd.DataFrame,
    *,
    preferred_annotators: tuple[str, ...] = ("zyc",),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose one annotator per session+code, never replacing a whole session."""

    selected_frames: list[pd.DataFrame] = []
    conflict_rows: list[dict[str, object]] = []
    group_keys = ["device_mac", "session_id", "code"]
    for key, group in source.groupby(group_keys, sort=True, dropna=False):
        annotators = tuple(sorted(set(group["annotator"])))
        chosen = next((name for name in preferred_annotators if name in annotators), None)
        if chosen is None:
            if len(annotators) == 1:
                chosen = annotators[0]
            else:
                # No configured reviewer is present. Choosing the newest export
                # is deterministic and fully recorded rather than double-counting.
                newest = group.sort_values(["exported_bj", "source_path"], kind="stable").iloc[-1]
                chosen = str(newest["annotator"])
        keep = group[group["annotator"] == chosen].copy()
        keep["adjudication"] = (
            "preferred_annotator" if len(annotators) > 1 and chosen in preferred_annotators else "single_annotator"
        )
        selected_frames.append(keep)
        for row in group[group["annotator"] != chosen].itertuples(index=False):
            conflict_rows.append(
                {
                    "device_mac": key[0],
                    "session_id": key[1],
                    "code": key[2],
                    "selected_annotator": chosen,
                    "dropped_annotator": row.annotator,
                    "dropped_start_ms": row.t_start_rel_ms,
                    "dropped_end_ms": row.t_end_rel_ms,
                    "dropped_source_path": row.source_path,
                    "reason": "preferred annotator selected for this session and label only",
                }
            )
    selected = pd.concat(selected_frames, ignore_index=True)
    selected = selected.sort_values(
        ["device_mac", "session_id", "t_start_rel_ms", "t_end_rel_ms", "code"],
        kind="stable",
    ).reset_index(drop=True)
    selected.insert(0, "event_id", [f"EVT-{index:05d}" for index in range(1, len(selected) + 1)])
    conflicts = pd.DataFrame(conflict_rows)
    return selected, conflicts


def find_body_overlaps(events: pd.DataFrame) -> pd.DataFrame:
    """Report mutually exclusive body-state intervals that overlap in time."""

    rows: list[dict[str, object]] = []
    body = events[events["code"].isin(BODY_CODES)]
    for (device, session), group in body.groupby(["device_mac", "session_id"], sort=True):
        records = group.sort_values("t_start_rel_ms").to_dict("records")
        for index, left in enumerate(records):
            for right in records[index + 1 :]:
                if float(right["t_start_rel_ms"]) >= float(left["t_end_rel_ms"]):
                    break
                overlap = min(float(left["t_end_rel_ms"]), float(right["t_end_rel_ms"])) - max(
                    float(left["t_start_rel_ms"]), float(right["t_start_rel_ms"])
                )
                if overlap > 0 and left["code"] != right["code"]:
                    rows.append(
                        {
                            "device_mac": device,
                            "session_id": session,
                            "left_event_id": left["event_id"],
                            "right_event_id": right["event_id"],
                            "left_code": left["code"],
                            "right_code": right["code"],
                            "overlap_ms": overlap,
                        }
                    )
    return pd.DataFrame(rows)
