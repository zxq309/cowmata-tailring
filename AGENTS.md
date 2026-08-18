# COWMATA repository instructions (20260819)

## Non-negotiable data and evaluation rules

- Preserve raw nine-axis IMU continuously at 50 Hz. Windowing is a training parameter,
  never a raw-data format.
- Never let a window, chunk, filter or rolling statistic cross a `segment` boundary.
- Split train / validation / test by `cow_id`. Validation must be cow-disjoint from
  training, not merely session-disjoint.
- Report independent cows and independent events. Overlapping windows are not samples.
- Report every metric per cow as well as pooled, with a cow-level bootstrap interval.
- Do not report event precision or false-alarm rates without exhaustive
  `review_coverage`. The metric code enforces this; do not route around it.
- There is exactly one model-selection objective: `cowmata.metrics.selection_score`.

## Repository boundaries

- Never commit `supervised_cache/session_cache/`, `samples.csv`, `runs/`, or archives.
- Keep the 60-second demo session and the deployed GBDT working from a fresh clone.
- `weights/deploy/gbdt_full.joblib` is byte-identical to a verified artefact. Do not
  rewrite it; `cowmata/compat.py` keeps it loadable.
- Changing the feature bank requires bumping `features.FEATURE_VERSION` and recording
  the version in every bundle that consumes it.

Before merging: `python tests/test_contracts.py`, `python tests/test_torch_contracts.py`
on the GPU host, and the demo prediction in `README.md`.
