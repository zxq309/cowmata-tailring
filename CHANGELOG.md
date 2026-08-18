# Changelog

All notable changes to this repository are recorded here.

## [0.3.1] - 2026-08-18

- First real-data training run of the 20260819 baseline: MS-TCN++ 8-fold strict LOCO
  (cow-disjoint validation), 40.5 min on one RTX 3090, early stopping at 9–21 epochs
  per fold. Per-cow results, cow-level bootstrap intervals, and the evidence status
  live in the bilingual experiment report (`docs/EXPERIMENTS_20260819.md`,
  `docs/EXPERIMENTS_20260819.zh-CN.md`).
- Retrained the GBDT deploy candidate on the full 351,127-row × 120-feature (v2) table
  with cow-disjoint validation cows 23381-w1 and 23509-9; per-event thresholds are
  written into the bundle. The artifact stays local under `runs/gbdt_full/` and is not
  promoted to `weights/deploy/` until review evidence exists.
- README gained an Experiment results section in English and Simplified Chinese.

## [0.3.0] - 2026-08-19

### Architecture
- Replaced the single-stage windowed TCN with `MultiTaskMSTCN`: strided frame stem,
  MS-TCN++ dual-dilated prediction generation, three refinement stages, and an
  ASRF-style boundary head. Every stage is supervised; T-MSE smoothing is in the loss.
- Replaced per-label sliding windows with `DenseSegmentDataset`, which supervises a
  contiguous chunk at every masked step.
- Removed the `pytorch-tcn` dependency; chunk-with-overlap inference is exact.

### Data contract
- Cache schema 2: int16 counts, 9 channels, sparse quality intervals, no stored
  timestamps. 52 → 18 bytes per frame. Schema 1 is read transparently.
- New per-session metadata field `tail_position`.

### Labels
- Added `MOUNTING` and `MOUNTED_BY`.
- Deprecated `TAIL_WAGGING` (readable, never trained, never reported).
- `FEEDING` remains readable and folded into `UPRIGHT`.

### Features
- `FEATURE_VERSION` gate. Version 1 reproduces the deployed 104-column list by name and
  order; version 2 adds amplitude self-calibration and two rotation invariants (120).

### Metrics
- Five-layer suite: frame, segment (`F1@tIoU`, edit score), deployment, generalisation
  (per-cow + cow-level bootstrap), day level (Se/Sp/PPV/lead time).
- One model-selection objective across the whole repository.

### Post-processing
- Hysteresis thresholding and boundary snapping in `postprocess.assemble_intervals`.
- Per-event thresholds are written into the bundle and honoured at inference.

### Pipelines
- The three batch scripts (`build_supervised_cache`, `build_feature_table`,
  `train_full_gbdt`) became `cowmata/pipelines.py`; `scripts/` keeps four-line shims.
- `build_cache` writes the dense label frame alongside `samples.csv`, so the deep and
  GBDT branches share one label derivation instead of two.
- `train_gbdt` selects each per-event threshold on a **cow-disjoint** validation split
  and writes it, with `feature_version`, into the joblib bundle. An event with no
  validation evidence keeps 0.5 and is reported as `default_no_validation_evidence`.
- Dense prediction files now carry `cache_key`, `cow_id`, `device_mac`, `session_id`,
  so mining and the day-level layer can group by animal without a manual rejoin.

### Runtime
- Torch is imported lazily and is no longer a hard dependency; install `cowmata[deep]`
  on the training host. `check-env` reports which subcommands work without it.

### Structure
- `cattle_imu` + `cowmata` + `scripts` → `cowmata` + thin shims; `compat.py` keeps the
  deployed joblib loadable.
- `fusion.py` moved to `experiments/`; `train_event_loco.py` and
  `predict_continuous.py` removed.

## [0.2.0] - 2026-08-18

- Rebuilt the GitHub-facing documentation in English.
- Expanded the README with product context, architecture, model inventory, data contracts, training/evaluation rules, verification evidence, roadmap, and team governance.
- Added official COWMATA logo and tail-sensor product assets from the company website.
- Added an original multimodal AI pipeline hero illustration.
- Added a curated external-project watchlist for time-series modeling, temporal segmentation, wearable representation learning, anomaly detection, and MLOps.
- Added the official company logo from the Chinese website and credited four named contributors with institutional affiliations.
- Numbered the external reference radar and introduced a concise README changelog section.
- Converted generated diagnostic reports and inference display labels to English.
- Removed the Chinese DOCX design artifact from GitHub while retaining it locally.
- Added a Simplified Chinese README with a language switcher and a company contact link.
- Trimmed verbose README statements and used Chinese author names in the Chinese README.
- Moved Quick start before the product details and added citation and contributing sections.
- Added English and Simplified Chinese overall-framework diagrams to the system-architecture section.

## [0.1.0] - 2026-08-18

- Created the clean COWMATA tail-ring algorithm baseline.
- Added a unified Python API and `cowmata` CLI.
- Retained the validated GBDT deploy weight and TCN development checkpoint.
- Added cow-level split contracts, tests, and a 60-second executable demo session.
- Removed the unused SSL cache, index, and disconnected SSL/prototype code path.
- Kept the full supervised cache external to Git while preserving its training interface.
