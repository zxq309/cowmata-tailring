# 20260818 Baseline Verification Record

Verification date: 2026-08-18. The target is the clean `20260818` repository; the source comparison is the `20260816` algorithm version.

## Verified results

- Editable installation succeeded and all 24 retained contract tests passed after removal of the disconnected SSL branch.
- Full supervised-data preflight read 132 cached sessions covering 131.5739 hours with no shape or non-finite-value errors.
- 344,287 supervised center points and six strict leave-one-cow-out folds were checked for session overlap.
- The retained TCN checkpoint declares `ssl_transfer=null`; 2.6527 GB of `ssl_cache`, a 19.897 MB `ssl_index`, and eight disconnected SSL/prototype files were therefore removed.
- The deploy GBDT artifact is SHA-256 identical to its source artifact.
- One real cached session produced 7,159 dense 2 Hz predictions and three event candidates.
- Old and refactored inference produced identical `center_index`/`center_time_ms` values; the maximum probability difference across eight tasks was `1.1102230246251565e-16`.
- The offline TCN development checkpoint ran four real-cache windows: posture shape `[4, 2]`, walking shape `[4]`, all outputs finite, and posture probabilities summing to one.
- The bundled `demo_session_60s` produced 120 dense 2 Hz points and completed both GBDT and TCN inference paths.
- CPU forward, backward, and optimizer-update smoke tests passed.
- GitHub Actions reproduced dependency installation, all tests, and demo inference on a clean Ubuntu runner.

The initial algorithm package contained approximately 16.11 MB of tracked files, with an 8.55 MB maximum file. Brand/product images and the visual hero added in `v0.2.0` remain well below GitHub's ordinary file limit.

## Event coverage in the reviewed baseline

| Event | Eligible intervals | Cows |
|---|---:|---:|
| `STANDING_UP` | 71 | 5 |
| `LYING_DOWN` | 60 | 4 |
| `URINATION` | 87 | 5 |
| `DEFECATION` | 15 | 4 |
| `TAIL_RAISED` | 26 | 4 |
| `TAIL_WAGGING` | 4 | 1 |

## Limitations that must remain visible

- Only four of six LOCO folds contain at least 1,000 test points; small folds must be interpreted in aggregate.
- Complete exhaustively reviewed ground-truth time is not yet available, so event Precision and false alarms per cow per 24 hours are not yet supported as acceptance claims.
- The Joblib package emits an older-XGBoost serialization warning. It loaded and predicted successfully with XGBoost 3.4.1, and numerical parity with the previous entry point was verified.
- The local machine exposes an NVIDIA GeForce RTX 2080 Ti, but the isolated verification environment used CPU PyTorch. CUDA training success is not claimed by this record.
- Website product statements and this repository's algorithm-verification evidence are separate scopes.
