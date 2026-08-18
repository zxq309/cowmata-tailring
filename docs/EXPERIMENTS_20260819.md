# 20260819 Experiment Report — MS-TCN++ LOCO Training and GBDT Retraining

Run date: 2026-08-18 · [简体中文版](EXPERIMENTS_20260819.zh-CN.md)

This is the first real-data training run of the 20260819 baseline. It has two parts:
an 8-fold strict leave-one-cow-out (LOCO) run of the deep model, and a compliant
retrain of the deployable GBDT candidate. Read it together with
[`VERIFICATION_20260819.md`](VERIFICATION_20260819.md) (code verification) and
[`METRICS.md`](METRICS.md) (evidence gating).

| Item | Value |
|---|---|
| Environment | RTX 3090, torch 2.12.1+cu130, bf16 autocast, conda env `cattle_imu_gpu` |
| Deep model | `MultiTaskMSTCN` (MS-TCN++): channels 64, 8 stage layers, 3 refinement stages, ASRF-style boundary head |
| Deep config | 40 epochs max, patience 8, batch 4, chunk 1200 steps, lr 5e-4 (ReduceLROnPlateau), offline mode |
| Splits | `loco_splits/loco_splits.json`, strict LOCO, cow-disjoint validation |
| Deep wall time | 40.5 min (18:42–19:22), 9–21 epochs per fold |
| GBDT | full feature table 351,127 × 120 (v2, offline, calibrated), xgboost GPU, ~5 min |

## 1. Deep model — 8-fold strict LOCO

Each fold holds out one cow for testing; validation is also cow-disjoint from
training. Early stopping (patience 8) ended every fold well before the 40-epoch cap.

| fold | test cow | val cow | best epoch | epochs | val score | test score | test points |
|---|---|---|---|---|---|---|---|
| 1 | 23381-w1 | 20201-3 | 3 | 11 | 0.6834 | 0.5262 | 43,200 |
| 2 | 23509-9 | 20201-3 | 2 | 10 | 0.7819 | 0.5533 | 396,001 |
| 3 | 21074-1 | 20201-3 | 7 | 15 | 0.5210 | 0.3524 | 302,401 |
| 4 | 20201-3 | 21074-1 | 4 | 12 | 0.3308 | 0.4609 | 165,591 |
| 5 | 24178-11 | 20201-3 | 1 | 9 | 0.7882 | 0.5078 | 144,000 |
| 6 | 21100-10 | 20201-3 | 2 | 10 | 0.8272 | 0.6364 | 72,000 |
| 7 | 23335-7 | 20201-3 | 12 | 20 | 0.7878 | 0.6274 | 288,000 |
| 8 | 23489-8 | 20201-3 | 13 | 21 | 0.7057 | 0.6423 | 36,000 |

Pooled test selection score: **mean 0.5383, 95% cow-level bootstrap CI [0.4702, 0.5973]**
(n = 8 cows). The spread between cows is large (0.35–0.64) and the two data-heavy
animals, 21074-1 and 20201-3, are exactly the hardest test cows — they dominate every
other fold's training set, so a model trained without them is a model trained on far
less data.

### Test event AP per cow (point level, labelled stretches only)

| test cow | STANDING_UP | LYING_DOWN | URINATION | DEFECATION | TAIL_RAISED |
|---|---|---|---|---|---|
| 23381-w1 | 0.685 | 0.436 | 0.543 | 0.529 | 0.766 |
| 23509-9 | 0.338 | 0.203 | 0.290 | 0.017 | 0.303 |
| 21074-1 | 0.113 | 0.071 | 0.056 | 0.010 | 0.072 |
| 20201-3 | 0.101 | 0.099 | 0.045 | 0.054 | 0.157 |
| 24178-11 | 0.474 | 0.485 | 0.467 | 0.574 | 0.636 |
| 21100-10 | 0.518 | 0.525 | 0.527 | 0.088 | 0.591 |
| 23335-7 | 0.932 | 0.855 | 0.978 | 0.980 | 0.986 |
| 23489-8 | 1.000 | 1.000 | 1.000 | 1.000 | 0.990 |

### Per-cow ranking (official selection_score) and the best cow

Each cow is scored by the model trained without it, ranked by the official overall
score:

| rank | test cow | selection_score | posture accuracy | WALKING AP | 5-event AP |
|---|---|---|---|---|---|
| 1 | 23489-8 | 0.642 | 1.0* | n/e | 0.99–1.00 |
| 2 | 21100-10 | 0.636 | 1.0* | 0.988 | 0.09–0.59 |
| 3 | **23335-7** | **0.627** | 1.0* | n/e | **0.86–0.99** |
| 4 | 23509-9 | 0.553 | 0.843 | 0.928 | 0.02–0.34 |
| 5 | 23381-w1 | 0.526 | 1.0* | n/e | 0.44–0.77 |
| 6 | 24178-11 | 0.508 | 1.0* | n/e | 0.47–0.64 |
| 7 | 20201-3 | 0.461 | 0.650 | 0.819 | 0.05–0.16 |
| 8 | 21074-1 | 0.352 | 0.566 | 0.529 | 0.01–0.11 |

`*` = degenerate (single posture class in the test set), `n/e` = not evaluable. The
nominal winner 23489-8 rests on 98 posture points (one class) and 13 events (1–4 per
class), so its AP of 1.00 is statistically fragile. The most credible best result is
**23335-7** — full 9-task detail:

| task | 23335-7 detail |
|---|---|
| Posture (standing/lying) | test set carries a single class (780 points); accuracy 1.0 is degenerate |
| WALKING | n/e — no walking positives in this test set |
| STANDING_UP | 22 true / 21 pred, recall 0.909, **F1@25 0.930**, edit 87.2, onset median 302 ms |
| LYING_DOWN | 19 / 15, recall 0.684, F1@25 0.765, edit 70.0, onset median 368 ms |
| URINATION | 16 / 18, recall 1.000, **F1@25 0.941**, edit 70.4, onset median 296 ms |
| DEFECATION | 21 / 23, recall 1.000, F1@25 0.818, edit 62.2, onset median 3.9 s |
| TAIL_RAISED | 4 / 42, recall 0.500, F1@25 0.087 — the one weak spot, ~10× over-reporting |
| MOUNTING / MOUNTED_BY | not_evaluable — zero annotated positives in the whole dataset |

On this cow the model nearly "reports one, hits one" for the three main events
(F1@25 0.82–0.94, onset error 0.3–0.4 s; defecation onset 3.9 s is the outlier).

Three reservations. (1) 23489-8's rank-1 score is inflated as described above.
(2) 23335-7's own posture/WALKING rows are degenerate or missing, so only the five
event tasks are really evaluated on this cow. (3) Every number here is still a
diagnostic on labelled stretches: each row is the score of a model that never saw that
cow, and claimable precision / false-alarm rates still await `review_coverage`.
23335-7 shows the model is already strong on clean-data cows; the weakness
concentrates on the dirty, distribution-shifted big-data animals.

### Pooled evaluation over all 8 test cows (official metric functions)

All 8 test-cow prediction files merged into one pool (~334k evaluation points, ~47
annotated hours), scored with the official metric functions using one threshold per
event — the median of that event's fold thresholds. There is deliberately no single
"event accuracy" in this report: events are sparse and predicting nothing scores >99%
point-wise, so the metric contract forbids quoting one. The posture head does carry a
real frame-level accuracy.

| task head | overall metric | value | events: true / predicted / hit |
|---|---|---|---|
| Posture (standing/lying) | accuracy (MoF) · macro-F1 | **0.596** · 0.374 | — |
| WALKING | average precision | **0.532** | — |
| STANDING_UP | recall@2.5 s · F1@25 · AP | 0.478 · 0.188 · 0.255 | 115 / 459 / 55 |
| LYING_DOWN | recall@2.5 s · F1@25 · AP | 0.381 · 0.176 · 0.203 | 97 / 301 / 37 |
| URINATION | recall@2.5 s · F1@25 · AP | 0.664 · 0.132 · 0.202 | 116 / 977 / 77 |
| DEFECATION | recall@2.5 s · F1@25 · AP | 0.780 · 0.073 · 0.048 | 50 / 992 / 39 |
| TAIL_RAISED | recall@2.5 s · F1@25 · AP | 0.625 · 0.033 · 0.314 | 32 / 1127 / 20 |
| MOUNTING / MOUNTED_BY | not_evaluable (zero annotated positives) | — | — |
| **selection_score** (the single model-selection objective) | | **0.387** | |

Across the five event heads the model produced 3,856 predicted intervals against 410
true ones — a ~9:1 over-prediction ratio. Recall is solid (0.38–0.78) and onset
localisation is good, but F1@25 (0.03–0.19) and AP (0.05–0.31) pay for the flood of
false intervals; thresholds selected on validation cows do not transfer to the two
data-heavy animals. These pooled figures read systematically optimistic and stay
diagnostics, not claims — see §3.

### Event-level diagnostics, cow-level bootstrap over the 8 test cows

Cow-level bootstrap 95% intervals in brackets (n = 8 unless noted). Every fold reports
`evidence_level: not_evaluable` — the gate requires ≥10 true events across ≥3 cows per
evaluation and each LOCO fold evaluates exactly one cow — so precision, F1 and
false-alarm rates stay **unclaimable**. The numbers below are diagnostics computed on
labelled stretches only and read systematically optimistic.

| event | recall (2.5 s tolerance) | F1@25 | edit score | onset median error |
|---|---|---|---|---|
| STANDING_UP | 0.460 [0.203, 0.737] | 0.368 [0.137, 0.630] | 52.5 [35.4, 71.4] | ~1.1 s (n=7) |
| LYING_DOWN | 0.488 [0.317, 0.625] | 0.370 [0.177, 0.564] | 49.9 [36.6, 65.0] | ~1.3 s (n=7) |
| URINATION | 0.745 [0.507, 0.940] | 0.444 [0.199, 0.704] | 38.4 [21.0, 56.4] | ~2.4 s (n=8) |
| DEFECATION | 0.703 [0.406, 0.906] | 0.385 [0.157, 0.659] | 42.8 [30.8, 55.4] | ~4.2 s (n=8) |
| TAIL_RAISED | 0.640 [0.374, 0.864] (n=7) | 0.108 [0.022, 0.220] | 41.8 [24.8, 60.9] | ~1.5 s (n=6) |

Reading: recall is solid (urination 75%, defecation 70% — the model finds most events),
but over-prediction collapses F1@25, and the failure concentrates on the two big-data
cows: URINATION on 21074-1 is 28 true intervals against **748** predicted ones,
DEFECATION 4 against 142, TAIL_RAISED on 20201-3 is 7 against 274. Their thresholds
were selected on the validation cows and do not transfer. This is exactly the
over-segmentation that motivated the 20260819 post-processing change, and it is the
quantity `review_coverage` exists to measure. Onset localisation is acceptable:
stand-up / lie-down median 0.2–3 s, urination 2.4 s, defecation 4.2 s.

<details>
<summary>Full per-event, per-cow table (40 rows)</summary>

| event | cow | true | pred | recall | F1@25 | edit | onset med (ms) |
|---|---|---|---|---|---|---|---|
| STANDING_UP | 23381-w1 | 3 | 0 | 0.000 | 0.000 | 50.0 | — |
| STANDING_UP | 23509-9 | 30 | 14 | 0.233 | 0.273 | 35.4 | 877 |
| STANDING_UP | 21074-1 | 23 | 249 | 0.913 | 0.132 | 26.0 | 5749 |
| STANDING_UP | 20201-3 | 12 | 55 | 0.250 | 0.090 | 35.8 | 187 |
| STANDING_UP | 24178-11 | 17 | 5 | 0.176 | 0.182 | 24.4 | 166 |
| STANDING_UP | 21100-10 | 5 | 1 | 0.200 | 0.333 | 61.1 | 261 |
| STANDING_UP | 23335-7 | 22 | 21 | 0.909 | 0.930 | 87.2 | 302 |
| STANDING_UP | 23489-8 | 3 | 3 | 1.000 | 1.000 | 100.0 | 141 |
| LYING_DOWN | 23381-w1 | 4 | 0 | 0.000 | 0.000 | 40.0 | — |
| LYING_DOWN | 23509-9 | 30 | 27 | 0.267 | 0.211 | 28.0 | 3063 |
| LYING_DOWN | 21074-1 | 18 | 134 | 0.667 | 0.092 | 41.3 | 1229 |
| LYING_DOWN | 20201-3 | 10 | 61 | 0.700 | 0.141 | 31.2 | 446 |
| LYING_DOWN | 24178-11 | 12 | 12 | 0.583 | 0.417 | 33.0 | 388 |
| LYING_DOWN | 21100-10 | 2 | 1 | 0.500 | 0.667 | 81.0 | 2869 |
| LYING_DOWN | 23335-7 | 19 | 15 | 0.684 | 0.765 | 70.0 | 368 |
| LYING_DOWN | 23489-8 | 2 | 1 | 0.500 | 0.667 | 75.0 | 442 |
| URINATION | 23381-w1 | 3 | 6 | 0.333 | 0.222 | 19.4 | 2690 |
| URINATION | 23509-9 | 40 | 17 | 0.100 | 0.035 | 19.6 | 8062 |
| URINATION | 21074-1 | 28 | 748 | 1.000 | 0.067 | 5.6 | 2146 |
| URINATION | 20201-3 | 12 | 153 | 0.667 | 0.085 | 13.9 | 5296 |
| URINATION | 24178-11 | 7 | 14 | 0.857 | 0.571 | 44.5 | 229 |
| URINATION | 21100-10 | 6 | 13 | 1.000 | 0.632 | 56.9 | 59 |
| URINATION | 23335-7 | 16 | 18 | 1.000 | 0.941 | 70.4 | 296 |
| URINATION | 23489-8 | 4 | 4 | 1.000 | 1.000 | 76.7 | 328 |
| DEFECATION | 23381-w1 | 4 | 4 | 0.250 | 0.250 | 21.7 | 375 |
| DEFECATION | 23509-9 | 2 | 43 | 1.000 | 0.089 | 25.0 | 8030 |
| DEFECATION | 21074-1 | 4 | 142 | 0.250 | 0.000 | 25.8 | 12769 |
| DEFECATION | 20201-3 | 8 | 16 | 0.125 | 0.083 | 60.7 | 5793 |
| DEFECATION | 24178-11 | 7 | 15 | 1.000 | 0.636 | 41.7 | 1702 |
| DEFECATION | 21100-10 | 1 | 9 | 1.000 | 0.200 | 36.1 | 492 |
| DEFECATION | 23335-7 | 21 | 23 | 1.000 | 0.818 | 62.2 | 3853 |
| DEFECATION | 23489-8 | 3 | 3 | 1.000 | 1.000 | 68.9 | 427 |
| TAIL_RAISED | 23381-w1 | 1 | 6 | 1.000 | 0.286 | 25.7 | 128 |
| TAIL_RAISED | 23509-9 | 3 | 62 | 0.667 | 0.031 | 28.9 | 644 |
| TAIL_RAISED | 21074-1 | 11 | 399 | 0.455 | 0.015 | 15.8 | 4729 |
| TAIL_RAISED | 20201-3 | 7 | 274 | 0.857 | 0.043 | 13.5 | 374 |
| TAIL_RAISED | 24178-11 | 0 | 23 | — | 0.000 | 30.2 | — |
| TAIL_RAISED | 21100-10 | 5 | 20 | 1.000 | 0.400 | 55.2 | 327 |
| TAIL_RAISED | 23335-7 | 4 | 42 | 0.500 | 0.087 | 85.3 | 2929 |
| TAIL_RAISED | 23489-8 | 1 | 7 | 0.000 | 0.000 | 80.0 | — |

</details>

Artifacts: `runs/loco_fold{1..8}/` (`best.pt`, `history.csv`, `report.json`,
`validation_predictions.csv`, `test_predictions.csv`); combined log
`runs/loco_training.log`. These stay local under the Git-ignored `runs/` tree.

## 2. GBDT retrain — compliant deploy candidate

The bundle trained at 18:22 the same day was non-compliant: its feature table covered
a single cow, `validation_cows` was empty, and every threshold degraded to the 0.5
default — which violates the cow-disjoint threshold-selection rule. The full chain was
rerun:

| Step | Value |
|---|---|
| Feature table | `runs/features_full/feature_table.parquet` — 351,127 rows × 120 features, `feature_version=2`, offline centred windows, calibration on, 19.4 s build |
| Training | xgboost GPU, 7 tasks, validation cows **23381-w1** and **23509-9** (cow-disjoint from training) |
| Thresholds | per-event, fitted on the validation cows, written into the bundle |

| Task | train pos | val pos | threshold |
|---|---|---|---|
| POSTURE_LYING | 133,510 | 1,139 | 0.998844 |
| WALKING | 8,217 | 2,346 | 0.000119 |
| STANDING_UP | 1,890 | 579 | 0.018815 |
| LYING_DOWN | 1,408 | 543 | 0.110056 |
| URINATION | 5,371 | 3,215 | 0.001529 |
| DEFECATION | 2,417 | 291 | 0.000905 |
| TAIL_RAISED | 8,876 | 3,663 | 0.009111 |
| MOUNTING / MOUNTED_BY | 0 positives | — | skipped by design |

Smoke test passed: the demo 60 s session yields 120 dense 2 Hz points and all seven
thresholds are read back unchanged by `COWMATA.predict`.

Artifact: `runs/gbdt_full/gbdt_full.joblib`, 6,079,950 bytes,
SHA-256 `f06350eece9b0c610a38c74b05eecbe0fbea140d82758b6863fe4b665fdaadfc`.

**It is not promoted to `weights/deploy/`.** The repo rule is that only reviewed
artifacts are promoted, and event precision is still `not_evaluable` without
`review_coverage`. `weights/deploy/gbdt_full.joblib` (feature_version 1, byte-identical
to the verified 20260818 artefact) remains the deployable model until a promotion
decision records the new bundle's verification evidence.

## 3. Evidence status — what may and may not be quoted

- **Event precision and false-alarm rates are not claimable.** `review_coverage` is
  still empty, so negatives are untrusted; the metric code reports `not_evaluable`
  rather than an over-optimistic number, and this report respects that.
- Every LOCO fold evaluates exactly **one** cow, below the ≥10-events-across-≥3-cows
  gate, so all event-level figures above are diagnostics on labelled stretches only
  and read systematically optimistic.
- `MOUNTING` / `MOUNTED_BY` have zero annotated intervals and are `not_evaluable`.
- The pooled deep-model figure (mean 0.538) is a mean of highly variable per-cow
  numbers (0.35–0.64) and must never be quoted without the per-cow breakdown.

## 4. Next steps

1. Collect `review_coverage` (exhaustive video review) — the only path to claimable
   precision and false-alarm rates, and the quantity that would turn §1's
   over-prediction diagnosis into a measurement.
2. Study per-cow / per-device adaptive thresholds: validation-cow thresholds clearly
   do not transfer to the two big-data cows.
3. Promotion decision for `runs/gbdt_full/gbdt_full.joblib` once review evidence exists.
